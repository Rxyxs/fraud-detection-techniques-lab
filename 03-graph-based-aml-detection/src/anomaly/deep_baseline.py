"""Enfoques de modelado complementarios al ensamble PyOD principal
(``ensemble_detector.py`` / ``transaction_scorer.py``), sobre las mismas
features transaccionales (``transaction_scorer.FEATURE_COLUMNS_TX``):

    (a) ``statistical_baseline_score`` -- baseline interpretable sin
        entrenamiento: suma de z-scores robustos (mediana/MAD, resistente a
        los propios outliers que se busca detectar) por feature. Sirve como
        piso de comparacion: si el ensamble/autoencoder no le gana a esto,
        no se justifica su complejidad adicional.
    (b) El ensamble de arboles (IsolationForest + COPOD + ECOD) ya existe en
        ``transaction_scorer.run_transaction_ensemble`` -- no se duplica
        aqui, solo se reutiliza en la comparacion (ver
        ``run_model_comparison.py``).
    (c) ``autoencoder_score`` -- autoencoder MLP en PyTorch entrenado sin
        etiquetas (reconstruye sus propias features), con el error de
        reconstruccion como score de anomalia. Permite comparar tres
        funciones de activacion (ReLU, GELU, Swish/SiLU) sobre el mismo
        dataset y arquitectura.

Ninguno de los tres usa ``es_ilicito``/``tipologia`` como input: esa columna
solo se usa despues, en ``evaluate_score``, para medir desempenio -- mismo
patron metodologico que el resto del repo.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from src.anomaly.transaction_scorer import FEATURE_COLUMNS_TX

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU(x) = x * sigmoid(x) == Swish
}


def _feature_matrix(tx_features: pl.DataFrame) -> np.ndarray:
    return tx_features.select(FEATURE_COLUMNS_TX).to_numpy().astype(float)


def statistical_baseline_score(tx_features: pl.DataFrame, contamination: float = 0.01) -> pl.DataFrame:
    """Baseline interpretable: suma de z-scores robustos (mediana/MAD) sobre
    las features transaccionales. Sin entrenamiento ni hiperparametros mas
    alla del presupuesto de alertas -- referencia minima e interpretable
    (cada z-score explica cuanto aporta cada feature) contra la que se mide
    el valor agregado del ensamble y del autoencoder."""
    X = _feature_matrix(tx_features)
    mediana = np.median(X, axis=0)
    mad = np.median(np.abs(X - mediana), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)
    z = np.abs(X - mediana) / (1.4826 * mad)  # 1.4826 -> MAD consistente con std bajo normalidad
    score = z.sum(axis=1)

    n_alertas = max(1, int(np.ceil(contamination * X.shape[0])))
    umbral = np.sort(score)[::-1][n_alertas - 1]

    return tx_features.with_columns(pl.Series("score_baseline", score)).with_columns(
        (pl.col("score_baseline") >= umbral).alias("alerta_baseline")
    )


class _Autoencoder(nn.Module):
    """MLP encoder-decoder simetrico y pequenio: el objetivo no es capacidad
    sino comparar el efecto de la funcion de activacion sobre el mismo
    presupuesto de parametros."""

    def __init__(self, n_features: int, activation: str = "relu", hidden: int = 16, bottleneck: int = 4):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), act_cls(),
            nn.Linear(hidden, bottleneck), act_cls(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), act_cls(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 2000  # epocas full-batch: cada una es un solo paso de gradiente
PATIENCE = 50
VAL_FRACTION = 0.15
CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints"


def _fingerprint(X: np.ndarray, **config) -> str:
    h = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _train_autoencoder(
    X: np.ndarray,
    activation: str,
    epochs: int = MAX_EPOCHS,
    lr: float = 1e-3,
    seed: int = 42,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
    return_history: bool = False,
):
    """Entrena el autoencoder sin etiquetas. Un ``VAL_FRACTION`` aleatorio de
    las transacciones (sin mirar ``es_ilicito``) se reserva para medir el
    error de reconstruccion fuera de muestra: al menos ``min_epochs`` epocas
    y desde ahi early stopping tras ``patience`` epocas sin mejora, con
    restauracion de la mejor epoca. El estado se guarda cada
    ``CHECKPOINT_EVERY`` epocas para retomar una corrida interrumpida. El
    score final es el error de reconstruccion de *todas* las transacciones."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(X.shape[0] * VAL_FRACTION))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    mu, sigma = X[fit_idx].mean(axis=0), X[fit_idx].std(axis=0)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)
    Xs = (X - mu) / sigma
    tensor = torch.tensor(Xs, dtype=torch.float32)
    fit_t, val_t = tensor[fit_idx], tensor[val_idx]

    model = _Autoencoder(n_features=X.shape[1], activation=activation)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    fp = _fingerprint(X, activation=activation, epochs=epochs, lr=lr, seed=seed, min_epochs=min_epochs,
                      patience=patience)
    state = {"epoch": 0, "train_loss": [], "val_loss": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": opt.state_dict(), "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < epochs:
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(fit_t), fit_t)
        loss.backward()
        opt.step()
        state["epoch"] += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_t), val_t).item())
        state["train_loss"].append(float(loss.item()))
        state["val_loss"].append(val_loss)
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        if state["epoch"] % CHECKPOINT_EVERY == 0:
            save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()

    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        error = ((recon - tensor) ** 2).mean(dim=1).numpy()
    if return_history:
        history = {k: state[k] for k in ("train_loss", "val_loss", "best_epoch")} | {"epochs_run": state["epoch"]}
        return error, history
    return error


def autoencoder_score(
    tx_features: pl.DataFrame,
    activation: str = "relu",
    contamination: float = 0.01,
    epochs: int = MAX_EPOCHS,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    """Entrena un autoencoder no supervisado (reconstruye sus propias
    features estandarizadas) y usa el error de reconstruccion cuadratico
    medio como score de anomalia: cuanto peor reconstruye una transaccion,
    mas se aleja del patron "normal" mayoritario que domino el entrenamiento."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation debe ser una de {list(ACTIVATIONS)}, recibido: {activation!r}")

    X = _feature_matrix(tx_features)
    ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"autoencoder_{activation}.ckpt"
    error = _train_autoencoder(X, activation=activation, epochs=epochs, checkpoint_path=ckpt)

    n_alertas = max(1, int(np.ceil(contamination * X.shape[0])))
    umbral = np.sort(error)[::-1][n_alertas - 1]

    col_score = f"score_autoencoder_{activation}"
    col_alerta = f"alerta_autoencoder_{activation}"
    return tx_features.with_columns(pl.Series(col_score, error)).with_columns(
        (pl.col(col_score) >= umbral).alias(col_alerta)
    )


def evaluate_score(tx_features_scored: pl.DataFrame, score_col: str, alerta_col: str) -> dict:
    """Mismo patron de evaluacion que ``transaction_scorer.evaluate_tx_against_ground_truth``,
    generalizado a cualquier columna de score/alerta -- para poder comparar
    baseline, ensamble y autoencoder con una sola funcion."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = tx_features_scored["es_ilicito"].to_numpy().astype(int)
    y_score = tx_features_scored[score_col].to_numpy()

    n_alertas_correctas = int(
        tx_features_scored.filter(pl.col(alerta_col) & (pl.col("es_ilicito") == 1)).height
    )
    n_alertas = int(tx_features_scored.filter(pl.col(alerta_col)).height)
    n_positivos = int(y_true.sum())

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if 0 < n_positivos < len(y_true) else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if n_positivos > 0 else float("nan"),
        "n_alertas": n_alertas,
        "n_alertas_correctas": n_alertas_correctas,
        "precision_en_alertas": (n_alertas_correctas / n_alertas) if n_alertas > 0 else float("nan"),
        "recall_sobre_verdad_terreno": (n_alertas_correctas / n_positivos) if n_positivos > 0 else float("nan"),
    }
