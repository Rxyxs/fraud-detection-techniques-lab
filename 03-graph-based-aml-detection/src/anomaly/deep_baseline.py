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


def _train_autoencoder(X: np.ndarray, activation: str, epochs: int = 60, lr: float = 1e-3, seed: int = 42) -> np.ndarray:
    torch.manual_seed(seed)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)
    Xs = (X - mu) / sigma

    tensor = torch.tensor(Xs, dtype=torch.float32)
    model = _Autoencoder(n_features=X.shape[1], activation=activation)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(tensor)
        loss = loss_fn(recon, tensor)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        error = ((recon - tensor) ** 2).mean(dim=1).numpy()
    return error


def autoencoder_score(
    tx_features: pl.DataFrame,
    activation: str = "relu",
    contamination: float = 0.01,
    epochs: int = 60,
) -> pl.DataFrame:
    """Entrena un autoencoder no supervisado (reconstruye sus propias
    features estandarizadas) y usa el error de reconstruccion cuadratico
    medio como score de anomalia: cuanto peor reconstruye una transaccion,
    mas se aleja del patron "normal" mayoritario que domino el entrenamiento."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation debe ser una de {list(ACTIVATIONS)}, recibido: {activation!r}")

    X = _feature_matrix(tx_features)
    error = _train_autoencoder(X, activation=activation, epochs=epochs)

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
