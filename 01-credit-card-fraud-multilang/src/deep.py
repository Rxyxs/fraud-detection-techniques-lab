"""Cuarto -- en realidad tercer enfoque *complementario* -- de modelado:
MLP en PyTorch con Focal Loss (Lin et al. 2017), disenada para clases
desbalanceadas/dificiles (aunque este dataset esta balanceado 50/50, Focal
Loss sigue siendo el disclosure correcto para fraude: penaliza mas los
ejemplos dificiles de clasificar, no solo los de la clase minoritaria) mas
una comparacion de funciones de activacion (ReLU vs. GELU vs. Swish/SiLU)
sobre la misma arquitectura y el mismo split, para dejar evidencia empirica
de cual converge mejor en este problema -- no solo asumirlo.

Entrenamiento: minimo 100 epocas, early stopping sobre una validacion
separada del train (nunca el test), restauracion de la mejor epoca y
checkpoints por epoca para retomar una corrida interrumpida.

    python -m src.deep
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 300
PATIENCE = 15     # epocas sin mejora en validacion antes de cortar (solo despues del minimo)
LR_PATIENCE = 5
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU == Swish (beta=1)
}


class FocalLoss(nn.Module):
    """Focal Loss binaria: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t).
    gamma>0 reduce el peso relativo de ejemplos faciles (bien clasificados),
    concentrando el gradiente en los dificiles -- estandar en deteccion de
    objetos/fraude bajo desbalance de clases."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class FraudMLP(nn.Module):
    def __init__(self, n_features: int, activation: type[nn.Module], hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), activation(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(X, y=None):
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    if y is None:
        return X_t
    return X_t, torch.tensor(np.asarray(y, dtype=np.float32))


def _fingerprint(*arrays, **config) -> str:
    """Huella de datos + configuracion: un checkpoint solo se retoma si coincide."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    activation_name: str,
    n_epochs: int = MAX_EPOCHS,
    batch_size: int = 512,
    lr: float = 1e-3,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
) -> tuple[FraudMLP, list[dict]]:
    """Entrena una MLP con Focal Loss usando la activacion dada. Devuelve el
    modelo entrenado y el historial de loss/epoch (train y validacion) para
    graficar curvas de convergencia.

    Entrena al menos ``min_epochs`` epocas (o ``n_epochs`` si es menor); desde
    ahi, early stopping corta cuando la loss de validacion no mejora en
    ``patience`` epocas y se restauran los pesos de la mejor epoca (marcada con
    ``is_best`` en el historial). La tasa de aprendizaje baja a la mitad cuando
    la validacion se estanca. Con ``checkpoint_path`` el estado completo se
    guarda cada epoca y una corrida interrumpida retoma donde quedo.
    """
    torch.manual_seed(RANDOM_STATE)

    X_train_t, y_train_t = _to_tensor(X_train, y_train)
    X_val_t, y_val_t = _to_tensor(X_val, y_val)

    generator = torch.Generator()
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=generator)

    model = FraudMLP(n_features=X_train_t.shape[1], activation=ACTIVATIONS[activation_name])
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=LR_PATIENCE)

    fp = _fingerprint(X_train, y_train, X_val, y_val, activation=activation_name, n_epochs=n_epochs,
                      batch_size=batch_size, lr=lr, min_epochs=min_epochs, patience=patience)
    state = {"epoch": 0, "history": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, tmp)
        tmp.replace(path)  # escritura atomica: un corte no deja el checkpoint a medias

    while not state["done"] and state["epoch"] < n_epochs:
        generator.manual_seed(RANDOM_STATE + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(RANDOM_STATE + 10_000 + state["epoch"])  # mascaras de dropout reproducibles
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        state["epoch"] += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        state["history"].append({
            "activation": activation_name,
            "epoch": state["epoch"],
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_loss": val_loss,
        })
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, n_epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()

    history = [dict(row, is_best=row["epoch"] == state["best_epoch"]) for row in state["history"]]
    return model, history


@torch.no_grad()
def predict_proba(model: FraudMLP, X) -> np.ndarray:
    model.eval()
    X_t = _to_tensor(X)
    logits = model(X_t)
    return torch.sigmoid(logits).numpy()


def compare_activations(
    X_train, y_train, X_val, y_val, n_epochs: int = MAX_EPOCHS, min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE, checkpoint_dir: Path | None = None,
) -> tuple[dict[str, FraudMLP], pd.DataFrame]:
    """Entrena la misma arquitectura MLP+FocalLoss con ReLU, GELU y Swish
    sobre el mismo split, devolviendo los modelos (con los pesos de su mejor
    epoca de validacion) y el historial de loss combinado (para graficar
    curvas de convergencia comparadas)."""
    models: dict[str, FraudMLP] = {}
    all_history: list[dict] = []
    for name in ACTIVATIONS:
        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"mlp_{name}.ckpt"
        model, history = train_mlp(X_train, y_train, X_val, y_val, activation_name=name, n_epochs=n_epochs,
                                   min_epochs=min_epochs, patience=patience, checkpoint_path=ckpt)
        models[name] = model
        all_history.extend(history)
    return models, pd.DataFrame(all_history)
