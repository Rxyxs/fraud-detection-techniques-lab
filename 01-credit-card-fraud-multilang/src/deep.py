"""Cuarto -- en realidad tercer enfoque *complementario* -- de modelado:
MLP en PyTorch con Focal Loss (Lin et al. 2017), disenada para clases
desbalanceadas/dificiles (aunque este dataset esta balanceado 50/50, Focal
Loss sigue siendo el disclosure correcto para fraude: penaliza mas los
ejemplos dificiles de clasificar, no solo los de la clase minoritaria) mas
una comparacion de funciones de activacion (ReLU vs. GELU vs. Swish/SiLU)
sobre la misma arquitectura y el mismo split, para dejar evidencia empirica
de cual converge mejor en este problema -- no solo asumirlo.

    python -m src.deep
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
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


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    activation_name: str,
    n_epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> tuple[FraudMLP, list[dict]]:
    """Entrena una MLP con Focal Loss usando la activacion dada. Devuelve el
    modelo entrenado y el historial de loss/epoch (train y validacion) para
    graficar curvas de convergencia."""
    torch.manual_seed(RANDOM_STATE)

    X_train_t, y_train_t = _to_tensor(X_train, y_train)
    X_val_t, y_val_t = _to_tensor(X_val, y_val)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = FraudMLP(n_features=X_train_t.shape[1], activation=ACTIVATIONS[activation_name])
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        history.append({
            "activation": activation_name,
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_loss": val_loss,
        })

    return model, history


@torch.no_grad()
def predict_proba(model: FraudMLP, X) -> np.ndarray:
    model.eval()
    X_t = _to_tensor(X)
    logits = model(X_t)
    return torch.sigmoid(logits).numpy()


def compare_activations(
    X_train, y_train, X_val, y_val, n_epochs: int = 15,
) -> tuple[dict[str, FraudMLP], pd.DataFrame]:
    """Entrena la misma arquitectura MLP+FocalLoss con ReLU, GELU y Swish
    sobre el mismo split, devolviendo los modelos y el historial de loss
    combinado (para graficar curvas de convergencia comparadas)."""
    models: dict[str, FraudMLP] = {}
    all_history: list[dict] = []
    for name in ACTIVATIONS:
        model, history = train_mlp(X_train, y_train, X_val, y_val, activation_name=name, n_epochs=n_epochs)
        models[name] = model
        all_history.extend(history)
    return models, pd.DataFrame(all_history)
