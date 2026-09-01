"""Third modeling approach (deep learning classifier), complementary to the
interpretable logistic baseline (``logistic_baseline.py``) and the CatBoost
/XGBoost tree ensemble (``catboost_fraud.py``).

The PyTorch autoencoder already in this repo (``autoencoder.py``) is an
*unsupervised* anomaly pre-filter -- it never sees fraud labels and only
contributes a single ``autoencoder_score`` feature. This module is a
*supervised* PyTorch classifier trained directly on the fraud label, using:

- **Focal Loss** instead of plain BCE, to concentrate gradient on the hard,
  still-misclassified minority-class (fraud) examples rather than being
  dominated by the ~99% of trivially-easy legit rows.
- A comparison across three activation functions (ReLU, GELU, Swish/SiLU) on
  identical data/splits/hyperparameters, so the choice is evidence-based
  rather than a default.

Same feature matrix (``FULL_FEATURE_COLUMNS`` from ``train.py``, including
the autoencoder anomaly score) and same ``evaluate()`` cost-sensitive metrics
as the other two approaches, so all three are directly comparable.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Binary focal loss: BCE weighted by ``(1 - p_t) ** gamma``, down-
    weighting already-easy (well-classified) examples so gradient focuses on
    the hard-to-separate fraud cases -- exactly where plain BCE struggles
    most under ~99% class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
}


class FraudMLP(nn.Module):
    def __init__(self, n_features: int, activation: str = "swish", hidden=(64, 32)):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_cls(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32))


def train_one_activation(
    activation: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[FraudMLP, list[dict]]:
    torch.manual_seed(seed)
    model = FraudMLP(n_features=X_train.shape[1], activation=activation)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t, y_val_t = _to_tensor(X_val), _to_tensor(y_val)
    history = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_loss": val_loss,
        })
    return model, history


def predict_proba(model: FraudMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_to_tensor(X))).numpy()


def compare_activations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 30,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Trains one MLP per activation function on identical data/epochs and
    returns (models, histories) keyed by activation name, so the caller can
    evaluate every variant with the shared ``evaluate()`` cost function."""
    models, histories = {}, {}
    for activation in ACTIVATIONS:
        model, history = train_one_activation(
            activation, X_train, y_train, X_val, y_val, n_epochs=n_epochs, seed=seed
        )
        models[activation] = model
        histories[activation] = history
    return models, histories
