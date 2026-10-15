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
- At least 100 epochs, then early stopping on the time-based validation
  split with the best epoch restored, and per-epoch checkpoints so an
  interrupted run resumes.

Same feature matrix (``FULL_FEATURE_COLUMNS`` from ``train.py``, including
the autoencoder anomaly score) and same ``evaluate()`` cost-sensitive metrics
as the other two approaches, so all three are directly comparable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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


MIN_EPOCHS = 100  # no network stops before 100 epochs
MAX_EPOCHS = 300
PATIENCE = 15     # epochs without validation improvement before stopping (only after the minimum)
LR_PATIENCE = 5


def _fingerprint(*arrays, **config) -> str:
    """Hash of data + config: a checkpoint is only resumed when it matches."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train_one_activation(
    activation: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = MAX_EPOCHS,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
) -> tuple[FraudMLP, list[dict]]:
    """Trains at least ``min_epochs`` epochs (or ``n_epochs`` if smaller), then
    early-stops once validation loss has not improved for ``patience`` epochs
    and restores the best epoch's weights (flagged ``is_best`` in the
    history). The learning rate halves when validation stalls. With
    ``checkpoint_path`` the full state is saved every epoch (atomic write), so
    an interrupted run resumes where it stopped."""
    torch.manual_seed(seed)
    model = FraudMLP(n_features=X_train.shape[1], activation=activation)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=LR_PATIENCE)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    generator = torch.Generator()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    X_val_t, y_val_t = _to_tensor(X_val), _to_tensor(y_val)

    fp = _fingerprint(X_train, y_train, X_val, y_val, activation=activation, n_epochs=n_epochs,
                      batch_size=batch_size, lr=lr, seed=seed, min_epochs=min_epochs, patience=patience)
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
        tmp.replace(path)

    while not state["done"] and state["epoch"] < n_epochs:
        generator.manual_seed(seed + state["epoch"])  # same batch order when resuming
        torch.manual_seed(seed + 10_000 + state["epoch"])  # reproducible dropout masks
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


def predict_proba(model: FraudMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_to_tensor(X))).numpy()


def compare_activations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = MAX_EPOCHS,
    seed: int = 42,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Trains one MLP per activation function on identical data and returns
    (models, histories) keyed by activation name, each model holding the
    weights of its best validation epoch."""
    models, histories = {}, {}
    for activation in ACTIVATIONS:
        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"mlp_focal_{activation}.ckpt"
        model, history = train_one_activation(
            activation, X_train, y_train, X_val, y_val, n_epochs=n_epochs, seed=seed,
            min_epochs=min_epochs, patience=patience, checkpoint_path=ckpt,
        )
        models[activation] = model
        histories[activation] = history
    return models, histories
