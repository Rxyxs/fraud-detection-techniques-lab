"""Bucle de entrenamiento compartido por el autoencoder, el VAE y Deep SVDD.

Los tres modelos se entrenan igual -- solo cambia la loss -- asi que el
protocolo vive en un solo lugar:

- Al menos ``MIN_EPOCHS`` epocas; desde ahi, early stopping cuando la loss
  de validacion (solo transacciones normales) no mejora en ``PATIENCE``
  epocas, como maximo ``MAX_EPOCHS``. Se restauran los pesos de la mejor
  epoca.
- Checkpoint del estado completo (pesos, optimizador, historia, mejor
  modelo) cada epoca, con escritura atomica. Si el proceso se corta, la
  siguiente corrida con los mismos datos y configuracion retoma desde la
  ultima epoca guardada; un checkpoint de otros datos no se reutiliza.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

MIN_EPOCHS = 100
MAX_EPOCHS = 300
PATIENCE = 15


def _fingerprint(*tensors: torch.Tensor, **config) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def fit(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    batch_loss: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    val_metrics: Callable[[torch.nn.Module, torch.Tensor], dict],
    lr: float,
    batch_size: int,
    seed: int,
    min_epochs: int = MIN_EPOCHS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
    config: dict | None = None,
) -> tuple[torch.nn.Module, pd.DataFrame, float]:
    """Entrena ``model`` y devuelve (modelo con los pesos de la mejor epoca,
    historia por epoca, mejor loss de validacion). ``batch_loss`` devuelve la
    loss escalar de un batch; ``val_metrics`` devuelve un dict con al menos
    ``val_loss`` sobre el set de validacion completo."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True, generator=generator)

    fp = _fingerprint(X_train, X_val, lr=lr, batch_size=batch_size, seed=seed, min_epochs=min_epochs,
                      max_epochs=max_epochs, patience=patience, model=type(model).__name__, **(config or {}))
    state = {"epoch": 0, "history": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]
            print(f"  Retomando desde el checkpoint en la epoca {state['epoch']}")

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < max_epochs:
        generator.manual_seed(seed + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(seed + 10_000 + state["epoch"])  # mismo ruido (reparametrizacion del VAE) al retomar
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        state["epoch"] += 1

        model.eval()
        with torch.no_grad():
            metrics = val_metrics(model, X_val)
        val_loss = metrics["val_loss"]
        state["history"].append({"epoch": state["epoch"], "train_loss": total / len(X_train), **metrics})

        if val_loss < state["best_val"] - 1e-6:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, max_epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
            print(f"  Early stopping en epoch {state['epoch']} "
                  f"(mejor epoca {state['best_epoch']}, val_loss={state['best_val']:.6f})")
        save_checkpoint()

    model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    history = pd.DataFrame(state["history"])
    history["is_best"] = history["epoch"] == state["best_epoch"]
    return model, history, state["best_val"]
