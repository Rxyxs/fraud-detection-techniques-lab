"""Entrena el autoencoder no supervisado sobre transacciones normales
unicamente, calibra umbrales de deteccion contra el set de validacion
(tambien solo-normal) y puntua el test set completo (normal + fraude).

Uso:
    python -m src.models.train_autoencoder
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_raw, make_splits, normal_only
from src.models.autoencoder import FraudAutoencoder, reconstruction_error

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RANDOM_STATE = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 8
LEARNING_RATE = 1e-3
BOTTLENECK_DIM = 8
THRESHOLD_PERCENTILES = [90, 95, 97, 99, 99.5]


def _to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.to_numpy(dtype=np.float32))


def train_autoencoder(X_train_normal: torch.Tensor, X_val_normal: torch.Tensor, seed: int = RANDOM_STATE):
    torch.manual_seed(seed)
    model = FraudAutoencoder(input_dim=X_train_normal.shape[1], bottleneck_dim=BOTTLENECK_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train_normal), batch_size=BATCH_SIZE, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))

    best_val_loss = float("inf")
    best_state = None
    epochs_sin_mejora = 0
    historia = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * batch.size(0)
        train_loss = train_loss_total / len(X_train_normal)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_normal), X_val_normal).item()
        historia.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"  Early stopping en epoch {epoch} (mejor val_loss={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(historia), best_val_loss


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset real y generando particiones...")
    df = load_raw()
    splits = make_splits(df)

    X_train_normal = normal_only(splits.X_train, splits.y_train)
    X_val_normal = normal_only(splits.X_val, splits.y_val)
    print(f"Train (solo normales): {len(X_train_normal):,} | Val (solo normales): {len(X_val_normal):,}")
    print(f"Test (normal + fraude, nunca visto): {len(splits.X_test):,} "
          f"({int(splits.y_test.sum())} fraudes)")

    X_train_t = _to_tensor(X_train_normal)
    X_val_t = _to_tensor(X_val_normal)
    X_test_t = _to_tensor(splits.X_test)

    print("Entrenando autoencoder (solo con transacciones normales)...")
    model, historia, best_val_loss = train_autoencoder(X_train_t, X_val_t)
    print(f"Mejor val_loss (MSE reconstruccion, solo normales): {best_val_loss:.6f}")

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    historia.to_csv(REPORTS_DIR / "autoencoder_training_history.csv", index=False)

    val_errors = reconstruction_error(model, X_val_t).numpy()
    test_errors = reconstruction_error(model, X_test_t).numpy()

    umbrales = {f"p{p}": float(np.percentile(val_errors, p)) for p in THRESHOLD_PERCENTILES}
    print("\nUmbrales candidatos (percentiles del error de reconstruccion en validacion, solo normales):")
    for k, v in umbrales.items():
        print(f"  {k}: {v:.6f}")

    resultados_test = pd.DataFrame({
        "reconstruction_error": test_errors,
        "y_true": splits.y_test.to_numpy(),
    })
    resultados_test.to_parquet(REPORTS_DIR / "autoencoder_test_scores.parquet")

    import json
    with open(REPORTS_DIR / "autoencoder_thresholds.json", "w") as f:
        json.dump(umbrales, f, indent=2)

    print(f"\nArtefactos escritos en {MODELS_DIR} y {REPORTS_DIR}")
    return model, resultados_test, umbrales


if __name__ == "__main__":
    main()
