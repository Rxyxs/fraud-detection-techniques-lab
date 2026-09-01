import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import FraudAutoencoder, reconstruction_error
from src.models.train_autoencoder import train_autoencoder


def test_forward_pass_shape():
    model = FraudAutoencoder(input_dim=10, bottleneck_dim=4)
    x = torch.randn(32, 10)
    out = model(x)
    assert out.shape == x.shape


def test_reconstruction_error_is_nonnegative_and_shaped():
    model = FraudAutoencoder(input_dim=10, bottleneck_dim=4)
    x = torch.randn(16, 10)
    err = reconstruction_error(model, x)
    assert err.shape == (16,)
    assert (err >= 0).all()


def test_reconstruction_error_zero_for_identity_like_data():
    torch.manual_seed(0)
    model = FraudAutoencoder(input_dim=5, bottleneck_dim=5)
    x = torch.zeros(8, 5)
    err = reconstruction_error(model, x)
    # sin entrenar, un input de puros ceros deberia dar un error acotado (no NaN/inf)
    assert torch.isfinite(err).all()


def test_training_reduces_reconstruction_error_on_normal_data():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]

    model, historia, best_val_loss = train_autoencoder(train, val, seed=0)
    err_after = reconstruction_error(model, val).mean().item()

    assert best_val_loss < historia["val_loss"].iloc[0]
    assert err_after < 1.0  # deberia aprender a reconstruir datos normales razonablemente bien


def test_autoencoder_flags_gross_outliers_after_training():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]
    model, _, _ = train_autoencoder(train, val, seed=0)

    outliers = torch.tensor(rng.normal(loc=15, scale=1, size=(20, 6)).astype(np.float32))
    err_normal = reconstruction_error(model, val).mean().item()
    err_outliers = reconstruction_error(model, outliers).mean().item()

    assert err_outliers > err_normal * 5
