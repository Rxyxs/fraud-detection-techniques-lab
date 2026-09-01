import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train_vae import train_vae
from src.models.vae import FraudVAE, reconstruction_error, vae_loss


def test_forward_pass_shapes():
    model = FraudVAE(input_dim=10, latent_dim=4)
    x = torch.randn(32, 10)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape == (32, 4)
    assert logvar.shape == (32, 4)


def test_vae_loss_is_nonnegative_and_shaped():
    model = FraudVAE(input_dim=8, latent_dim=4)
    x = torch.randn(16, 8)
    x_hat, mu, logvar = model(x)
    loss, recon, kld = vae_loss(x, x_hat, mu, logvar)
    assert loss.shape == (16,)
    assert (recon >= 0).all()


def test_reconstruction_error_is_deterministic():
    torch.manual_seed(0)
    model = FraudVAE(input_dim=6, latent_dim=3)
    x = torch.randn(10, 6)
    err1 = reconstruction_error(model, x)
    err2 = reconstruction_error(model, x)
    assert torch.allclose(err1, err2)


def test_training_reduces_val_loss_on_normal_data():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]

    model, historia, best_val_loss = train_vae(train, val, seed=0)
    assert best_val_loss <= historia["val_loss"].iloc[0]


def test_vae_flags_gross_outliers_after_training():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]
    model, _, _ = train_vae(train, val, seed=0)

    outliers = torch.tensor(rng.normal(loc=15, scale=1, size=(20, 6)).astype(np.float32))
    err_normal = reconstruction_error(model, val).mean().item()
    err_outliers = reconstruction_error(model, outliers).mean().item()

    assert err_outliers > err_normal * 5
