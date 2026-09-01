import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deep_svdd import DeepSVDDNet, initialize_center, svdd_distance
from src.models.train_deep_svdd import train_deep_svdd


def test_forward_pass_shape():
    model = DeepSVDDNet(input_dim=10, latent_dim=4)
    x = torch.randn(32, 10)
    out = model(x)
    assert out.shape == (32, 4)


def test_layers_have_no_bias():
    model = DeepSVDDNet(input_dim=10, latent_dim=4)
    for module in model.net:
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None


def test_initialize_center_avoids_exact_zero():
    torch.manual_seed(0)
    model = DeepSVDDNet(input_dim=6, latent_dim=4)
    x = torch.randn(50, 6)
    center = initialize_center(model, x)
    assert (center.abs() >= 0.1 - 1e-6).all()
    assert torch.equal(model.center, center)


def test_svdd_distance_is_nonnegative():
    torch.manual_seed(0)
    model = DeepSVDDNet(input_dim=6, latent_dim=4)
    x = torch.randn(20, 6)
    initialize_center(model, x)
    dist = svdd_distance(model, x)
    assert (dist >= 0).all()
    assert dist.shape == (20,)


def test_training_reduces_distance_to_center_on_normal_data():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]

    model, historia, best_val_loss = train_deep_svdd(train, val, seed=0)
    assert best_val_loss <= historia["val_loss"].iloc[0]


def test_deep_svdd_flags_gross_outliers_after_training():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    normal = torch.tensor(rng.normal(loc=0, scale=1, size=(500, 6)).astype(np.float32))
    train, val = normal[:400], normal[400:]
    model, _, _ = train_deep_svdd(train, val, seed=0)

    outliers = torch.tensor(rng.normal(loc=15, scale=1, size=(20, 6)).astype(np.float32))
    dist_normal = svdd_distance(model, val).mean().item()
    dist_outliers = svdd_distance(model, outliers).mean().item()

    assert dist_outliers > dist_normal
