import numpy as np

from src.models.autoencoder import FraudAutoencoder, reconstruction_error, train_autoencoder


def test_forward_pass_shape():
    model = FraudAutoencoder(n_features=6, latent_dim=3)
    x = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    import torch

    out = model(torch.tensor(x))
    assert out.shape == (10, 6)


def test_reconstruction_error_is_nonnegative():
    model = FraudAutoencoder(n_features=5, latent_dim=2)
    x = np.random.default_rng(1).normal(size=(20, 5))
    errors = reconstruction_error(model, x)
    assert errors.shape == (20,)
    assert np.all(errors >= 0)


def test_training_reduces_reconstruction_loss():
    rng = np.random.default_rng(42)
    # A tight cluster is trivially learnable by a small autoencoder.
    X = rng.normal(loc=0.0, scale=1.0, size=(500, 4))

    artifacts = train_autoencoder(X, n_features=4, latent_dim=2, epochs=20, batch_size=64)
    losses = artifacts.train_losses

    assert losses[-1] < losses[0]

    errors_normal = reconstruction_error(artifacts.model, X)
    # An out-of-distribution point should reconstruct far worse than in-distribution data.
    outlier = np.full((1, 4), 50.0)
    errors_outlier = reconstruction_error(artifacts.model, outlier)
    assert errors_outlier[0] > errors_normal.mean() * 5
