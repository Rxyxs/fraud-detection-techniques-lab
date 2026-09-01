"""PyTorch autoencoder used as an unsupervised pre-filter: it learns the
manifold of *normal* transaction behaviour, and its reconstruction error is
fed into the supervised classifier as an extra "anomaly score" feature. This
lets the pipeline flag anomalous patterns that don't look like the specific
fraud examples seen during training -- useful given how few labeled fraud
rows exist under < 1% prevalence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FraudAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 4):
        super().__init__()
        # LeakyReLU rather than plain ReLU: inputs are StandardScaler-scaled
        # (mean 0), so roughly half of every pre-activation is negative, and a
        # 4-unit bottleneck has little capacity to spare -- ReLU zeroing out
        # all of that would starve the encoder, not just the decoder's output.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(8, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(16, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class AutoencoderArtifacts:
    model: FraudAutoencoder
    train_losses: list[float]


def train_autoencoder(
    X_train_normal: np.ndarray,
    n_features: int,
    latent_dim: int = 4,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> AutoencoderArtifacts:
    """Trains on (already scaled) rows believed to be non-fraud only, so the
    reconstruction error is meaningful as an anomaly score at inference time.
    """
    torch.manual_seed(seed)

    model = FraudAutoencoder(n_features=n_features, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    losses = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return AutoencoderArtifacts(model=model, train_losses=losses)


@torch.no_grad()
def reconstruction_error(model: FraudAutoencoder, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Per-row mean squared reconstruction error -- higher means more
    anomalous relative to the learned "normal" manifold."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    reconstruction = model(X_tensor)
    error = torch.mean((reconstruction - X_tensor) ** 2, dim=1)
    return error.cpu().numpy()
