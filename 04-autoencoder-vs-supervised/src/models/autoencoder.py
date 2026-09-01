"""Arquitectura del autoencoder (PyTorch) para deteccion de anomalias no
supervisada.

Se entrena EXCLUSIVAMENTE con transacciones normales: aprende a
reconstruir el patron "legitimo" y, por construccion, reconstruye peor
una transaccion que no se parezca a ese patron. El error de reconstruccion
(MSE por fila) es el score de anomalia — sin ninguna etiqueta de fraude
involucrada en el ajuste de los pesos.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, bottleneck_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16), nn.ReLU(),
            nn.Linear(16, 24), nn.ReLU(),
            nn.Linear(24, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(model: FraudAutoencoder, X: torch.Tensor) -> torch.Tensor:
    """MSE de reconstruccion por fila (no reducido), usado como score de anomalia."""
    model.eval()
    with torch.no_grad():
        X_hat = model(X)
        return torch.mean((X - X_hat) ** 2, dim=1)
