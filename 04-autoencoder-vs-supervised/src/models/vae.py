"""Variational Autoencoder (PyTorch) para deteccion de anomalias no
supervisada, alternativa probabilistica al autoencoder estandar de
`autoencoder.py`.

Igual que el autoencoder estandar, se entrena EXCLUSIVAMENTE con
transacciones normales. La diferencia es que el encoder no produce un
vector latente unico, sino los parametros (`mu`, `logvar`) de una
distribucion Gaussiana sobre el espacio latente, muestreada con el truco
de reparametrizacion (`z = mu + sigma * eps`, `eps ~ N(0, I)`) para que el
gradiente siga siendo diferenciable a traves del muestreo. La funcion de
perdida (ELBO negativo) combina el error de reconstruccion con una
divergencia KL que regulariza el espacio latente hacia una Gaussiana
estandar `N(0, I)` — esa regularizacion es lo que en teoria produce un
espacio latente mas suave y generativo que el de un autoencoder estandar,
a costo de una reconstruccion en promedio algo peor (el modelo no puede
memorizar libremente, tiene que "pagar" en KL por cada bit de informacion
que codifica).

El score de anomalia usado aguas abajo es el error de reconstruccion
determinista (decodificando `mu` directamente, sin muestrear) para que sea
comparable en las mismas unidades que el autoencoder estandar.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FraudVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 16), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 24), nn.ReLU(),
            nn.Linear(24, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
    kld_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO negativo por fila: MSE de reconstruccion + `kld_weight` * KL
    divergence hacia N(0, I). `kld_weight` (a veces llamado "beta", como en
    beta-VAE) pondera cuanto se sacrifica reconstruccion por regularizacion
    del espacio latente; 1.0 es el ELBO estandar sin ponderar."""
    recon = torch.sum((x - x_hat) ** 2, dim=1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon + kld_weight * kld, recon, kld


def reconstruction_error(model: FraudVAE, X: torch.Tensor) -> torch.Tensor:
    """Score de anomalia: MSE de reconstruccion determinista (decodificando
    `mu`, sin muestrear), en las mismas unidades que `autoencoder.reconstruction_error`
    para que ambos modelos sean comparables directamente."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X)
        X_hat = model.decode(mu)
        return torch.mean((X - X_hat) ** 2, dim=1)
