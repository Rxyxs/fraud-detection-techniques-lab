"""Deep SVDD (Ruff et al., 2018) para deteccion de anomalias no supervisada:
tercera alternativa a la reconstruccion (autoencoder / VAE) en este motor.

En vez de aprender a reconstruir la entrada, Deep SVDD entrena una red que
mapea las transacciones normales a un espacio latente donde queden lo mas
apretadas posible alrededor de un centro fijo `c` (la hiperesfera de menor
volumen que las encierra) — la adaptacion profunda de One-Class SVM /
Support Vector Data Description. El objetivo de entrenamiento es
simplemente minimizar la distancia cuadratica media al centro:

    L = (1/n) * sum_i || phi(x_i) - c ||^2

sin ningun termino de reconstruccion. Una transaccion anomala, al no
parecerse a los patrones normales vistos en entrenamiento, cae lejos del
centro en el espacio latente — esa distancia es el score de anomalia.

Dos detalles de implementacion no triviales, ambos del paper original:
1. Las capas de la red **no llevan bias** (`bias=False`): un bias permite a
   la red aprender una solucion trivial degenerada ("hypersphere collapse")
   donde todos los pesos y el centro colapsan a un punto constante,
   logrando distancia cero sin haber aprendido nada util del input.
2. El centro `c` **no es un parametro entrenable**: se fija una sola vez,
   antes de entrenar, como el promedio de las salidas de la red (sin
   inicializar) sobre el set de entrenamiento, con las dimensiones
   cercanas a cero desplazadas a `+-eps` (si `c` pudiera moverse junto con
   los pesos, la red podria "perseguir" a `c` hacia una solucion trivial
   en vez de organizar el espacio latente alrededor de un punto fijo).
"""

from __future__ import annotations

import torch
import torch.nn as nn

CENTER_EPS = 0.1


class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24, bias=False), nn.ReLU(),
            nn.Linear(24, 16, bias=False), nn.ReLU(),
            nn.Linear(16, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def initialize_center(model: DeepSVDDNet, X: torch.Tensor, eps: float = CENTER_EPS) -> torch.Tensor:
    """Fija `model.center` como el promedio de las salidas de la red (con
    sus pesos de inicializacion aleatoria) sobre `X`, con las dimensiones
    cercanas a cero desplazadas a `+-eps` para evitar el colapso trivial."""
    model.eval()
    outputs = model(X)
    center = outputs.mean(dim=0)
    center[(center.abs() < eps) & (center >= 0)] = eps
    center[(center.abs() < eps) & (center < 0)] = -eps
    model.center.copy_(center)
    return center


def svdd_distance(model: DeepSVDDNet, X: torch.Tensor) -> torch.Tensor:
    """Score de anomalia: distancia euclidiana al cuadrado entre la
    proyeccion de cada fila y el centro fijo de la hiperesfera."""
    model.eval()
    with torch.no_grad():
        return torch.sum((model(X) - model.center) ** 2, dim=1)
