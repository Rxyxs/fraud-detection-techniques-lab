"""Optimizacion de umbral sensible a costos (cost-sensitive thresholding).

Los percentiles usados en `train_*.py` (p95, p99, ...) son puntos de
partida razonables, pero son arbitrarios desde el punto de vista de
negocio: no dicen nada sobre cuanto cuesta, en dolares, cada tipo de error.
Este modulo reemplaza esa eleccion arbitraria por una busqueda que minimiza
el costo financiero esperado, usando una matriz de perdida asimetrica y
realista para fraude de tarjetas:

- **Costo de un Falso Positivo (FP)**: una transaccion legitima marcada
  como sospechosa. El costo no es el monto de la transaccion (el cliente no
  pierde su dinero) sino el costo *operativo* de revisarla — tiempo de un
  analista de fraude, y en casos donde se bloquea la tarjeta, friccion y
  riesgo de perder al cliente. Se modela como un costo fijo por alerta
  (`DEFAULT_COST_FALSE_POSITIVE`, en USD).
- **Costo de un Falso Negativo (FN)**: un fraude real que no se detecta. A
  diferencia del FP, este costo NO es fijo — es el monto real de esa
  transaccion fraudulenta especifica (`Amount`), porque eso es exactamente
  el dinero que el fraude se lleva sin ser detectado. Usar el monto real
  por fila, en vez de un promedio, es lo que hace que este umbral sea
  "sensible a costos" en el sentido correcto: fraudes grandes deben pesar
  mas en la decision que fraudes pequenos, no todos los FN son iguales.

El umbral optimo es el que minimiza `costo_total = costo_FP + costo_FN`
barriendo sobre los scores del modelo como candidatos a umbral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_FALSE_POSITIVE = 5.0  # USD: costo operativo de revisar una alerta que resulta ser legitima
N_THRESHOLD_CANDIDATES = 300


def cost_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray, threshold: float,
    cost_false_positive: float = DEFAULT_COST_FALSE_POSITIVE,
) -> dict:
    """Costo financiero total de operar el modelo con `threshold` como
    umbral de alerta (`y_score >= threshold` -> transaccion marcada)."""
    flagged = y_score >= threshold
    fp_mask = flagged & (y_true == 0)
    fn_mask = (~flagged) & (y_true == 1)
    tp_mask = flagged & (y_true == 1)

    cost_fp = float(fp_mask.sum()) * cost_false_positive
    cost_fn = float(amounts[fn_mask].sum())

    return {
        "threshold": float(threshold),
        "n_flagged": int(flagged.sum()),
        "tp": int(tp_mask.sum()),
        "fp": int(fp_mask.sum()),
        "fn": int(fn_mask.sum()),
        "fraud_amount_caught_usd": float(amounts[tp_mask].sum()),
        "cost_false_positives_usd": cost_fp,
        "cost_false_negatives_usd": cost_fn,
        "total_cost_usd": cost_fp + cost_fn,
    }


def optimize_cost_sensitive_threshold(
    y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray,
    cost_false_positive: float = DEFAULT_COST_FALSE_POSITIVE,
    n_candidates: int = N_THRESHOLD_CANDIDATES,
) -> tuple[dict, pd.DataFrame]:
    """Barre `n_candidates` umbrales (percentiles del score observado) y
    devuelve el punto de costo minimo, junto con el barrido completo para
    graficar costo-vs-umbral."""
    candidates = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, n_candidates)))
    sweep = [
        cost_at_threshold(y_true, y_score, amounts, t, cost_false_positive)
        for t in candidates
    ]
    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.loc[sweep_df["total_cost_usd"].idxmin()].to_dict()
    return best, sweep_df


def cost_at_naive_flag_all(y_true: np.ndarray, amounts: np.ndarray) -> float:
    """Costo de referencia: no usar ningun modelo y dejar pasar todo (costo
    = suma de todo el fraude no detectado). El umbral optimo debe, como
    minimo, mejorar sustancialmente sobre esta linea base trivial."""
    return float(amounts[y_true == 1].sum())
