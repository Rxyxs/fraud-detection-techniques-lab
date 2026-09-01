"""Feature engineering sobre componentes PCA + Amount (unica variable no
transformada). Sin columnas de entidad cruda, las features utiles aqui son:
interacciones entre los componentes PCA de mayor peso discriminante
(determinados empiricamente via correlacion absoluta con Class en el set
de entrenamiento, nunca en test), un proxy de "distancia al centroide"
(norma L2 del vector V1..V28, una medida agregada de cuan atipica es la
transaccion en el espacio latente) y una transformacion log de Amount
(la distribucion de montos es fuertemente asimetrica)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, PCA_COLUMNS


def _top_correlated_columns(df: pd.DataFrame, n: int = 4) -> list[str]:
    corr = df[PCA_COLUMNS].corrwith(df["Class"]).abs().sort_values(ascending=False)
    return corr.head(n).index.tolist()


def engineer_features(df: pd.DataFrame, top_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    if top_cols is None:
        top_cols = _top_correlated_columns(out)

    out["amount_log"] = np.log1p(out["Amount"])
    out["v_l2_norm"] = np.sqrt((out[PCA_COLUMNS] ** 2).sum(axis=1))

    for i in range(len(top_cols)):
        for j in range(i + 1, len(top_cols)):
            a, b = top_cols[i], top_cols[j]
            out[f"{a}_x_{b}"] = out[a] * out[b]

    feature_cols = (
        FEATURE_COLUMNS
        + ["amount_log", "v_l2_norm"]
        + [f"{top_cols[i]}_x_{top_cols[j]}" for i in range(len(top_cols)) for j in range(i + 1, len(top_cols))]
    )
    return out, feature_cols
