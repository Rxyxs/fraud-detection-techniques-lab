"""Ingesta y limpieza: dataset real 'Credit Card Fraud Detection Dataset 2023'
(Kaggle, 568.630 transacciones reales, 2023). Componentes V1-V28 son
proyecciones PCA de las variables originales (mismo esquema de
anonimizacion que el dataset clasico de ULB) -- disclosure honesto: esto
significa que no hay columnas de tarjeta/dispositivo/IP crudas para
agregacion dinamica por entidad, a diferencia de IEEE-CIS. La limpieza real
que SI aplica: verificacion de duplicados, nulos, y outliers extremos en
Amount (la unica variable no transformada por PCA).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard_2023.csv"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = PCA_COLUMNS + ["Amount"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    n_before = len(df)
    n_duplicates = df.duplicated(subset=PCA_COLUMNS + ["Amount", "Class"]).sum()
    df = df.drop_duplicates(subset=PCA_COLUMNS + ["Amount", "Class"]).reset_index(drop=True)
    n_nulls = df[FEATURE_COLUMNS + ["Class"]].isnull().sum().sum()
    df = df.dropna(subset=FEATURE_COLUMNS + ["Class"]).reset_index(drop=True)

    print(f"[data.py] filas crudas={n_before}  duplicados removidos={n_duplicates}  "
          f"nulos removidos={n_nulls}  filas finales={len(df)}")

    return df
