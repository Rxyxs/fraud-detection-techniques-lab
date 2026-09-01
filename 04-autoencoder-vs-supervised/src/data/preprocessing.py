"""Preprocesamiento y particion del dataset real de fraude (ULB/OpenML).

Estrategia de split, pensada para comparar de forma justa un enfoque no
supervisado (autoencoder, entrenado SOLO con transacciones normales, como
en un escenario real sin fraude confirmado todavia) contra uno supervisado
(XGBoost, que si puede usar las etiquetas de fraude en el set de
entrenamiento):

    1. Split estratificado 70/15/15 (train/val/test) preservando la
       proporcion real de fraude (~0.172%) en cada particion.
    2. El autoencoder entrena y calibra su umbral SOLO con las
       transacciones normales de train/val (nunca ve un fraude durante
       el entrenamiento ni la seleccion de umbral).
    3. El modelo supervisado entrena con train completo (normal + fraude).
    4. Ambos se evaluan sobre el MISMO test set (normal + fraude), nunca
       usado por ninguno de los dos durante ajuste o calibracion.

`Time` y `Amount` se escalan (las columnas V1-V28 ya vienen como
componentes PCA aproximadamente estandarizados por los autores originales
del dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"
FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET_COLUMN = "Class"

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: RobustScaler


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecuta primero `python data/download_dataset.py`."
        )
    df = pd.read_csv(path)
    # OpenML exporta el atributo nominal `Class` del ARFF original como los
    # strings literales "'0'"/"'1'" (con comillas incluidas en el CSV).
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip("'").astype(int)
    return df


def make_splits(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> DatasetSplits:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=random_state,
    )
    relative_test_frac = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_frac, stratify=y_temp, random_state=random_state,
    )

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_val_scaled[["Time", "Amount"]] = scaler.transform(X_val[["Time", "Amount"]])
    X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return DatasetSplits(
        X_train=X_train_scaled, y_train=y_train,
        X_val=X_val_scaled, y_val=y_val,
        X_test=X_test_scaled, y_test=y_test,
        scaler=scaler,
    )


def normal_only(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Subconjunto de transacciones legitimas, usado para entrenar y calibrar
    el autoencoder sin ver ningun fraude confirmado."""
    return X.loc[y == 0]


def dataset_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    n_fraud = int(df[TARGET_COLUMN].sum())
    return {
        "n_transacciones": n,
        "n_fraude": n_fraud,
        "n_normal": n - n_fraud,
        "proporcion_fraude": n_fraud / n,
        "monto_total_fraude_usd": float(df.loc[df[TARGET_COLUMN] == 1, "Amount"].sum()),
        "monto_promedio_fraude_usd": float(df.loc[df[TARGET_COLUMN] == 1, "Amount"].mean()),
        "monto_promedio_normal_usd": float(df.loc[df[TARGET_COLUMN] == 0, "Amount"].mean()),
    }
