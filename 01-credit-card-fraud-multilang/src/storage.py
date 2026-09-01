"""Persistencia en SQLite (stdlib, sin dependencias adicionales) -- metricas
de modelo y predicciones de scoring."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "fraud.sqlite"


def export_results(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    metrics.to_sql("model_metrics", con, if_exists="replace", index=False)
    predictions.to_sql("predictions", con, if_exists="replace", index=False)
    con.close()
