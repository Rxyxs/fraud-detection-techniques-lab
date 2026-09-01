"""Persistencia local (DuckDB) de las metricas comparativas entre los
enfoques de modelado (baseline estadistico, ensamble PyOD, autoencoder
PyTorch por activacion). Permite consultar/trackear resultados entre
corridas del pipeline sin depender de un servicio externo -- coherente con
el patron del resto del repo (SQLite para decisiones de analista en
``src/api/store.py``, aqui DuckDB porque el consumo es analitico/tabular)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS model_comparison (
        run_timestamp TIMESTAMP,
        modelo VARCHAR,
        roc_auc DOUBLE,
        average_precision DOUBLE,
        precision_en_alertas DOUBLE,
        recall_sobre_verdad_terreno DOUBLE,
        n_alertas INTEGER
    )
"""


def persist_comparison(metrics_by_model: dict[str, dict], db_path: Path) -> None:
    """Inserta una fila por modelo para la corrida actual (``run_timestamp``
    comun), preservando corridas anteriores para poder comparar evolucion."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_SCHEMA)
        ts = datetime.now(timezone.utc)
        for modelo, m in metrics_by_model.items():
            con.execute(
                "INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ts,
                    modelo,
                    m.get("roc_auc"),
                    m.get("average_precision"),
                    m.get("precision_en_alertas"),
                    m.get("recall_sobre_verdad_terreno"),
                    m.get("n_alertas"),
                ],
            )
    finally:
        con.close()


def latest_comparison(db_path: Path) -> list:
    """Retorna las filas de la corrida mas reciente, ordenadas por ROC-AUC
    descendente. Lista vacia si la tabla aun no existe."""
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT modelo, roc_auc, average_precision, precision_en_alertas,
                   recall_sobre_verdad_terreno, n_alertas
            FROM model_comparison
            WHERE run_timestamp = (SELECT max(run_timestamp) FROM model_comparison)
            ORDER BY roc_auc DESC
            """
        ).fetchall()
    except duckdb.CatalogException:
        rows = []
    finally:
        con.close()
    return rows
