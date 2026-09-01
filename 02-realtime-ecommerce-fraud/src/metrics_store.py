"""Persistence of comparative metrics/predictions across the three modeling
approaches (logistic baseline, CatBoost/XGBoost ensemble, PyTorch MLP with
Focal Loss) in a local DuckDB file, so pipeline runs can be compared without
re-parsing JSON reports by hand.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "metrics.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    run_ts TIMESTAMP DEFAULT current_timestamp,
    approach VARCHAR,      -- 'baseline_interpretable' | 'tree_ensemble' | 'deep_learning'
    model_name VARCHAR,
    precision DOUBLE,
    recall DOUBLE,
    f1 DOUBLE,
    roc_auc DOUBLE,
    pr_auc DOUBLE,
    threshold DOUBLE,
    cost_savings_vs_no_model_clp DOUBLE,
    extra JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    run_ts TIMESTAMP DEFAULT current_timestamp,
    approach VARCHAR,
    row_index BIGINT,
    y_true INTEGER,
    y_proba DOUBLE
);
"""


def connect(db_path: pathlib.Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_now = now  # internal alias kept for backward-compatible callers


def persist_metrics(
    con: duckdb.DuckDBPyConnection,
    approach: str,
    model_name: str,
    metrics: dict,
    extra: dict | None = None,
    run_ts: datetime | None = None,
) -> None:
    run_ts = run_ts or _now()
    con.execute(
        "INSERT INTO model_runs "
        "(run_ts, approach, model_name, precision, recall, f1, roc_auc, pr_auc, "
        " threshold, cost_savings_vs_no_model_clp, extra) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_ts, approach, model_name,
            metrics["precision"], metrics["recall"], metrics["f1"],
            metrics["roc_auc"], metrics["pr_auc"], metrics["threshold"],
            metrics["cost_savings_vs_no_model_clp"],
            json.dumps(extra or {}),
        ],
    )


def persist_predictions(
    con: duckdb.DuckDBPyConnection,
    approach: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    run_ts: datetime | None = None,
) -> None:
    run_ts = run_ts or _now()
    df = pd.DataFrame({
        "row_index": np.arange(len(y_true)),
        "y_true": np.asarray(y_true).astype(int),
        "y_proba": np.asarray(y_proba).astype(float),
    })
    con.register("preds_tmp", df)
    con.execute(
        "INSERT INTO predictions (run_ts, approach, row_index, y_true, y_proba) "
        "SELECT ?, ?, row_index, y_true, y_proba FROM preds_tmp",
        [run_ts, approach],
    )
    con.unregister("preds_tmp")


def latest_comparison(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT approach, model_name, precision, recall, f1, roc_auc, pr_auc, "
        "       cost_savings_vs_no_model_clp "
        "FROM model_runs WHERE run_ts = (SELECT max(run_ts) FROM model_runs) "
        "ORDER BY roc_auc DESC"
    ).fetchdf()


if __name__ == "__main__":
    con = connect()
    print(latest_comparison(con).to_string(index=False))
    con.close()
