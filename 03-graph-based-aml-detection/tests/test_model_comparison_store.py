import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.model_comparison_store import latest_comparison, persist_comparison


def test_persist_and_read_back_latest_comparison(tmp_path):
    db_path = tmp_path / "model_comparison.duckdb"
    metricas = {
        "baseline_estadistico": {
            "roc_auc": 0.71, "average_precision": 0.30,
            "precision_en_alertas": 0.4, "recall_sobre_verdad_terreno": 0.2, "n_alertas": 10,
        },
        "ensamble_pyod": {
            "roc_auc": 0.90, "average_precision": 0.40,
            "precision_en_alertas": 0.6, "recall_sobre_verdad_terreno": 0.3, "n_alertas": 10,
        },
    }

    persist_comparison(metricas, db_path)
    assert db_path.exists()

    filas = latest_comparison(db_path)
    assert len(filas) == 2
    # ordenado por roc_auc descendente -> ensamble_pyod primero
    assert filas[0][0] == "ensamble_pyod"
    assert filas[1][0] == "baseline_estadistico"


def test_latest_comparison_missing_db_returns_empty(tmp_path):
    db_path = tmp_path / "does_not_exist.duckdb"
    assert latest_comparison(db_path) == []


def test_persist_comparison_accumulates_multiple_runs(tmp_path):
    db_path = tmp_path / "model_comparison.duckdb"
    m1 = {"modelo_a": {"roc_auc": 0.5, "average_precision": 0.1, "precision_en_alertas": 0.1,
                        "recall_sobre_verdad_terreno": 0.1, "n_alertas": 5}}
    m2 = {"modelo_a": {"roc_auc": 0.8, "average_precision": 0.2, "precision_en_alertas": 0.2,
                        "recall_sobre_verdad_terreno": 0.2, "n_alertas": 5}}

    persist_comparison(m1, db_path)
    persist_comparison(m2, db_path)

    # latest_comparison solo debe traer la corrida mas reciente
    filas = latest_comparison(db_path)
    assert len(filas) == 1
    assert filas[0][1] == 0.8
