import numpy as np
import pytest

from src import metrics_store


@pytest.fixture()
def con(tmp_path):
    db_path = tmp_path / "test_metrics.duckdb"
    connection = metrics_store.connect(db_path)
    yield connection
    connection.close()


def _fake_metrics(roc_auc=0.9, recall=0.8):
    return {
        "threshold": 0.5,
        "precision": 0.95,
        "recall": recall,
        "f1": 0.87,
        "roc_auc": roc_auc,
        "pr_auc": 0.9,
        "cost_savings_vs_no_model_clp": 1_000_000.0,
    }


def test_persist_metrics_inserts_one_row(con):
    metrics_store.persist_metrics(con, "baseline_interpretable", "logistic_regression", _fake_metrics())
    rows = con.execute("SELECT approach, model_name, roc_auc FROM model_runs").fetchall()
    assert rows == [("baseline_interpretable", "logistic_regression", 0.9)]


def test_persist_predictions_round_trips_arrays(con):
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.3, 0.95])
    metrics_store.persist_predictions(con, "deep_learning", y_true, y_proba)

    rows = con.execute(
        "SELECT y_true, y_proba FROM predictions ORDER BY row_index"
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 0, 1]
    assert [r[1] for r in rows] == pytest.approx([0.1, 0.8, 0.3, 0.95])


def test_latest_comparison_orders_by_roc_auc_desc_and_keeps_only_latest_run(con):
    run1 = metrics_store.now()
    metrics_store.persist_metrics(con, "tree_ensemble", "catboost", _fake_metrics(roc_auc=0.5), run_ts=run1)

    run2 = metrics_store.now()
    metrics_store.persist_metrics(con, "baseline_interpretable", "logistic_regression", _fake_metrics(roc_auc=0.7), run_ts=run2)
    metrics_store.persist_metrics(con, "deep_learning", "mlp_relu_focal_loss", _fake_metrics(roc_auc=0.95), run_ts=run2)

    latest = metrics_store.latest_comparison(con)

    assert len(latest) == 2
    assert list(latest["model_name"]) == ["mlp_relu_focal_loss", "logistic_regression"]
