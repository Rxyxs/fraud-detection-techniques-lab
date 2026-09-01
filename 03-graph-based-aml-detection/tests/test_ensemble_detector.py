import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.ensemble_detector import FEATURE_COLUMNS, evaluate_against_ground_truth, run_ensemble


def _synthetic_feature_table(n_normal=95, n_outliers=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_normal + n_outliers
    data = {col: rng.normal(loc=1.0, scale=0.2, size=n) for col in FEATURE_COLUMNS}
    # empuja los ultimos n_outliers a valores extremos en varias columnas a la vez
    for col in ["monto_max_enviado", "burst_score_24h", "centralidad_intermediacion", "n_cercanas_umbral"]:
        data[col][-n_outliers:] = data[col][-n_outliers:] * 50 + 100
    data["account_id"] = [f"CTA{i:03d}" for i in range(n)]
    cols = ["account_id"] + FEATURE_COLUMNS
    return pl.DataFrame({c: data[c] for c in cols}), [f"CTA{i:03d}" for i in range(n_normal, n)]


def test_run_ensemble_adds_expected_columns():
    features, _ = _synthetic_feature_table()
    scored = run_ensemble(features, contamination=0.05)
    for col in ["score_iforest", "score_copod", "score_ecod", "score_ensamble", "alerta"]:
        assert col in scored.columns
    assert scored.height == features.height
    assert scored["score_ensamble"].null_count() == 0


def test_ensemble_flags_injected_outliers():
    features, outlier_ids = _synthetic_feature_table(n_normal=95, n_outliers=5)
    scored = run_ensemble(features, contamination=0.05)
    alertadas = set(scored.filter(pl.col("alerta"))["account_id"].to_list())
    # al menos parte de los outliers inyectados deben quedar entre las alertas
    assert len(alertadas & set(outlier_ids)) >= 3


def test_contamination_controls_alert_count():
    features, _ = _synthetic_feature_table(n_normal=180, n_outliers=20)
    scored = run_ensemble(features, contamination=0.10)
    assert scored.filter(pl.col("alerta")).height == round(0.10 * features.height)


def test_evaluate_against_ground_truth():
    features, outlier_ids = _synthetic_feature_table(n_normal=95, n_outliers=5)
    scored = run_ensemble(features, contamination=0.05)
    ground_truth = pl.DataFrame({"account_id": outlier_ids})
    metrics = evaluate_against_ground_truth(scored, ground_truth)
    assert metrics["n_cuentas_ilicitas_verdad_terreno"] == len(outlier_ids)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["roc_auc"] > 0.5  # el ensamble debe superar el azar en este caso disenado
