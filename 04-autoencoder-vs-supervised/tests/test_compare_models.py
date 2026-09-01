import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare_models import _metrics_at_budget, evaluar


def test_metrics_at_budget_perfect_ranking():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # los dos positivos son los mas altos
    resultado = _metrics_at_budget(y_true, y_score, k=2)
    assert resultado["precision"] == 1.0
    assert resultado["recall"] == 1.0
    assert resultado["tp"] == 2


def test_metrics_at_budget_random_ranking():
    y_true = np.array([1, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.7, 0.6])  # el positivo tiene el score mas bajo
    resultado = _metrics_at_budget(y_true, y_score, k=1)
    assert resultado["precision"] == 0.0
    assert resultado["recall"] == 0.0


def test_evaluar_returns_expected_keys():
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.ones(10), np.zeros(990)]).astype(int)
    y_score = rng.random(1000)
    resultado = evaluar("modelo_test", y_true, y_score)
    assert resultado["nombre"] == "modelo_test"
    assert 0.0 <= resultado["roc_auc"] <= 1.0
    assert 0.0 <= resultado["pr_auc"] <= 1.0
    assert len(resultado["sweep_por_presupuesto"]) > 0


def test_evaluar_high_score_for_perfect_separation():
    y_true = np.array([0] * 100 + [1] * 5)
    y_score = np.array([0.0] * 100 + [1.0] * 5)
    resultado = evaluar("perfecto", y_true, y_score)
    assert resultado["roc_auc"] == 1.0
    assert resultado["pr_auc"] == 1.0
