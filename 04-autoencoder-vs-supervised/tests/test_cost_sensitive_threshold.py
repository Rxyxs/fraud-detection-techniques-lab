import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.cost_sensitive_threshold import (
    cost_at_naive_flag_all,
    cost_at_threshold,
    optimize_cost_sensitive_threshold,
)


def _toy_data():
    # 4 normales (amount irrelevante) + 2 fraudes ($100 y $900)
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    amounts = np.array([10.0, 20.0, 30.0, 40.0, 100.0, 900.0])
    return y_true, y_score, amounts


def test_cost_at_threshold_perfect_separation_has_zero_cost():
    y_true, y_score, amounts = _toy_data()
    result = cost_at_threshold(y_true, y_score, amounts, threshold=0.5, cost_false_positive=5.0)
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tp"] == 2
    assert result["total_cost_usd"] == 0.0


def test_cost_at_threshold_flag_none_costs_full_fraud_amount():
    y_true, y_score, amounts = _toy_data()
    result = cost_at_threshold(y_true, y_score, amounts, threshold=2.0, cost_false_positive=5.0)
    assert result["n_flagged"] == 0
    assert result["fn"] == 2
    assert result["total_cost_usd"] == amounts[y_true == 1].sum()


def test_optimize_cost_sensitive_threshold_finds_the_zero_cost_point():
    y_true, y_score, amounts = _toy_data()
    best, sweep = optimize_cost_sensitive_threshold(y_true, y_score, amounts, cost_false_positive=5.0, n_candidates=50)
    assert best["total_cost_usd"] == sweep["total_cost_usd"].min()
    assert best["total_cost_usd"] <= 5.0  # deberia encontrar (cerca de) la separacion perfecta


def test_cost_at_naive_flag_all_equals_total_fraud_amount():
    y_true, _, amounts = _toy_data()
    assert cost_at_naive_flag_all(y_true, amounts) == 1000.0


def test_larger_false_negative_amount_shifts_optimal_threshold_lower():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.5, 0.6, 0.55, 0.9])
    amounts_small_fraud = np.array([10.0, 10.0, 10.0, 20.0, 20.0])
    amounts_large_fraud = np.array([10.0, 10.0, 10.0, 5000.0, 5000.0])

    best_small, _ = optimize_cost_sensitive_threshold(y_true, y_score, amounts_small_fraud, cost_false_positive=50.0)
    best_large, _ = optimize_cost_sensitive_threshold(y_true, y_score, amounts_large_fraud, cost_false_positive=50.0)

    # con fraude mucho mas caro, el umbral optimo debe volverse mas permisivo (mas bajo)
    assert best_large["threshold"] <= best_small["threshold"]
