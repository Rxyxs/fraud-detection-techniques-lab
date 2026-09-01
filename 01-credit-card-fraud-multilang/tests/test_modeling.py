import numpy as np

from src.modeling import business_cost, calibrate_threshold_by_cost


def test_business_cost_zero_for_perfect_prediction():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert business_cost(y_true, y_pred) == 0


def test_business_cost_weighs_false_negative_more_than_false_positive():
    y_true = np.array([1, 0])
    fn_cost = business_cost(y_true, np.array([0, 0]))  # 1 falso negativo
    fp_cost = business_cost(np.array([0, 1]), np.array([1, 1]))  # 1 falso positivo
    assert fn_cost > fp_cost


def test_calibrated_threshold_never_worse_than_default_half():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    threshold, cost_at_best = calibrate_threshold_by_cost(y_true, y_proba)
    cost_at_half = business_cost(y_true, (y_proba >= 0.5).astype(int))
    assert cost_at_best <= cost_at_half
