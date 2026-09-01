import numpy as np
import pandas as pd
import pytest

from src.models.catboost_fraud import (
    business_cost,
    compute_sample_weights,
    evaluate,
    find_optimal_threshold,
    resample_train_split,
    train_models,
)


def _make_synthetic_split(n=2000, fraud_rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount_clp": rng.normal(20000, 5000, n_legit).clip(min=500),
        "velocity_ratio": rng.normal(1.0, 0.3, n_legit).clip(min=0),
        "distance_from_prev_km": rng.normal(5, 3, n_legit).clip(min=0),
    })
    fraud = pd.DataFrame({
        "amount_clp": rng.normal(500000, 100000, n_fraud).clip(min=1000),
        "velocity_ratio": rng.uniform(0.0, 0.2, n_fraud),
        "distance_from_prev_km": rng.normal(400, 100, n_fraud).clip(min=0),
    })
    X = pd.concat([legit, fraud], ignore_index=True)
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


def test_compute_sample_weights_scales_with_amount_and_imbalance():
    y = np.array([0, 0, 0, 0, 1, 1])
    amount = np.array([1000, 1000, 1000, 1000, 10000, 1000])
    weights = compute_sample_weights(y, amount)

    assert np.all(weights[y == 0] == 1.0)
    # Higher-amount fraud row should get a bigger weight than lower-amount fraud row.
    assert weights[4] > weights[5]
    # weight = class_ratio (n_neg/n_pos = 4/2 = 2) * amount / avg_fraud_amount (5500).
    assert weights[4] == pytest.approx(2 * 10_000 / 5_500)
    assert weights[5] == pytest.approx(2 * 1_000 / 5_500)


def test_resample_train_split_balances_classes():
    X, y = _make_synthetic_split(n=1000, fraud_rate=0.02)
    X_res, y_res = resample_train_split(X, y, seed=0)
    fraud_ratio_before = y.mean()
    fraud_ratio_after = y_res.mean()
    assert fraud_ratio_after > fraud_ratio_before
    assert fraud_ratio_after > 0.3  # SMOTE should bring it much closer to balanced


def test_business_cost_counts_fn_and_fp_correctly():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])  # 1 FN (row 0), 1 FP (row 2)
    amount = np.array([100_000, 50_000, 20_000, 30_000])
    total, fn_cost, fp_cost = business_cost(y_true, y_pred, amount, review_cost=3_000)
    assert fn_cost == 100_000
    assert fp_cost == 3_000
    assert total == 103_000


def test_train_and_evaluate_end_to_end_on_synthetic_data():
    X_train, y_train = _make_synthetic_split(n=3000, fraud_rate=0.02, seed=1)
    X_test, y_test = _make_synthetic_split(n=1000, fraud_rate=0.02, seed=2)

    X_res, y_res = resample_train_split(X_train, y_train, seed=1)
    weights = compute_sample_weights(y_res, X_res["amount_clp"].to_numpy())
    models = train_models(X_res, y_res, sample_weight_train=weights, seed=1)

    proba = models.catboost.predict_proba(X_test)[:, 1]
    amount_test = X_test["amount_clp"].to_numpy()

    threshold, _ = find_optimal_threshold(y_test, proba, amount_test)
    metrics = evaluate(y_test, proba, amount_test, threshold)

    # This synthetic split is easily separable (fraud is a different amount
    # and velocity regime entirely) -> the model should recover most of it.
    assert metrics["recall"] > 0.7
    assert metrics["pr_auc"] > 0.7
    assert metrics["cost_savings_vs_no_model_clp"] > 0
