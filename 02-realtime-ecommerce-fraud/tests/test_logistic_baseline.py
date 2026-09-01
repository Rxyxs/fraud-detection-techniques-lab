import numpy as np
import pandas as pd

from src.models.logistic_baseline import (
    coefficient_report,
    predict_proba,
    train_logistic_baseline,
)


def _make_synthetic_split(n=2000, fraud_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount_clp": rng.normal(20000, 5000, n_legit).clip(min=500),
        "velocity_ratio": rng.normal(1.0, 0.3, n_legit).clip(min=0),
    })
    fraud = pd.DataFrame({
        "amount_clp": rng.normal(500000, 100000, n_fraud).clip(min=1000),
        "velocity_ratio": rng.uniform(0.0, 0.2, n_fraud),
    })
    X = pd.concat([legit, fraud], ignore_index=True)
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


def test_train_logistic_baseline_returns_fitted_model_and_scaler():
    X, y = _make_synthetic_split()
    model, scaler = train_logistic_baseline(X, y, seed=1)
    proba = predict_proba(model, scaler, X)

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_logistic_baseline_separates_easy_synthetic_fraud():
    X_train, y_train = _make_synthetic_split(n=3000, fraud_rate=0.05, seed=1)
    X_test, y_test = _make_synthetic_split(n=1000, fraud_rate=0.05, seed=2)

    model, scaler = train_logistic_baseline(X_train, y_train, seed=1)
    proba = predict_proba(model, scaler, X_test)

    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y_test, proba) > 0.9


def test_coefficient_report_is_sorted_by_magnitude_and_covers_all_features():
    X, y = _make_synthetic_split()
    model, _ = train_logistic_baseline(X, y, seed=1)
    report = coefficient_report(model, list(X.columns))

    assert set(report.keys()) == set(X.columns)
    magnitudes = [abs(v) for v in report.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)
