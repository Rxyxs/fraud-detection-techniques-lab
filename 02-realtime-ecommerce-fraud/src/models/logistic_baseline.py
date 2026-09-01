"""First modeling approach (interpretable baseline), complementary to the
CatBoost/XGBoost ensemble in ``catboost_fraud.py`` and the PyTorch MLP in
``mlp_focal.py``.

Logistic regression on the same feature matrix (``FULL_FEATURE_COLUMNS``,
including the autoencoder anomaly score), same cost-sensitive sample
weights, same business-cost threshold tuning -- so the three approaches are
directly comparable on identical features, splits and metrics, with no
methodological shortcut favoring one of them.

A linear model is not competitive on raw accuracy against gradient-boosted
trees here, but it is auditable: every coefficient has a sign and a
magnitude a fraud analyst can reason about directly, which is valuable as a
sanity check and a regulatory-friendly fallback even when a stronger
challenger is deployed.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_logistic_baseline(
    X_train,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fits a StandardScaler + L2-regularized logistic regression.

    Logistic regression is scale-sensitive (unlike CatBoost/XGBoost), so
    features are standardized first; the scaler is returned alongside the
    model so callers can apply the identical transform at inference time.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
    return model, scaler


def predict_proba(model: LogisticRegression, scaler: StandardScaler, X) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def coefficient_report(model: LogisticRegression, feature_columns: list[str]) -> dict:
    """Signed coefficients, sorted by magnitude -- the interpretability
    payoff of this approach over the tree ensemble / MLP challengers."""
    coefs = model.coef_.ravel()
    order = np.argsort(-np.abs(coefs))
    return {
        feature_columns[i]: float(coefs[i])
        for i in order
    }
