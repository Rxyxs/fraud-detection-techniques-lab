import numpy as np
import torch

from src.models.mlp_focal import (
    ACTIVATIONS,
    FocalLoss,
    FraudMLP,
    compare_activations,
    predict_proba,
    train_one_activation,
)


def _make_synthetic_data(n=1500, n_features=5, fraud_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    n_legit = n - n_fraud
    legit = rng.normal(0.0, 1.0, size=(n_legit, n_features))
    fraud = rng.normal(4.0, 1.0, size=(n_fraud, n_features))
    X = np.vstack([legit, fraud]).astype(np.float32)
    y = np.array([0.0] * n_legit + [1.0] * n_fraud, dtype=np.float32)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_focal_loss_is_nonnegative_and_finite():
    criterion = FocalLoss()
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = criterion(logits, targets)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_forward_pass_shape_for_all_activations():
    x = torch.randn(8, 6)
    for activation in ACTIVATIONS:
        model = FraudMLP(n_features=6, activation=activation)
        model.eval()
        out = model(x)
        assert out.shape == (8,)


def test_train_one_activation_reduces_training_loss():
    X, y = _make_synthetic_data(n=800, seed=1)
    X_train, X_val = X[:600], X[600:]
    y_train, y_val = y[:600], y[600:]

    _, history = train_one_activation(
        "relu", X_train, y_train, X_val, y_val, n_epochs=15, seed=1
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_compare_activations_returns_all_three_and_separates_easy_data():
    X, y = _make_synthetic_data(n=1000, seed=2)
    X_train, X_val = X[:700], X[700:]
    y_train, y_val = y[:700], y[700:]

    models, histories = compare_activations(X_train, y_train, X_val, y_val, n_epochs=20, seed=2)

    assert set(models.keys()) == set(ACTIVATIONS.keys())
    assert set(histories.keys()) == set(ACTIVATIONS.keys())

    from sklearn.metrics import roc_auc_score
    for activation, model in models.items():
        proba = predict_proba(model, X_val)
        assert proba.shape == (len(X_val),)
        assert roc_auc_score(y_val, proba) > 0.8
