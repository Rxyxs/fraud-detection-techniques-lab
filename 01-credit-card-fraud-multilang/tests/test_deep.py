import numpy as np

from src.deep import FocalLoss, compare_activations, predict_proba, train_mlp


def _toy_data(n=200, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    weights = rng.normal(size=n_features)
    logits = X @ weights
    y = (logits > np.median(logits)).astype(int)
    return X, y


def test_focal_loss_is_finite_and_nonnegative():
    import torch

    loss_fn = FocalLoss()
    logits = torch.tensor([2.0, -2.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_train_mlp_reduces_loss_over_epochs():
    X, y = _toy_data()
    model, history = train_mlp(X[:150], y[:150], X[150:], y[150:], activation_name="relu", n_epochs=10)
    assert history[-1]["train_loss"] <= history[0]["train_loss"]
    assert len(history) == 10


def test_predict_proba_returns_values_in_unit_interval():
    X, y = _toy_data()
    model, _ = train_mlp(X[:150], y[:150], X[150:], y[150:], activation_name="gelu", n_epochs=3)
    proba = predict_proba(model, X[150:])
    assert proba.shape[0] == 50
    assert (proba >= 0).all() and (proba <= 1).all()


def test_compare_activations_trains_all_three_and_history_has_expected_columns():
    X, y = _toy_data()
    models, history = compare_activations(X[:150], y[:150], X[150:], y[150:], n_epochs=3)
    assert set(models.keys()) == {"relu", "gelu", "swish"}
    assert set(history["activation"].unique()) == {"relu", "gelu", "swish"}
    assert {"epoch", "train_loss", "val_loss"}.issubset(history.columns)
