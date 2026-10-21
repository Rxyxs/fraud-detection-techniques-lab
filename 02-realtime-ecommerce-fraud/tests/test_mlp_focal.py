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


def test_early_stopping_respects_minimum_epochs_and_marks_best_epoch():
    X_train, y_train, X_val, y_val = _split(_make_synthetic_data(n=600, seed=3))
    _model, history = train_one_activation("relu", X_train, y_train, X_val, y_val,
                                           n_epochs=60, min_epochs=20, patience=2, seed=3)
    assert len(history) >= 20
    best = [h for h in history if h["is_best"]]
    assert len(best) == 1 and best[0]["val_loss"] == min(h["val_loss"] for h in history)


def test_interrupted_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    import pytest

    X_train, y_train, X_val, y_val = _split(_make_synthetic_data(n=600, seed=4))
    kwargs = dict(n_epochs=6, min_epochs=6, patience=50, batch_size=128, seed=4)
    reference, ref_history = train_one_activation("gelu", X_train, y_train, X_val, y_val, **kwargs)

    ckpt = tmp_path / "gelu.ckpt"
    real_step = torch.optim.AdamW.step
    calls = {"n": 0}
    n_batches = -(-len(X_train) // 128)

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > n_batches * 3:  # cut during epoch 4
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_one_activation("gelu", X_train, y_train, X_val, y_val, checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", real_step)

    resumed, history = train_one_activation("gelu", X_train, y_train, X_val, y_val, checkpoint_path=ckpt, **kwargs)
    assert [h["val_loss"] for h in history] == pytest.approx([h["val_loss"] for h in ref_history], rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)


def _split(data, frac=0.8):
    X, y = data
    cut = int(len(X) * frac)
    return X[:cut], y[:cut], X[cut:], y[cut:]
