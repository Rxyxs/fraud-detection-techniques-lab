import numpy as np
import pytest
import torch

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


def test_early_stopping_respects_minimum_epochs_and_marks_best_epoch():
    X, y = _toy_data()
    _model, history = train_mlp(X[:150], y[:150], X[150:], y[150:], activation_name="relu",
                                n_epochs=60, min_epochs=20, patience=2)
    assert len(history) >= 20
    best = [row for row in history if row["is_best"]]
    assert len(best) == 1
    assert best[0]["val_loss"] == min(row["val_loss"] for row in history)


def test_interrupted_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    X, y = _toy_data()
    kwargs = dict(activation_name="swish", n_epochs=8, min_epochs=8, patience=50, batch_size=32)
    reference, ref_history = train_mlp(X[:150], y[:150], X[150:], y[150:], **kwargs)

    ckpt = tmp_path / "swish.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > 5 * 3:  # 150 filas / 32 = 5 batches por epoca: corta en la epoca 4
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_mlp(X[:150], y[:150], X[150:], y[150:], checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    resumed, history = train_mlp(X[:150], y[:150], X[150:], y[150:], checkpoint_path=ckpt, **kwargs)
    assert [r["val_loss"] for r in history] == pytest.approx([r["val_loss"] for r in ref_history], rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)
