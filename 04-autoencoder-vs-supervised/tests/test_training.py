import pytest
import torch

from src.models.autoencoder import FraudAutoencoder
from src.models.training import fit


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(256, 10, generator=g), torch.randn(64, 10, generator=g)


def _fit(model, train, val, **kwargs):
    loss_fn = torch.nn.MSELoss()
    return fit(model, train, val,
               batch_loss=lambda m, b: loss_fn(m(b), b),
               val_metrics=lambda m, X: {"val_loss": loss_fn(m(X), X).item()},
               lr=1e-3, batch_size=64, seed=0, **kwargs)


def test_never_stops_before_minimum_and_restores_best_epoch():
    train, val = _data()
    torch.manual_seed(0)
    model, history, best = _fit(FraudAutoencoder(input_dim=10, bottleneck_dim=4), train, val,
                                min_epochs=30, max_epochs=80, patience=2)
    assert len(history) >= 30
    assert history.loc[history["is_best"], "val_loss"].item() == pytest.approx(history["val_loss"].min())
    with torch.no_grad():
        assert torch.nn.functional.mse_loss(model(val), val).item() == pytest.approx(best, rel=1e-5)


def test_interrupted_run_resumes_from_checkpoint(tmp_path, monkeypatch):
    train, val = _data(1)
    kwargs = dict(min_epochs=6, max_epochs=6, patience=100)
    torch.manual_seed(0)
    reference, ref_history, _ = _fit(FraudAutoencoder(input_dim=10, bottleneck_dim=4), train, val, **kwargs)

    ckpt = tmp_path / "ae.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > 4 * 3:  # 256 / 64 = 4 batches por epoca: corta en la epoca 4
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    torch.manual_seed(0)
    with pytest.raises(KeyboardInterrupt):
        _fit(FraudAutoencoder(input_dim=10, bottleneck_dim=4), train, val, checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    torch.manual_seed(0)
    resumed, history, _ = _fit(FraudAutoencoder(input_dim=10, bottleneck_dim=4), train, val,
                               checkpoint_path=ckpt, **kwargs)
    assert list(history["val_loss"]) == pytest.approx(list(ref_history["val_loss"]), rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)
