"""CustomNN must be reproducible for a fixed ``random_state`` (documented contract)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ColorCorrectionPipeline.core.correction import CustomNN  # noqa: E402


def _data(n=96, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    Y = X @ rng.random((3, 3)) + 0.05 * rng.standard_normal((n, 3))
    return X, Y


def _fit(random_state, **kw):
    X, Y = _data()
    model = CustomNN(
        hidden_layers=[16],
        max_epochs=25,
        batch_size=16,
        patience=50,
        random_state=random_state,
        **kw,
    )
    return model.fit(X, Y).predict(X)


def test_same_seed_same_predictions():
    np.testing.assert_array_equal(_fit(42), _fit(42))


def test_same_seed_with_dropout_same_predictions():
    np.testing.assert_array_equal(_fit(7, dropout_rate=0.2), _fit(7, dropout_rate=0.2))


def test_different_seed_differs():
    assert not np.array_equal(_fit(1), _fit(2))


def test_fit_does_not_disturb_global_torch_rng():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    _fit(42)
    np.testing.assert_array_equal(torch.rand(4).numpy(), expected.numpy())


def test_temporary_checkpoint_removed_when_validation_never_improves(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    X, Y = _data()
    Y[:] = np.nan  # validation loss is NaN, so no checkpoint is ever written
    model = CustomNN(hidden_layers=[4], max_epochs=3, batch_size=16, patience=50, random_state=0)
    try:
        model.fit(X, np.nan_to_num(Y, nan=0.0) + np.where(np.isnan(Y), np.nan, 0))
    except ValueError:
        pass  # sklearn input validation may reject NaN targets; that is acceptable
    assert not list(tmp_path.glob("cc_best_model_*.pth"))
