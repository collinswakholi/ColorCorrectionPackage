"""Inference helper overrides and layer metadata cannot bless stale tables."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from ColorCorrectionPipeline.core.correction import Regressor_Model, _predict_direct
from ColorCorrectionPipeline.core.execution import evaluate_direct
from ColorCorrectionPipeline.core.lut_cache import inference_identity, ensure_lut
from ColorCorrectionPipeline.core.accel import build_3d_lut
from ColorCorrectionPipeline.fitting import fit_observations


def linear():
    x = np.random.default_rng(125).uniform(.05, .95, (24, 3))
    model = fit_observations(x, x*.8+.02, [str(i) for i in range(24)],
                             {"degree": 1, "lut_grid_size": 3}, "reproducible-v1")
    return model


def test_linear_instance_helper_override_disables_cache_and_fusion():
    model = linear()
    assert ensure_lut(model, build_3d_lut)
    model.model._decision_function = lambda x: np.full((len(x), 3), .73)
    assert inference_identity(model)[0] is None
    assert ensure_lut(model, build_3d_lut)
    np.testing.assert_array_equal(model._lut, np.full_like(model._lut, .73))
    actual, diagnostic = evaluate_direct(np.full((3, 3), .3), model, backend="numpy")
    np.testing.assert_array_equal(actual, np.full((3, 3), .73))
    assert diagnostic["actual_backend"] == "reference-direct"
    with pytest.raises(ValueError):
        evaluate_direct(np.full((3, 3), .3), model, backend="numpy", allow_fallback=False)


def test_linear_class_helper_mutation_invalidates_identity(monkeypatch):
    model = linear()
    assert ensure_lut(model, build_3d_lut)
    before = model._lut_identity
    monkeypatch.setattr(LinearRegression, "_decision_function", lambda self, x: np.full((len(x), 3), .63))
    assert inference_identity(model)[0] != before
    assert ensure_lut(model, build_3d_lut)
    np.testing.assert_array_equal(model._lut, np.full_like(model._lut, .63))


def test_mlp_helper_override_and_layer_count_invalidate():
    rng = np.random.default_rng(124)
    estimator = MLPRegressor(hidden_layer_sizes=(3,), random_state=3, max_iter=1)
    estimator.fit(rng.random((12, 4)), rng.random((12, 3)))
    model = Regressor_Model()
    model.model, model.degree = estimator, 1
    model.portable_inference = True
    model._lut_grid_size = 3
    model._lut_predict_fn = model._predict_for_lut
    before = inference_identity(model)[0]
    estimator.n_layers_ = 2
    assert inference_identity(model)[0] != before
    estimator._forward_pass_fast = lambda x, **kwargs: np.full((len(x), 3), .27)
    assert inference_identity(model)[0] is None
    assert ensure_lut(model, build_3d_lut)
    np.testing.assert_array_equal(model._lut, np.full_like(model._lut, .27))
