"""Adversarial contracts for the opt-in independent-cell fitting policy."""
import copy
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from ColorCorrectionPipeline.fitting import (FittingPolicy, collapse_cell_observations,
    fit_observations, fit_conventional_observations)
from ColorCorrectionPipeline.core.correction import fit_model, predict_


@pytest.fixture
def paired():
    x = np.random.default_rng(31415).uniform(.02, .98, (24, 3))
    y = x @ np.array([[.8, .03, .01], [.02, .9, .04], [.01, .02, .85]]) + .01
    return x, y, ["r%dc%d" % (r+1, c+1) for r in range(4) for c in range(6)]


def fit(x, y, groups, **parameters):
    return fit_observations(x, y, groups, {"degree": 1, "use_lut": False, **parameters}, "reproducible-v1")


def test_linear_reproduces_affine_map_and_survives_pickle(paired):
    x, y, g = paired
    model = fit(x, y, g)
    np.testing.assert_allclose(predict_(x, model), y, atol=1e-14, rtol=0)
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(predict_(x, model), predict_(x, loaded))
    assert model.fit_report["unique_measured_cells"] == 24
    assert model.fit_report["design"]["rank"] == 4
    assert model.fit_report["estimator_precision"] == "CPU float64"
    assert model.fit_report["estimator_parameters"]["fit_intercept"] is False
    assert "scikit-learn" in model.fit_report["dependencies"]
    json.dumps(model.fit_report, allow_nan=False)


def test_repeating_one_cell_cannot_change_its_weight(paired):
    x, y, g = paired
    y = y + np.random.default_rng(7).normal(0, .03, y.shape)
    original = fit(x, y, g)
    repeated = fit(np.vstack((x, np.repeat(x[:1], 100, axis=0))),
                   np.vstack((y, np.repeat(y[:1], 100, axis=0))), g + [g[0]] * 100)
    np.testing.assert_allclose(predict_(x, original), predict_(x, repeated), atol=1e-14, rtol=0)
    assert repeated.fit_report["input_rows"] == 124
    assert repeated.fit_report["unique_measured_cells"] == 24


def test_synthetic_anchors_are_explicit_and_default_excluded(paired):
    x, y, g = paired
    xa = np.vstack((x, [[0, 0, 0], [1, 1, 1]]))
    ya = np.vstack((y, [[0, 0, 0], [1, 1, 1]]))
    mask = np.array([False] * 24 + [True] * 2)
    baseline = fit(x, y, g)
    without = fit_observations(xa, ya, g + ["anchor0", "anchor1"], {"degree": 1, "use_lut": False}, "reproducible-v1", mask)
    np.testing.assert_allclose(predict_(x, baseline), predict_(x, without), atol=1e-14, rtol=0)
    weighted = fit_observations(xa, ya, g + ["anchor0", "anchor1"], {"degree": 1},
                FittingPolicy("reproducible-v1", anchor_weight=2), mask)
    assert weighted.fit_report["synthetic_training_groups"] == 2
    assert np.max(np.abs(predict_(x, baseline) - predict_(x, weighted))) > 1e-5


def test_inconsistent_reference_within_cell_is_rejected(paired):
    x, y, g = paired
    with pytest.raises(ValueError, match="inconsistent"):
        fit(np.vstack((x, x[:1])), np.vstack((y, y[:1] + .1)), g + [g[0]])


@pytest.mark.parametrize("parameter,value", [("degree", 0), ("degree", 2.2), ("degree", True),
    ("max_iterations", -1), ("max_iterations", 0), ("random_state", -2), ("random_state", True),
    ("tol", 0), ("tol", np.nan), ("learning_rate", 0), ("batch_size", 0),
    ("patience", 0), ("hidden_layers", [0]), ("hidden_layers", []),
    ("dropout_rate", 1), ("optim_type", "invented"), ("device", "cuda"),
    ("param_search", True), ("use_lut", "true"), ("unknown", 123)])
def test_invalid_effective_controls_rejected(paired, parameter, value):
    x, y, g = paired
    with pytest.raises((ValueError, TypeError)):
        fit(x, y, g, **{parameter: value})


def test_rank_deficiency_and_bad_condition_rejected(paired):
    x, y, g = paired
    with pytest.raises(ValueError, match="rank deficient"):
        fit(np.tile(.5, (24, 3)), y, g)
    x[:, 2] = x[:, 0] + x[:, 1] * 1e-14
    with pytest.raises(ValueError, match="rank deficient"):
        fit(x, y, g)


def test_pls_components_bounded_by_centered_rank(paired):
    x, y, g = paired
    with pytest.raises(ValueError, match="ncomp"):
        fit(x, y, g, mtd="pls", ncomp=4)
    model = fit(x, y, g, mtd="pls", ncomp=3)
    assert model.ncomp == 3
    assert model.fit_report["effective_parameters"]["ncomp"] == 3


def test_mlp_controls_are_effective_and_no_row_holdout(paired):
    x, y, g = paired
    model = fit(x, y, g, mtd="nn", learning_rate=.012, hidden_layers=[5], max_iterations=4,
                batch_size=8, patience=3)
    assert model.model.learning_rate_init == .012
    assert model.model.batch_size == 8
    assert model.model.n_iter_no_change == 3
    assert not model.model.early_stopping
    second = fit(x, y, g, mtd="nn", learning_rate=.012, hidden_layers=[5], max_iterations=4,
                batch_size=8, patience=3)
    np.testing.assert_array_equal(predict_(x, model), predict_(x, second))


def test_custom_nn_scopes_rng_and_is_repeatable(paired):
    import torch
    x, y, g = paired
    state = torch.random.get_rng_state().clone()
    np_state = copy.deepcopy(np.random.get_state())
    kwargs = dict(mtd="custom", learning_rate=.01, hidden_layers=[5], max_iterations=4,
                  batch_size=8, patience=3, dropout_rate=.2)
    first, second = fit(x, y, g, **kwargs), fit(x, y, g, **kwargs)
    assert torch.equal(state, torch.random.get_rng_state())
    after = np.random.get_state()
    assert np_state[0] == after[0]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    np.testing.assert_array_equal(predict_(x, first), predict_(x, second))
    assert first.fit_report["estimator_precision"] == "CPU float32"
    assert "torch" in first.fit_report["dependencies"]
    assert first.fit_report["estimator_parameters"]["optimizer_defaults"]["lr"] == .01
    json.dumps(first.fit_report, allow_nan=False)
    split = first.fit_report["internal_split"]
    assert not set(split["training_groups"]) & set(split["validation_groups"])
    loaded = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(predict_(x, loaded), predict_(x, first))


def test_core_fit_dispatch_requires_real_groups(paired):
    x, y, g = paired
    with pytest.raises(ValueError, match="groups"):
        fit_model(x, y, {"fitting_policy": "reproducible-v1"})
    model = fit_model(x, y, {"fitting_policy": "reproducible-v1", "groups": g, "degree": 1})
    assert model.fitting_version == "reproducible-v1"


def test_conventional_root_setting_reaches_actual_basis(paired):
    import colour
    x, y, g = paired
    for root in (True, False):
        ccm, params, report = fit_conventional_observations(x, y, g,
            {"degree": 2, "root_polynomial_expansion": root}, "reproducible-v1")
        prediction = colour.characterisation.apply_matrix_colour_correction(x, ccm, **params)
        np.testing.assert_allclose(prediction, y - .01, atol=.02, rtol=0)
        assert params["root_polynomial_expansion"] is root
        assert report["design"]["rank"] == ccm.shape[1]


@pytest.mark.parametrize("parameters", [{"method": "Cheung 2004"}, {"degree": 5},
    {"root_polynomial_expansion": None}, {"terms": 10}])
def test_conventional_unsupported_parameters_rejected(paired, parameters):
    with pytest.raises((ValueError, TypeError)):
        fit_conventional_observations(*paired, parameters, "reproducible-v1")


@pytest.mark.parametrize("method,unused", [("linear", "learning_rate"), ("linear", "hidden_layers"),
    ("linear", "ncomp"), ("pls", "batch_size"), ("nn", "dropout_rate"), ("custom", "ncomp")])
def test_method_inapplicable_settings_rejected(paired, method, unused):
    with pytest.raises(ValueError, match="unsupported"):
        fit(*paired, mtd=method, **{unused: 1})


@pytest.mark.parametrize("key,value", [("anchor_weight", True), ("anchor_weight", "1"),
    ("anchor_weight", np.nan), ("condition_limit", True), ("condition_limit", "1"),
    ("condition_limit", np.inf)])
def test_policy_numbers_not_coerced_from_booleans_or_strings(key, value):
    with pytest.raises(ValueError):
        FittingPolicy("reproducible-v1", **{key: value})


def test_supported_methods_reproduce_in_fresh_processes():
    code = r'''
import hashlib, importlib.util, json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)
if os.environ.get('CCP_STAGED_FITTING_ROOT'):
    import ColorCorrectionPipeline, ColorCorrectionPipeline.core
    stage = Path(os.environ['CCP_STAGED_FITTING_ROOT'])
    for name, path in [('ColorCorrectionPipeline.core.correction', 'ColorCorrectionPipeline/core/correction.py'),
                       ('ColorCorrectionPipeline.fitting', 'ColorCorrectionPipeline/fitting.py')]:
        spec=importlib.util.spec_from_file_location(name, stage / path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from ColorCorrectionPipeline.fitting import fit_observations
from ColorCorrectionPipeline.core.correction import predict_
x=np.random.default_rng(91).uniform(.02,.98,(24,3));y=x*.9+.02;g=['g%d'%i for i in range(24)]
result={}
for method in ('linear','pls','nn','custom'):
    parameters={'mtd':method,'degree':1,'use_lut':False}
    if method!='linear': parameters.update(max_iterations=3)
    if method in ('nn','custom'): parameters.update(hidden_layers=[5],batch_size=8,patience=2)
    model=fit_observations(x,y,g,parameters,'reproducible-v1')
    result[method]=hashlib.sha256(predict_(x,model).tobytes()).hexdigest()
print(json.dumps(result,sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    first = subprocess.run([sys.executable, "-B", "-c", code], env=env, text=True, capture_output=True, check=True, timeout=90)
    second = subprocess.run([sys.executable, "-B", "-c", code], env=env, text=True, capture_output=True, check=True, timeout=90)
    assert json.loads(first.stdout.strip()) == json.loads(second.stdout.strip())


def test_cpu_custom_fit_ignores_non_cpu_torch_default_device(paired):
    import torch
    if not hasattr(torch, "get_default_device"):
        pytest.skip("this Torch version has no process default-device API")
    original = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        fitted = fit(*paired, mtd="custom", degree=1, max_iterations=2,
                     hidden_layers=[4], batch_size=8, patience=2)
        assert all(parameter.device.type == "cpu" for parameter in fitted.model.model.parameters())
        assert np.isfinite(predict_(paired[0], fitted)).all()
    finally:
        torch.set_default_device(original)


def test_custom_nn_failure_restores_rng_and_determinism_policy(paired, monkeypatch):
    import torch
    x, y, g = paired
    state = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    def fail(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")
    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        fit(x, y, g, mtd="custom", hidden_layers=[4], max_iterations=2)
    assert torch.equal(state, torch.random.get_rng_state())
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn_only == torch.is_deterministic_algorithms_warn_only_enabled()


def test_concurrent_supported_custom_fits_are_serialized_and_repeatable(paired):
    from concurrent.futures import ThreadPoolExecutor
    x, y, g = paired
    def run():
        return predict_(x, fit(x, y, g, mtd="custom", hidden_layers=[4], max_iterations=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: run(), range(2)))
    np.testing.assert_array_equal(first, second)


def test_custom_cpu_fit_ignores_global_dtype_and_does_not_seed_other_devices(paired, monkeypatch):
    import torch
    x, y, g = paired
    original_dtype = torch.get_default_dtype()
    baseline = fit(x, y, g, mtd="custom", hidden_layers=[4], max_iterations=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fitting must not call multi-device torch.manual_seed")
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    try:
        torch.set_default_dtype(torch.float64)
        changed = fit(x, y, g, mtd="custom", hidden_layers=[4], max_iterations=2)
    finally:
        torch.set_default_dtype(original_dtype)
    assert next(changed.model.model.parameters()).dtype == torch.float32
    assert changed.model.device == "cpu"
    np.testing.assert_array_equal(predict_(x, baseline), predict_(x, changed))


@pytest.mark.parametrize("conventional", [False, True])
@pytest.mark.parametrize("weight", [1e30, 1e200, 1e308])
def test_extreme_anchor_weight_cannot_conceal_bad_weighted_conditioning(paired, conventional, weight):
    x, y, groups = paired
    x = np.vstack((x, [[0., 0., 0.], [1., 1., 1.]]))
    y = np.vstack((y, [[0., 0., 0.], [1., 1., 1.]]))
    groups = groups + ["synthetic-black", "synthetic-white"]
    synthetic = np.array([False] * 24 + [True, True])
    policy = FittingPolicy("reproducible-v1", anchor_weight=weight)
    fitter = fit_conventional_observations if conventional else fit_observations
    with pytest.raises(ValueError, match="rank deficient|condition_limit"):
        fitter(x, y, groups, {"degree": 1}, policy, synthetic)
