"""Regression tests for trusted legacy model persistence and failure recovery."""

import pickle

import numpy as np
import pytest

import ColorCorrectionPipeline.models as models_module
from ColorCorrectionPipeline.core.correction import fit_model, predict_
from ColorCorrectionPipeline.models import MyModels


def _active_models():
    active = MyModels()
    active.model_ffc = np.ones((3, 4))
    active.model_gc = np.array([1.0, 0.0])
    active.model_wb = np.eye(3) * 0.95
    active.model_cc = (np.eye(3), {"degree": 1}, "conv")
    return active


@pytest.mark.parametrize("method", ["linear", "pls", "nn", "custom"])
@pytest.mark.parametrize("build_before_save", [False, True])
def test_fitted_models_round_trip_with_lazy_or_built_lut(
    tmp_path, method, build_before_save
):
    rng = np.random.default_rng(17)
    measured = rng.uniform(0.05, 0.95, (32, 3))
    reference = measured * np.array([0.8, 0.9, 0.85]) + 0.03
    params = {
        "mtd": method, "degree": 1, "ncomp": 2,
        "hidden_layers": [4], "max_iterations": 20 if method == "nn" else 2,
        "batch_size": 8, "use_batch_norm": False, "dropout_rate": 0,
        "lut_grid_size": 5, "lut_min_pixels": 10, "lazy_lut": True,
    }
    fitted = fit_model(measured, reference, params)
    image = measured[:16].reshape(4, 4, 3)
    direct_before = predict_(measured[:4], fitted)
    if build_before_save:
        predict_(image, fitted)
    assert (fitted._lut is not None) == build_before_save

    active = _active_models()
    active.model_cc = (fitted, params, "ours")
    active.save(str(tmp_path), "calibration")
    loaded = MyModels()
    loaded.load(str(tmp_path), "calibration")
    restored = loaded.model_cc[0]

    np.testing.assert_allclose(predict_(measured[:4], restored), direct_before, atol=1e-12)
    np.testing.assert_allclose(predict_(image, restored), predict_(image, fitted), atol=1e-12)
    np.testing.assert_array_equal(loaded.model_ffc, active.model_ffc)
    np.testing.assert_array_equal(loaded.model_gc, active.model_gc)
    np.testing.assert_array_equal(loaded.model_wb, active.model_wb)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failure", ["serialization", "fsync", "readback", "replace"])
@pytest.mark.parametrize("existing_destination", [False, True])
def test_failed_save_preserves_destination_and_cleans_temporary_file(
    tmp_path, monkeypatch, failure, existing_destination
):
    active = _active_models()
    destination = tmp_path / "models.pkl"
    original_bytes = b"previous successful calibration"
    if existing_destination:
        destination.write_bytes(original_bytes)

    def fail(*args, **kwargs):
        raise OSError("injected persistence failure")

    if failure == "serialization":
        def partial_dump(obj, stream, **kwargs):
            stream.write(b"partial serialization")
            raise ValueError("injected serialization failure")
        monkeypatch.setattr(models_module.pickle, "dump", partial_dump)
    elif failure == "fsync":
        monkeypatch.setattr(models_module.os, "fsync", fail)
    elif failure == "readback":
        monkeypatch.setattr(models_module.pickle, "load", fail)
    else:
        monkeypatch.setattr(models_module.os, "replace", fail)

    with pytest.raises(OSError, match="Failed to save models"):
        active.save(str(tmp_path))

    if existing_destination:
        assert destination.read_bytes() == original_bytes
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.iterdir())) == int(existing_destination)


def test_save_atomically_replaces_existing_valid_file(tmp_path):
    old = _active_models()
    old.save(str(tmp_path))
    candidate = _active_models()
    candidate.model_wb = np.eye(3) * 0.8
    candidate.save(str(tmp_path))

    loaded = MyModels()
    loaded.load(str(tmp_path))
    np.testing.assert_array_equal(loaded.model_wb, candidate.model_wb)
    assert list(tmp_path.iterdir()) == [tmp_path / "models.pkl"]


@pytest.mark.parametrize("invalid_file", ["truncated", "wrong_type", "incomplete", "missing"])
def test_failed_load_keeps_every_active_stage(tmp_path, invalid_file):
    active = _active_models()
    prior_state = vars(active).copy()
    destination = tmp_path / "models.pkl"
    if invalid_file == "truncated":
        destination.write_bytes(b"\x80\x05partial pickle")
    elif invalid_file == "wrong_type":
        destination.write_bytes(pickle.dumps({"model_ffc": None}))
    elif invalid_file == "incomplete":
        incomplete = MyModels()
        del incomplete.model_cc
        destination.write_bytes(pickle.dumps(incomplete))

    with pytest.raises(OSError):
        active.load(str(tmp_path))

    assert vars(active).keys() == prior_state.keys()
    for name, value in prior_state.items():
        assert vars(active)[name] is value


def test_successful_load_clears_previously_active_disabled_stages(tmp_path):
    MyModels().save(str(tmp_path))
    active = _active_models()
    active.load(str(tmp_path))
    assert not active.has_models()
    assert all(getattr(active, name) is None for name in ("model_ffc", "model_gc", "model_wb", "model_cc"))
    assert active.metadata == {}
