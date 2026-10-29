"""Failure, reuse and concurrent-read regressions for the run transaction."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.models import MyModels


def config(**changes):
    values = dict(do_ffc=False, do_gc=False, do_wb=False, do_cc=False, check_saturation=False)
    values.update(changes)
    return Config(**values)


def prior_pipeline(cls=ColorCorrection):
    pipeline = cls()
    pipeline.get_reference_values()
    pipeline.models.model_wb = np.eye(3) * 1.5
    pipeline.Image = np.full((4, 4, 3), 0.2)
    pipeline.White_Image = np.full((4, 4, 3), 200, dtype=np.uint8)
    pipeline.Models_path = "previous-model.pkl"
    return pipeline


def test_successful_disabled_run_clears_old_models():
    pipeline = prior_pipeline()
    image = np.full((4, 4, 3), 0.4)
    previous = pipeline.models
    assert pipeline.run(image, config=config())[2] is False
    assert pipeline.models is not previous
    assert not pipeline.models.has_models()
    assert pipeline.Models_path is None
    assert pipeline.last_run["success"] is True
    assert all(s["status"] == "disabled" for s in pipeline.last_run["stages"].values())
    np.testing.assert_array_equal(pipeline.predict_image(image)["CC"], image)


def test_failed_run_keeps_complete_previous_state_and_caller_pixels(tmp_path):
    class FailedWB(ColorCorrection):
        def do_white_balance(self, Image, do_wb=True, wb_kwargs=None):
            self.models.model_wb = np.eye(3) * 0.5
            Image[:] = 0
            return Image, None, True

    pipeline = prior_pipeline(FailedWB)
    attributes = ("models", "Image", "White_Image", "Models_path", "REF_ILLUMINANT",
                  "REFERENCE_CHART", "REFERENCE_RGB_PD", "REFERENCE_NEUTRAL_PATCHES_PD")
    previous = {name: getattr(pipeline, name) for name in attributes}
    image = np.full((4, 4, 3), 0.4)
    selected = config(do_wb=True, save=True, save_path=str(tmp_path), REF_ILLUMINANT=np.array([0.34567, 0.35850]))
    _, _, failed = pipeline.run(image, name_="failed", config=selected)
    assert failed is True
    for name, value in previous.items():
        assert getattr(pipeline, name) is value
    np.testing.assert_array_equal(image, np.full_like(image, 0.4))
    np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.6)
    assert pipeline.last_run["stages"]["WB"]["status"] == "failed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("white", [None, "definitely-missing-white.jpg"])
def test_required_white_input_fails_without_mutating_config(white):
    pipeline = prior_pipeline()
    previous = pipeline.models
    selected = config(do_ffc=True)
    assert pipeline.run(np.full((4, 4, 3), 0.4), White_Image=white, config=selected)[2] is True
    assert selected.do_ffc is True
    assert pipeline.models is previous
    assert pipeline.last_run["stages"]["FFC"]["status"] == "failed"
    assert pipeline.last_run["errors"][0]["message"]


@pytest.mark.parametrize("stage", ["GC", "WB", "CC"])
def test_missing_chart_is_not_a_successful_identity_fit(stage, monkeypatch):
    monkeypatch.setattr("ColorCorrectionPipeline.core.correction.extract_neutral_patches", lambda *a, **k: (None, None))
    pipeline = prior_pipeline()
    previous = pipeline.models
    changes = {"do_" + stage.lower(): True}
    if stage == "CC":
        changes["CC_kwargs"] = {"cc_method": "conv", "degree": 1, "get_deltaE": False}
    assert pipeline.run(np.full((4, 4, 3), 0.4), config=config(**changes))[2] is True
    assert pipeline.models is previous
    assert "requires a detected Classic24" in pipeline.last_run["errors"][0]["message"]


def test_unknown_method_is_reported_as_failure():
    pipeline = prior_pipeline()
    previous = pipeline.models
    assert pipeline.run(np.full((4, 4, 3), 0.4), config=config(do_cc=True, CC_kwargs={"cc_method": "typo"}))[2] is True
    assert pipeline.models is previous
    assert "Invalid color-correction method" in pipeline.last_run["errors"][0]["message"]


def test_unknown_regression_method_is_rejected():
    from ColorCorrectionPipeline.core.correction import fit_model
    x = np.random.default_rng(3).random((12, 3))
    with pytest.raises(ValueError, match="Unknown regression method"):
        fit_model(x, x, {"mtd": "typo", "degree": 1})


def test_saturation_exception_fails_and_does_not_save(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("saturation fixture failure")
    monkeypatch.setattr("ColorCorrectionPipeline.pipeline.extrapolate_if_sat_image", broken)
    pipeline = prior_pipeline()
    previous = pipeline.models
    assert pipeline.run(np.ones((4, 4, 3)), config=config(check_saturation=True, save=True, save_path=str(tmp_path)))[2] is True
    assert pipeline.models is previous
    assert pipeline.last_run["errors"][0]["stage"] == "saturation"
    assert not list(tmp_path.iterdir())


def test_prediction_sees_previous_calibration_until_successful_commit():
    entered, release = threading.Event(), threading.Event()

    class BlockingWB(ColorCorrection):
        def do_white_balance(self, Image, do_wb=True, wb_kwargs=None):
            self.models.model_wb = np.eye(3) * 0.5
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test timed out")
            return Image * 0.5, {}, False

    pipeline = prior_pipeline(BlockingWB)
    image = np.full((4, 4, 3), 0.4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fitting = pool.submit(pipeline.run, image, config=config(do_wb=True))
        try:
            assert entered.wait(10)
            np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.6)
        finally:
            release.set()
        assert fitting.result(timeout=10)[2] is False
    np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.2)


def test_whole_batch_uses_one_calibration_even_when_progress_refits():
    release = threading.Event()

    class PausingPrediction(ColorCorrection):
        def _predict_image(self, Image, show=False):
            if Image[0, 0, 0] > 0.4 and not release.wait(10):
                raise RuntimeError("test timed out")
            return super()._predict_image(Image, show)

    pipeline = prior_pipeline(PausingPrediction)
    images = [np.full((4, 4, 3), 0.4 + index * 0.01) for index in range(4)]

    def progress(completed, total, name):
        if completed == 1:
            try:
                assert pipeline.run(images[0], config=config())[2] is False
            finally:
                release.set()

    results = pipeline.predict_images(images, on_progress=progress, max_workers=1)
    for image, result in zip(images, results):
        np.testing.assert_allclose(result["CC"], image * 1.5)
    assert not pipeline.models.has_models()


def test_ffc_does_not_require_chart_for_optional_metrics(monkeypatch):
    from ColorCorrectionPipeline.detection import NoChartDetectedError

    class UnitGain:
        def __init__(self, image, **kwargs):
            self.image = image

        def compute_multiplier(self, **kwargs):
            return np.ones(self.image.shape[:2])

    def no_chart(*a, **k):
        raise NoChartDetectedError("no chart in scene")

    monkeypatch.setattr("ColorCorrectionPipeline.pipeline.FlatFieldCorrection", UnitGain)
    monkeypatch.setattr("ColorCorrectionPipeline.pipeline.extract_neutral_patches", no_chart)
    pipeline = ColorCorrection()
    _, _, failed = pipeline.run(np.full((4, 4, 3), 0.4), White_Image=np.full((4, 4, 3), 200, dtype=np.uint8), config=config(do_ffc=True))
    assert failed is False
    assert pipeline.models.model_ffc is not None
    assert pipeline.last_run["warnings"][0]["metrics"] == "unavailable"


def test_save_exception_preserves_active_state_and_existing_outputs(tmp_path, monkeypatch):
    class SuccessfulWB(ColorCorrection):
        def do_white_balance(self, Image, do_wb=True, wb_kwargs=None):
            self.models.model_wb = np.eye(3)
            return Image, {"fixture": 1.0}, False

    def broken(*a, **k):
        raise OSError("disk fixture failure")

    pipeline = prior_pipeline(SuccessfulWB)
    previous = pipeline.models
    destination = tmp_path / "case_models.pkl"
    destination.write_bytes(b"previous model")
    metrics = tmp_path / "case_metrics.csv"
    metrics.write_bytes(b"previous metrics")
    monkeypatch.setattr(MyModels, "save", broken)
    with pytest.raises(OSError, match="disk fixture failure"):
        pipeline.run(np.full((4, 4, 3), 0.4), name_="case", config=config(do_wb=True, save=True, save_path=str(tmp_path)))
    assert pipeline.models is previous
    assert destination.read_bytes() == b"previous model"
    assert metrics.read_bytes() == b"previous metrics"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["case_metrics.csv", "case_models.pkl"]
    assert pipeline.last_run["success"] is False


def test_config_copies_nested_values_and_roundtrip_inputs():
    supplied = {"hidden_layers": [4, 8]}
    selected = config(CC_kwargs=supplied)
    supplied["hidden_layers"][0] = 99
    assert selected.CC_kwargs["hidden_layers"] == [4, 8]
    serialized = selected.to_dict()
    restored = Config.from_dict(serialized)
    serialized_before = copy.deepcopy(serialized)
    restored.CC_kwargs["hidden_layers"].append(16)
    assert serialized == serialized_before
    assert selected.CC_kwargs["hidden_layers"] == [4, 8]


@pytest.mark.parametrize("image", [np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4, 3), np.nan), np.full((4, 4, 3), 1.1), np.zeros((4, 4))])
def test_invalid_input_preserves_calibration(image):
    pipeline = prior_pipeline()
    previous = pipeline.models
    with pytest.raises((TypeError, ValueError)):
        pipeline.run(image, config=config())
    assert pipeline.models is previous
    assert pipeline.last_run["success"] is False
    assert pipeline.last_run["errors"]
