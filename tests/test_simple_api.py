"""The two-call API must be a faithful shortcut for the full Config/ColorCorrection path."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline import (
    CalibrationError,
    Calibration,
    CCOptions,
    ColorCorrection,
    Config,
    FFCOptions,
    Steps,
    calibrate,
    correct,
)

IMAGES = Path(__file__).resolve().parents[1] / "Images"
CHART, WHITE, SCENE = (str(IMAGES / n) for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("simple") / "cal.ccpb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = calibrate(CHART, white=WHITE, device="cpu", save_to=bundle)
    return cal, bundle


def test_calibrate_returns_metrics_and_writes_bundle(calibration):
    cal, bundle = calibration
    assert isinstance(cal, Calibration)
    assert bundle.is_file()
    assert cal.delta_e == pytest.approx(cal.metrics["chart_CC"]["delta_e00_after_mean"])
    assert 0 < cal.delta_e < 5


def test_correct_is_the_same_for_calibration_pipeline_and_bundle(calibration):
    cal, bundle = calibration
    direct = cal.pipeline.predict_image(SCENE, stages="final")["CC"]
    np.testing.assert_array_equal(correct(SCENE, cal), direct)
    np.testing.assert_array_equal(cal.correct(SCENE), direct)
    np.testing.assert_array_equal(correct(SCENE, cal.pipeline), direct)
    np.testing.assert_array_equal(correct(SCENE, bundle), direct)
    assert direct.dtype.kind == "f" and 0.0 <= direct.min() and direct.max() <= 1.0


def test_calibrate_matches_the_equivalent_full_config(calibration):
    cal, _ = calibration
    config = Config(
        model="rpcc2",
        input={"frame_index": 0},
        chart={"device": "cpu"},
        steps=Steps(ffc=FFCOptions(enabled=True), cc=CCOptions(fit={"model": "rpcc2"})),
    )
    manual = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manual.fit(CHART, WHITE, name="chart", config=config)
    np.testing.assert_array_equal(manual.predict(SCENE, stages="final")["CC"], correct(SCENE, cal))


def test_fit_and_predict_aliases_match_original_methods():
    config = Config.from_preset("ccm", white=False, device="cpu")
    a, b = ColorCorrection(), ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa = a.fit(CHART, name="chart", config=config)
        fb = b.fit_or_raise(CHART, None, name_="chart", config=config)
    assert (
        fa.metrics["chart_CC"]["delta_e00_after_mean"]
        == fb.metrics["chart_CC"]["delta_e00_after_mean"]
    )
    np.testing.assert_array_equal(a.predict(SCENE)["CC"], b.predict_image(SCENE)["CC"])


@pytest.mark.parametrize(
    "name, model, precision",
    [
        ("rpcc2", "rpcc2", "auto"),
        ("rpcc3", "rpcc3", "auto"),
        ("ccm", "ccm", "auto"),
        ("3x3", "ccm", "auto"),
        ("PLS", "pls", "auto"),
        ("high-precision", "rpcc2", "high"),
    ],
)
def test_presets(name, model, precision):
    config = Config.from_preset(name)
    assert config.CC_kwargs["model"] == model
    assert config.precision == precision
    assert config.image_options == {"frame_index": 0}
    assert config.do_ffc is True


def test_preset_overrides_merge_grouped_options():
    config = Config.from_preset(
        "rpcc2",
        white=False,
        device="cpu",
        chart={"samples_per_patch": 20, "sampling_seed": 42},
        frame_index=None,
    )
    assert config.do_ffc is False
    assert config.chart_detection["device"] == "cpu"
    assert config.chart_detection["samples_per_patch"] == 20
    assert "frame_index" not in config.image_options


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        Config.from_preset("rpcc9")


def test_calibrate_rejects_config_plus_options():
    with pytest.raises(TypeError):
        calibrate(CHART, config=Config.from_preset(), numerics={"precision": "high"})


def test_no_chart_raises_calibration_error():
    blank = Path(__file__).resolve().parents[2] / "Test_Packages" / "Test_images1" / "Blank.JPG"
    if not blank.is_file():
        pytest.skip("sample without a chart is not available")
    with warnings.catch_warnings(), pytest.raises(CalibrationError):
        warnings.simplefilter("ignore")
        calibrate(str(blank), device="cpu")


def test_multi_frame_error_explains_the_fix():
    with warnings.catch_warnings(), pytest.raises(ValueError, match="frame_index"):
        warnings.simplefilter("ignore")
        ColorCorrection().fit(CHART, config=Config(model="ccm", chart={"device": "cpu"}))
