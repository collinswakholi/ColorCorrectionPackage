"""Real-photo qualification of opt-in numerical stages, not physical accuracy."""
import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData

CORPUS = os.environ.get("CCP_PHOTO_DIR")
pytestmark = pytest.mark.skipif(not CORPUS, reason="set CCP_PHOTO_DIR for real-photo modern pipeline integration")


def load_photo(name):
    bgr = cv2.imread(str(Path(CORPUS) / name))
    scale = min(1., 1000 / max(bgr.shape[:2]))
    return ImageData.from_array(cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), channel_order="BGR")


@pytest.mark.parametrize("name", ["ColorCard_Big.JPG", "ColorCard_Combo.JPG", "Sample_3572.JPG", "Sample_3573.JPG"])
def test_modern_photo_full_fit_apply_reload_and_accounted_validation(name, tmp_path):
    image = load_photo(name)
    cc = ColorCorrection()
    cfg = Config(do_ffc=False, color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1",
                 fitting_policy="reproducible-v1", CC_kwargs={"degree": 1}, chart_detection={"device": "cpu"},
                 save=True, save_path=str(tmp_path))
    _, images, failed = cc.run(image, name_="modern", config=cfg)
    assert not failed, cc.last_run
    report = cc.numerical_report
    validation = report["CC"]["validation"]
    assert validation["expected_groups"] == 24
    assert validation["expected_groups"] == validation["scored_groups"] + validation["excluded_groups"] + validation["failed_groups"]
    assert validation["status"] == "available", validation
    assert validation["whole_pipeline"] is False
    original = cc.observations.observations[0]
    for observation in cc.stage_observations.values():
        np.testing.assert_array_equal(original.quadrilaterals, observation.observations[0].quadrilaterals)
    prediction = cc.predict_image(image)
    for stage in ("GC", "WB", "CC"):
        np.testing.assert_allclose(prediction[stage], images["modern_" + stage], atol=1e-12, rtol=1e-12)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "modern_models")
    np.testing.assert_array_equal(restored.predict_image(image)["CC"], prediction["CC"])
    ramp = cc.models.model_gc.predict(np.linspace(0, 100, 10001))
    assert ramp[0] == 0 and ramp[-1] == 100 and np.all(np.diff(ramp) >= -1e-12)


def test_modern_photo_guarded_ffc_fit_and_application_match(tmp_path):
    image = load_photo("ColorCard_Big.JPG")
    h, w = image.shape[:2]
    white_luma = np.rint(170 + 20*np.linspace(0, 1, h)[:, None] + 10*np.linspace(0, 1, w)[None, :]).astype(np.uint8)
    white = np.repeat(white_luma[..., None], 3, axis=2)
    cc = ColorCorrection()
    cfg = Config(color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1", fitting_policy="reproducible-v1",
                 CC_kwargs={"degree": 1}, chart_detection={"device": "cpu"},
                 FFC_kwargs={"crop_rect": [0, 0, w, h], "bins": 15, "degree": 2},
                 save=True, save_path=str(tmp_path))
    _, images, failed = cc.run(image, white, "ffc", cfg)
    assert not failed, cc.last_run
    assert cc.numerical_report["FFC"]["plane_selection"] == "explicit-crop"
    assert cc.numerical_report["FFC"]["minimum_gain"] > 0
    np.testing.assert_allclose(cc.predict_image(image)["CC"], images["ffc_CC"], atol=1e-12, rtol=1e-12)


def test_modern_blank_is_failure_with_no_active_calibration():
    cc = ColorCorrection()
    cfg = Config(do_ffc=False, color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1",
                 fitting_policy="reproducible-v1", chart_detection={"device": "cpu"})
    assert cc.run(load_photo("Blank.JPG"), config=cfg)[2]
    assert not cc.models.has_models()
    assert cc.last_run["errors"]
