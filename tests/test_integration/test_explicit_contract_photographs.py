"""Opt-in real-photo regression for explicit contracts and saved semantics."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData
from ColorCorrectionPipeline.colorimetry import srgb_to_linear

CORPUS = os.environ.get("CCP_PHOTO_DIR")
pytestmark = pytest.mark.skipif(not CORPUS, reason="set CCP_PHOTO_DIR for real-photo contract integration")


@pytest.mark.parametrize("name", ["ColorCard_Big.JPG", "ColorCard_Combo.JPG", "Sample_3572.JPG", "Sample_3573.JPG"])
def test_explicit_photo_fit_fixed_observations_common_white_and_reload(name, tmp_path):
    bgr = cv2.imread(str(Path(CORPUS) / name))
    scale = min(1.0, 1200 / max(bgr.shape[:2]))
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = ImageData.from_array(bgr, channel_order="BGR")
    cc = ColorCorrection()
    config = Config(do_ffc=False, check_saturation=False, save=True, save_path=str(tmp_path),
                    color_semantics={"version": "explicit-v1", "output_encoding": "linear-srgb"},
                    CC_kwargs={"degree": 2, "n_samples": 9, "get_deltaE": True})
    metrics, stages, failed = cc.run(image, name_="explicit", config=config)
    assert not failed, cc.last_run
    assert cc.last_run["color_semantics"]["adaptation"] == "Bradford"
    assert cc.image_data.source_dtype == "uint8"
    original = cc.observations.observations[0]
    for stage, observations in cc.stage_observations.items():
        after = observations.observations[0]
        assert after.cell_ids == original.cell_ids
        np.testing.assert_array_equal(after.quadrilaterals, original.quadrilaterals)
        after.validate_reference(cc.reference_profile)
        assert metrics["explicit_" + stage]["evaluation_white"] == str(cc.color_semantics.evaluation_white)
        assert metrics["explicit_" + stage]["reference_digest"] == cc.reference_profile.digest
    predicted = cc.predict_image(image)["CC"]
    np.testing.assert_allclose(predicted, stages["explicit_CC"], atol=1e-12, rtol=1e-12)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "explicit_models")
    assert restored.models.metadata["color_semantics"]["output_encoding"] == "linear-srgb"
    assert restored.models.metadata["detector_execution"][0]["execution_verified"] is True
    np.testing.assert_array_equal(restored.predict_image(image)["CC"], predicted)
    assert np.isfinite(predicted).all() and predicted.min() >= 0 and predicted.max() <= 1


def test_explicit_linear_input_matches_equivalent_typed_srgb_for_one_saved_model(tmp_path):
    bgr = cv2.imread(str(Path(CORPUS) / "ColorCard_Big.JPG"))
    scale = 800 / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    srgb = bgr[:, :, ::-1].astype(np.float64)/255
    linear = srgb_to_linear(srgb)
    cc = ColorCorrection()
    config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
                    color_semantics={"version": "explicit-v1", "input_encoding": "linear-srgb", "output_encoding": "linear-srgb"},
                    save=True, save_path=str(tmp_path), CC_kwargs={"degree": 1, "n_samples": 4, "get_deltaE": False})
    assert not cc.run(ImageData.from_array(linear, color_space="linear-srgb"), name_="linear", config=config)[2], cc.last_run
    raw_linear_result = cc.predict_image(linear)["CC"]
    typed_srgb_result = cc.predict_image(ImageData.from_array(srgb))["CC"]
    np.testing.assert_allclose(raw_linear_result, typed_srgb_result, atol=1e-12, rtol=1e-12)
    loaded = ColorCorrection()
    loaded.models.load(str(tmp_path), "linear_models")
    np.testing.assert_array_equal(loaded.predict_image(linear)["CC"], raw_linear_result)
