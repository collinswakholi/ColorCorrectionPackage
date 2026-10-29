"""Opt-in real photograph fit/apply/persistence tests.

Set CCP_PHOTO_DIR to the reviewed five-photo corpus directory. The photos are
demonstration inputs, not independent physical color truth. Tests resize to a
1600-pixel longest edge to bound memory; model inference still uses its1280 canvas.
"""

import hashlib
import copy
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config


CORPUS = os.environ.get("CCP_PHOTO_DIR")
pytestmark = pytest.mark.skipif(not CORPUS, reason="set CCP_PHOTO_DIR to run real-photo integration")


def load(name):
    path = Path(CORPUS) / name
    bgr = cv2.imread(str(path))
    assert bgr is not None, str(path)
    scale = min(1.0, 1600 / max(bgr.shape[:2]))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr


@pytest.mark.parametrize("name", ["ColorCard_Big.JPG", "ColorCard_Combo.JPG", "Sample_3572.JPG", "Sample_3573.JPG"])
def test_real_photo_fit_apply_and_saved_reload(name, tmp_path):
    bgr = load(name)
    image = bgr[:, :, ::-1].astype(np.float64) / 255
    pipeline = ColorCorrection()
    selected = Config(do_ffc=False, check_saturation=False, save=True, save_path=str(tmp_path),
                      GC_kwargs={"get_deltaE": False}, WB_kwargs={"get_deltaE": False},
                      CC_kwargs={"mtd": "linear", "degree": 2, "n_samples": 9, "get_deltaE": False})
    _, stages, error = pipeline.run(image, name_="photo", config=selected)
    assert error is False, pipeline.last_run
    assert pipeline.last_run["success"] is True
    assert pipeline.last_run["detection"][0]["backend"] == "model"
    assert len(pipeline.last_run["detection"]) == 1  # one forward, fixed geometry
    assert pipeline.last_run["detection"][0]["sha256"] == "1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240"
    applied = pipeline.predict_image(image)["CC"]
    np.testing.assert_allclose(applied, stages["photo_CC"], atol=1e-12, rtol=1e-12)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "photo_models")
    np.testing.assert_array_equal(restored.predict_image(image)["CC"], applied)
    assert np.isfinite(applied).all() and applied.min() >= 0 and applied.max() <= 1


@pytest.mark.parametrize("cc_method", ["conv", "ours"])
def test_real_ffc_and_color_fit_save_apply(cc_method, tmp_path):
    bgr = load("ColorCard_Big.JPG")
    white = cv2.resize(load("Blank.JPG"), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = white.shape[:2]
    image = bgr[:, :, ::-1].astype(np.float64) / 255
    pipeline = ColorCorrection()
    config = Config(check_saturation=False, save=True, save_path=str(tmp_path),
                    FFC_kwargs={"crop_rect": [w // 10, h // 10, w * 9 // 10, h * 9 // 10],
                                "bins": 12, "degree": 2, "fit_method": "linear", "get_deltaE": True},
                    GC_kwargs={"get_deltaE": False}, WB_kwargs={"get_deltaE": False},
                    CC_kwargs={"cc_method": cc_method, "mtd": "linear", "degree": 1,
                               "n_samples": 9, "get_deltaE": False})
    _, stages, error = pipeline.run(image, White_Image=white, name_="ffc", config=config)
    assert error is False, pipeline.last_run
    assert all(pipeline.last_run["stages"][name]["status"] == "succeeded" for name in ("FFC", "GC", "WB", "CC"))
    np.testing.assert_allclose(pipeline.predict_image(image)["CC"], stages["ffc_CC"], atol=1e-12, rtol=1e-12)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "ffc_models")
    np.testing.assert_array_equal(restored.predict_image(image)["CC"], stages["ffc_CC"])
    no_metrics = copy.deepcopy(config)
    no_metrics.save = False
    no_metrics.FFC_kwargs["get_deltaE"] = False
    _, without_metrics, failed = pipeline.run(image, White_Image=white, name_="no_metrics", config=no_metrics)
    assert failed is False, pipeline.last_run
    assert pipeline.last_run["detection"][0]["geometry_source"] == "source_image"
    np.testing.assert_array_equal(without_metrics["no_metrics_CC"], stages["ffc_CC"])


def test_real_blank_failure_preserves_previous_fit_and_existing_model(tmp_path):
    good = load("ColorCard_Big.JPG")[:, :, ::-1].astype(np.float64) / 255
    pipeline = ColorCorrection()
    config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
                    save=True, save_path=str(tmp_path), CC_kwargs={"degree": 1, "n_samples": 4, "get_deltaE": False})
    assert pipeline.run(good, name_="cal", config=config)[2] is False
    previous = pipeline.models
    model_file = tmp_path / "cal_models.pkl"
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    blank = load("Blank.JPG")[:, :, ::-1].astype(np.float64) / 255
    assert pipeline.run(blank, name_="cal", config=config)[2] is True
    assert pipeline.models is previous
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == digest
    assert pipeline.last_run["errors"][0]["stage"] == "CC"
    assert pipeline.last_run["detection"][-1]["error_type"] == "NoChartDetectedError"


def test_real_model_can_load_and_predict_in_a_fresh_process(tmp_path):
    image = load("ColorCard_Big.JPG")[:, :, ::-1].astype(np.float64) / 255
    pipeline = ColorCorrection()
    config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
                    save=True, save_path=str(tmp_path), CC_kwargs={"degree": 1, "n_samples": 4, "get_deltaE": False})
    assert pipeline.run(image, name_="fresh", config=config)[2] is False
    expected = pipeline.predict_image(image[:16, :16])["CC"]
    np.save(tmp_path / "input.npy", image[:16, :16])
    code = """
import sys, numpy as np
from pathlib import Path
from ColorCorrectionPipeline import ColorCorrection
p=Path(sys.argv[1]); cc=ColorCorrection(); cc.models.load(str(p),'fresh_models')
np.save(p/'fresh_output.npy',cc.predict_image(np.load(p/'input.npy'))['CC'])
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], check=True, timeout=120,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    np.testing.assert_array_equal(np.load(tmp_path / "fresh_output.npy"), expected)
