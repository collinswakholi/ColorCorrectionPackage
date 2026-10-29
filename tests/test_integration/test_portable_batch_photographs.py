"""Real-photo portable fit/reload, explicit file parity and streamed export."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData
from ColorCorrectionPipeline.io import read_image_data

CORPUS = os.environ.get("CCP_PHOTO_DIR")
pytestmark = pytest.mark.skipif(not CORPUS, reason="set CCP_PHOTO_DIR for portable/batch photo integration")


def photo(name):
    bgr = cv2.imread(str(Path(CORPUS) / name))
    scale = min(1., 900 / max(bgr.shape[:2]))
    return ImageData.from_array(cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), channel_order="BGR")


@pytest.mark.parametrize("name", ["ColorCard_Big.JPG", "ColorCard_Combo.JPG", "Sample_3572.JPG", "Sample_3573.JPG"])
def test_modern_photo_bundle_fit_apply_export_and_batch_reload(name, tmp_path):
    image = photo(name)
    pipeline = ColorCorrection()
    config = Config(do_ffc=False, color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1",
                    fitting_policy="reproducible-v1", CC_kwargs={"degree": 1}, chart_detection={"device": "cpu"},
                    save=True, model_format="bundle", save_path=str(tmp_path))
    _, stages, failed = pipeline.run(image, name_="portable", config=config)
    assert not failed, pipeline.last_run
    assert (tmp_path / "portable_models.ccpb").exists()
    assert not (tmp_path / "portable_models.pkl").exists()
    restored = ColorCorrection().load_bundle(tmp_path / "portable_models.ccpb")
    predicted = restored.predict_image(image)["CC"]
    np.testing.assert_allclose(predicted, stages["portable_CC"], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(predicted, pipeline.predict_image(image)["CC"])
    result = restored.predict_batch([image, tmp_path / "missing.png", image], ordered=True,
                                    output_dir=tmp_path / "exports", max_workers=2, max_in_flight=2)
    assert [item.status for item in result.items] == ["succeeded", "failed", "succeeded"]
    assert len({item.calibration_id for item in result.items}) == 1
    for item in (result.items[0], result.items[2]):
        exported = read_image_data(item.exports["CC"]["path"])
        np.testing.assert_allclose(exported.pixels, predicted, atol=1e-12, rtol=1e-12)
        assert exported.color_space == "srgb" and exported.source_dtype == "float64"
    saved = json.loads((Path(result.summary["output_directory"]) / "batch.json").read_text())
    assert saved["succeeded"] == 2 and saved["failed"] == 1


def test_explicit_uint16_file_and_array_entry_points_match(tmp_path):
    import tifffile
    image = photo("ColorCard_Big.JPG")
    integer = np.rint(image.pixels * 65535).astype(np.uint16)
    path = tmp_path / "source.tiff"
    tifffile.imwrite(path, integer, photometric="rgb")
    config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
                    color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1",
                    fitting_policy="reproducible-v1", CC_kwargs={"degree": 1}, chart_detection={"device": "cpu"})
    file_pipeline, array_pipeline = ColorCorrection(), ColorCorrection()
    assert not file_pipeline.run(path, config=config)[2], file_pipeline.last_run
    assert not array_pipeline.run(integer, config=config)[2], array_pipeline.last_run
    np.testing.assert_array_equal(file_pipeline.Image, array_pipeline.Image)
    np.testing.assert_array_equal(file_pipeline.predict_image(path)["CC"], file_pipeline.predict_image(integer)["CC"])
    np.testing.assert_allclose(file_pipeline.predict_image(path)["CC"], array_pipeline.predict_image(integer)["CC"], atol=1e-12, rtol=1e-12)
