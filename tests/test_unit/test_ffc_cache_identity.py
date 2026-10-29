"""A candidate calibration must not reuse gains for different inputs/assets."""

import numpy as np

from ColorCorrectionPipeline import ColorCorrection


def test_white_pixel_outside_old_thumbnail_changes_cache_identity():
    first = np.zeros((64, 256, 3), dtype=np.uint8)
    second = first.copy()
    second[1, 1] = 1
    assert ColorCorrection._hash_image_for_cache(first) != ColorCorrection._hash_image_for_cache(second)


def test_changing_detector_bytes_at_same_path_invalidates_ffc_key(tmp_path):
    pipeline = ColorCorrection()
    pipeline.White_Image = np.full((8, 8, 3), 200, dtype=np.uint8)
    model = tmp_path / "plane.onnx"
    params = {"model_path": str(model), "manual_crop": False, "crop_rect": None}
    model.write_bytes(b"first detector bytes")
    first = pipeline._ffc_cache_key(params, {"degree": 2})
    model.write_bytes(b"other detector bytes")
    assert pipeline._ffc_cache_key(params, {"degree": 2}) != first


def test_unused_detector_does_not_affect_manual_crop_key(tmp_path):
    pipeline = ColorCorrection()
    pipeline.White_Image = np.full((8, 8, 3), 200, dtype=np.uint8)
    model = tmp_path / "plane.onnx"
    params = {"model_path": str(model), "manual_crop": False, "crop_rect": [0, 0, 8, 8]}
    model.write_bytes(b"first detector bytes")
    first = pipeline._ffc_cache_key(params, {"degree": 2})
    model.write_bytes(b"other detector bytes")
    assert pipeline._ffc_cache_key(params, {"degree": 2}) == first
