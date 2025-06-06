import os
import cv2
import numpy as np
import pandas as pd
import pytest

from ColorCorrectionPipeline.core import ColorCorrection, Config
from ColorCorrectionPipeline.key_functions import to_uint8, to_float64

# Utility to create a synthetic "color checker"‐like image (random patches)
def synthetic_patch_image(width=320, height=240, patch_rows=4, patch_cols=6):
    """
    Generate a synthetic RGB image with patch_rows × patch_cols solid‐color patches.
    Each patch is filled with a random color in [0,1].
    Returns a float64 RGB image shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.float64)
    pr = patch_rows
    pc = patch_cols
    patch_h = height // pr
    patch_w = width // pc
    for i in range(pr):
        for j in range(pc):
            color = np.random.rand(3)
            y0 = i * patch_h
            x0 = j * patch_w
            img[y0 : y0 + patch_h, x0 : x0 + patch_w, :] = color
    return np.clip(img, 0.0, 1.0)

def synthetic_white_image(rgb_img):
    """
    Given an RGB float64 [0,1] image, return a matching uint8 BGR white image.
    """
    h, w, _ = rgb_img.shape
    return np.ones((h, w, 3), dtype=np.uint8) * 255

def test_full_pipeline_no_errors(tmp_path):
    """
    Ensure that running the full pipeline on a synthetic patch image + white image
    completes without errors, and returns images of correct shape and dtype.
    """
    # Create synthetic images
    img_rgb = synthetic_patch_image()  # float64 in [0,1], shape (240,320,3)
    white_bgr = synthetic_white_image(img_rgb)  # uint8 white image

    # Minimal kwargs (all defaults); no actual fitting will fail because patch values are arbitrary
    ffc_kwargs = {
        "model_path": "",       # force manual crop → user would be prompted, but override to skip
        "manual_crop": True,    # skip YOLO and use entire image
        "show": False,
        "bins": 10,
        "smooth_window": 3,
        "get_deltaE": False,
        "fit_method": "linear",
        "interactions": False,
        "max_iter": 10,
        "tol": 1e-8,
        "verbose": False,
        "random_seed": 0,
    }
    gc_kwargs = {"max_degree": 2, "show": False, "get_deltaE": False}
    wb_kwargs = {"show": False, "get_deltaE": False}
    cc_kwargs = {
        "mtd": "linear",
        "degree": 2,
        "max_iterations": 10,
        "random_state": 0,
        "tol": 1e-8,
        "verbose": False,
        "ncomp": 1,
        "nlayers": 10,
        "param_search": False,
        "hidden_layers": [8],
        "learning_rate": 0.001,
        "batch_size": 4,
        "patience": 2,
        "dropout_rate": 0.1,
        "use_batch_norm": False,
        "show": False,
        "get_deltaE": False,
        "n_samples": 5,
    }

    config = Config(
        do_ffc=True,
        do_gc=True,
        do_wb=True,
        do_cc=True,
        save=False,
        save_path=str(tmp_path),
        check_saturation=False,
        REF_ILLUMINANT=None,
        FFC_kwargs=ffc_kwargs,
        GC_kwargs=gc_kwargs,
        WB_kwargs=wb_kwargs,
        CC_kwargs=cc_kwargs,
    )

    cc = ColorCorrection()
    metrics, images, errors = cc.run(
        Image=img_rgb,
        White_Image=white_bgr,
        name_="synthetic_test",
        config=config,
    )
    assert errors is False, "Pipeline encountered an unexpected error."

    # Check that each stage image exists and has correct dtype and shape
    for step in ["synthetic_test_FFC", "synthetic_test_GC", "synthetic_test_WB", "synthetic_test_CC"]:
        assert step in images, f"Missing output for step: {step}"
        out_img = images[step]
        assert isinstance(out_img, np.ndarray)
        assert out_img.dtype == np.float64
        assert out_img.shape == img_rgb.shape

def test_predict_chain(monkeypatch):
    """
    Manually set models in ColorCorrection.models, then run predict_image
    on a random float64 RGB image and ensure outputs are valid [0,1] arrays.
    """
    cc = ColorCorrection()

    # Assign dummy models:
    # - FFC: identity multiplier (all ones)
    cc.models.model_ffc = np.ones((10, 10))  # shape doesn't match, but our monkeypatch bypasses resize prompt
    # - GC: None (skip)
    cc.models.model_gc = None
    # - WB: identity diagonal matrix
    cc.models.model_wb = np.eye(3)
    # - CC: identity matrix in "conv" mode
    dummy_ccm = np.eye(3)
    cc.models.model_cc = (dummy_ccm, {}, "conv")

    # Create a random RGB float64 image
    img = np.random.rand(64, 64, 3).astype(np.float64)

    # Monkeypatch FlatFieldCorrection.apply_ffc to simply return the input BGR unchanged
    class DummyFFC:
        def apply_ffc(self, img, multiplier):
            # Return the passed img as-is (uint8)
            return img

    monkeypatch.setattr(
        "color_correction_pipeline.core.FlatFieldCorrection", DummyFFC
    )

    # Also monkeypatch to_uint8 and to_float64 to ensure compatibility
    # (But in this test, the original to_uint8/to_float64 would work fine.)

    results = cc.predict_image(img, show=False)

    # The returned dictionary should have keys "FFC","GC","WB","CC"
    for key in ["FFC", "GC", "WB", "CC"]:
        assert key in results, f"Missing key {key} in predict results"
        out = results[key]
        # Even if CC or GC is None, ensure dtype and range if array
        if out is not None:
            assert isinstance(out, np.ndarray)
            assert out.dtype == np.float64
            assert np.all(out >= 0.0) and np.all(out <= 1.0)
