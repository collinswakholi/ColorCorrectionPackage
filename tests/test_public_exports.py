"""Every name a module advertises in ``__all__`` must resolve."""
import importlib

import numpy as np
import pytest

MODULES = [
    "ColorCorrectionPipeline",
    "ColorCorrectionPipeline.core",
    "ColorCorrectionPipeline.core.accel",
    "ColorCorrectionPipeline.io",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_all_names_resolve(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in getattr(module, "__all__", ()) if not hasattr(module, name)]
    assert missing == []


def test_accel_star_import():
    namespace = {}
    exec("from ColorCorrectionPipeline.core.accel import *", namespace)
    assert "apply_srgb_to_linear_image" in namespace


def test_accel_srgb_helpers_round_trip_and_match_colour():
    colour = pytest.importorskip("colour")
    from ColorCorrectionPipeline.core.accel import (
        apply_linear_to_srgb_image,
        apply_srgb_to_linear_image,
    )

    srgb = np.linspace(0.0, 1.0, 257).reshape(-1, 1, 1).repeat(3, axis=2)
    linear = apply_srgb_to_linear_image(srgb)
    np.testing.assert_allclose(linear, colour.cctf_decoding(srgb, "sRGB"), atol=1e-12)
    np.testing.assert_allclose(apply_linear_to_srgb_image(linear), srgb, atol=1e-12)
    # Integer input is promoted rather than truncated.
    assert apply_srgb_to_linear_image(np.array([1], dtype=np.uint8)).dtype == np.float64
