"""Numeric helper contracts: rounding, scale handling and the saturation curve."""

import numpy as np
import pytest

from ColorCorrectionPipeline.core.color_spaces import extrapolate_if_sat_image
from ColorCorrectionPipeline.core.utils import to_float64, to_uint8

# --- to_uint8 / to_float64 (core) -------------------------------------------------


def test_core_to_uint8_rounds_to_nearest_code():
    codes = np.arange(256, dtype=np.float64)
    assert np.array_equal(to_uint8((codes + 0.4) / 255.0), codes.astype(np.uint8))
    assert np.array_equal(to_uint8((codes[:-1] + 0.6) / 255.0), (codes[:-1] + 1).astype(np.uint8))


def test_core_to_uint8_clips_and_round_trips():
    assert to_uint8(np.array([-0.2, 1.3])).tolist() == [0, 255]
    codes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(to_uint8(to_float64(codes)), codes)


def test_core_to_float64_explicit_scale_overrides_guess():
    dark = np.array([0.5, 0.9, 1.5])  # float data on a 0-1 scale with overshoot
    np.testing.assert_allclose(to_float64(dark, scale=1.0), dark)
    np.testing.assert_allclose(to_float64(np.array([127.5]), scale=255.0), [0.5])
    # Default behaviour (scale guessed from content) is unchanged.
    np.testing.assert_allclose(to_float64(np.array([0.25, 1.0])), [0.25, 1.0])


# --- legacy key_functions helpers --------------------------------------------------


def test_legacy_to_uint8_clips_and_rounds_instead_of_wrapping():
    from ColorCorrectionPipeline import key_functions as kf

    assert kf.to_uint8(np.array([1.01, -0.01, 0.999, 0.5])).tolist() == [255, 0, 255, 128]
    assert kf.to_uint8(np.array([7], dtype=np.uint8)).tolist() == [7]


def test_legacy_to_float64_scales_integer_types_by_their_range():
    from ColorCorrectionPipeline import key_functions as kf

    np.testing.assert_allclose(kf.to_float64(np.array([255], dtype=np.uint8)), [1.0])
    np.testing.assert_allclose(kf.to_float64(np.array([65535], dtype=np.uint16)), [1.0])


# --- saturation curve -----------------------------------------------------------------


def _curve(values):
    img = np.asarray(values, dtype=np.float64).reshape(-1, 1, 1).repeat(3, axis=2)
    out, _, _ = extrapolate_if_sat_image(img, None)
    return out[:, 0, 0]


def test_saturation_curve_is_continuous_and_monotonic():
    x = np.linspace(0.0, 1.0, 20001)
    y = _curve(x)
    assert np.all(np.diff(y) >= 0)
    assert np.max(np.abs(np.diff(y))) < 1e-3  # no jump at the 0.99 threshold
    assert y[-1] == pytest.approx(0.95)  # keeps the historical 5% headroom at full scale


def test_saturation_curve_leaves_unsaturated_channels_alone():
    img = np.array([[[1.0, 0.5, 0.2]], [[0.3, 0.3, 0.3]]])
    out, values, ids = extrapolate_if_sat_image(img, None)
    assert out[0, 0, 1] == 0.5 and out[0, 0, 2] == 0.2
    assert np.array_equal(out[1], img[1])
    assert ids.tolist() == [0]
    np.testing.assert_array_equal(values, img.reshape(-1, 3)[[0]])


def test_saturation_curve_identity_without_saturation():
    img = np.full((4, 4, 3), 0.8)
    out, values, ids = extrapolate_if_sat_image(img, np.zeros((24, 3)))
    assert out is img and values is None and ids is None
