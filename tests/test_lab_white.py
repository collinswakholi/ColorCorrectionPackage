"""Legacy Lab conversion must be consistent for non-D65 reference whites."""
import numpy as np
import pytest

from ColorCorrectionPipeline.constants import WP_DEFAULT
from ColorCorrectionPipeline.core.color_spaces import convert_to_lab

D50 = np.array([0.3457, 0.3585])


@pytest.mark.parametrize("white", [WP_DEFAULT, D50])
def test_srgb_white_is_neutral_under_reference_white(white):
    lab = convert_to_lab(np.array([[1.0, 1.0, 1.0]]), white, c_space="srgb")[0]
    np.testing.assert_allclose(lab, [100.0, 0.0, 0.0], atol=0.05)


def test_d65_output_unchanged():
    import colour

    rgb = np.random.default_rng(0).random((50, 3))
    expected = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb), WP_DEFAULT)
    np.testing.assert_array_equal(convert_to_lab(rgb, WP_DEFAULT, c_space="srgb"), expected)


def test_legacy_reference_chart_lab_matches_chart_definition_under_d50():
    import colour

    from ColorCorrectionPipeline import ColorCorrection

    cc = ColorCorrection()
    ref_rgb = cc.get_reference_values(D50).values if hasattr(cc, "get_reference_values") else None
    if ref_rgb is None:
        pytest.skip("reference API unavailable")
    chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    expected = colour.XYZ_to_Lab(colour.xyY_to_XYZ(list(chart.data.values())), chart.illuminant)
    got = convert_to_lab(ref_rgb, D50, c_space="srgb")
    # Patches inside the sRGB gamut must round-trip; allow a few out-of-gamut ones.
    err = np.linalg.norm(got - expected, axis=1)
    assert np.median(err) < 0.5
