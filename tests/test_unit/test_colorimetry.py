"""Independent vectors, explicit-domain failures, and common-white invariants."""

import json
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline.colorimetry import (
    ColorSemantics, D50, D65, adapt_xyz, adaptation_matrix, delta_e00,
    delta_e00_reference, display_reference_rgb, lab_to_xyz,
    linear_to_srgb, reference_to_working_lab, reference_to_working_rgb, reference_values_to_lab_d50,
    rgb_to_xyz, rgb_to_xyz_matrix, srgb_to_linear, white_xy, white_xyz,
    xyz_to_lab, xyz_to_rgb,
)


@pytest.mark.parametrize("encoded,linear", [(0., 0.), (0.04045, 0.0031308049535603713),
                                           (0.5, 0.21404114048223255), (1., 1.)])
def test_published_srgb_transfer_values(encoded, linear):
    np.testing.assert_allclose(srgb_to_linear(encoded), linear, atol=1e-15, rtol=0)


def test_transfer_roundtrip_and_signed_extension():
    samples = np.array([-1.2, -0.1, -0.003, 0., 0.0031308, 0.5, 1., 1.2])
    encoded = linear_to_srgb(samples)
    np.testing.assert_allclose(srgb_to_linear(encoded, domain="extended"), samples, atol=3e-8, rtol=0)
    np.testing.assert_allclose(encoded[0], -encoded[-1], atol=0, rtol=0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        srgb_to_linear(encoded)
    with pytest.raises(ValueError, match="outside"):
        linear_to_srgb(samples, policy="raise")
    assert linear_to_srgb(samples, policy="clip").min() == 0
    assert linear_to_srgb(samples, policy="clip").max() == 1


def test_published_srgb_matrix_with_documented_white_rounding():
    # W3C CSS Color4 normative-D65 matrix rounded to 7 decimals. The contract
    # intentionally retains CIE D65(.31271,.32902), so allow stated white rounding.
    published = [[0.4123908, 0.3575843, 0.1804808], [0.2126390, 0.7151687, 0.0721923],
                 [0.0193308, 0.1191948, 0.9505322]]
    np.testing.assert_allclose(rgb_to_xyz_matrix(), published, atol=2e-4, rtol=0)
    np.testing.assert_allclose(rgb_to_xyz([1., 1., 1.]), white_xyz(D65), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(rgb_to_xyz([0., 0., 0.]), [0., 0., 0.])


@pytest.mark.parametrize("encoding", ["srgb", "linear-srgb"])
def test_rgb_xyz_roundtrip_preserves_shape_and_precision(encoding):
    values = np.random.default_rng(0).uniform(0.001, 0.999, (3, 4, 3))
    np.testing.assert_allclose(xyz_to_rgb(rgb_to_xyz(values, encoding=encoding), encoding=encoding), values, atol=2e-14, rtol=0)
    assert rgb_to_xyz(values, encoding=encoding).shape == values.shape


@pytest.mark.parametrize("transform", ["Bradford", "CAT02"])
def test_adaptation_maps_white_and_roundtrips(transform):
    np.testing.assert_allclose(adapt_xyz(white_xyz(D65), D65, D50, transform=transform), white_xyz(D50), atol=1e-15, rtol=0)
    values = np.array([[0., 0., 0.], [.2, .3, .7], [.4, .5, .6]])
    adapted = adapt_xyz(values, D65, D50, transform=transform)
    np.testing.assert_allclose(adapt_xyz(adapted, D50, D65, transform=transform), values, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(adapt_xyz(values, D65, D65, transform=transform), values)


def test_bradford_matches_published_matrix_with_white_rounding():
    published = [[1.0479297925, .0229468706, -.0501922663], [.0296278088, .9904344268, -.0170737991],
                 [-.0092430406, .0150551915, .7518742814]]
    np.testing.assert_allclose(adaptation_matrix(D65, D50, transform="Bradford"), published, atol=2e-4, rtol=0)
    sample = np.array([.2, .3, .7])
    assert np.max(np.abs(adapt_xyz(sample, D65, D50, transform="Bradford") - adapt_xyz(sample, D65, D50, transform="CAT02"))) > .001


@pytest.mark.parametrize("white", ["D50", "D65"])
def test_lab_white_black_and_scaled_xyz_roundtrip(white):
    np.testing.assert_allclose(xyz_to_lab(white_xyz(white), white=white), [100., 0., 0.], atol=1e-13, rtol=0)
    np.testing.assert_array_equal(xyz_to_lab([0., 0., 0.], white=white), [0., 0., 0.])
    labs = np.array([[1e-10, 0, 0], [8., -3., 4.], [60., 5., -5.], [100., 0., 0.]])
    for scale in (1, 100):
        np.testing.assert_allclose(xyz_to_lab(lab_to_xyz(labs, white=white, scale=scale), white=white, scale=scale), labs, atol=1e-12, rtol=0)


def test_lab_reference_truth_is_exact_and_display_is_derived():
    truth = np.array([[60., 180., 150.], [50., -100., -80.]])
    snapshot = truth.copy()
    canonical = reference_values_to_lab_d50(truth, color_space="lab", white="D50", scale=1)
    np.testing.assert_array_equal(canonical, truth)
    canonical[0, 0] = 10
    np.testing.assert_array_equal(truth, snapshot)
    unbounded = reference_to_working_rgb(truth)
    assert np.any(unbounded < 0) or np.any(unbounded > 1)
    display = display_reference_rgb(truth)
    assert display.min() >= 0 and display.max() <= 1
    np.testing.assert_array_equal(truth, snapshot)


def test_explicit_reference_scales_and_encodings_agree():
    pixels = np.array([[.2, .3, .7], [.8, .4, .2]])
    expected = reference_values_to_lab_d50(pixels, color_space="srgb", white="D65", scale=1)
    for scale in (255, 65535):
        np.testing.assert_allclose(reference_values_to_lab_d50(pixels * scale, color_space="srgb", white=D65, scale=scale), expected, atol=1e-13, rtol=0)
    np.testing.assert_allclose(reference_values_to_lab_d50(srgb_to_linear(pixels), color_space="linear-srgb", white=D65, scale=1), expected, atol=1e-13, rtol=0)
    xyz = rgb_to_xyz(pixels)
    for scale in (1, 100):
        np.testing.assert_allclose(reference_values_to_lab_d50(xyz * scale, color_space="xyz", white=D65, scale=scale), expected, atol=1e-13, rtol=0)


def test_common_white_scoring_does_not_confuse_linear_and_encoded_pixels():
    truth = np.array([[60., 5., -5.]])
    pixels = reference_to_working_rgb(truth)
    linear = srgb_to_linear(pixels)
    for white in ("D50", "D65"):
        assert float(delta_e00_reference(pixels, truth, evaluation_white=white)[0]) < 1e-12
        assert float(delta_e00_reference(linear, truth, input_encoding="linear-srgb", evaluation_white=white)[0]) < 1e-12
    assert float(delta_e00_reference(linear, truth)[0]) > 10


def test_no_double_adaptation_for_d50_reference():
    truth = np.array([[60., 5., -5.]])
    pixels = reference_to_working_rgb(truth)
    correct = delta_e00_reference(pixels, truth)
    wrong_truth = xyz_to_lab(adapt_xyz(lab_to_xyz(truth, white=D50), D65, D50), white=D50)
    assert correct[0] < 1e-12
    assert delta_e00_reference(pixels, wrong_truth)[0] > .1


def test_canonical_working_lightness_bypasses_negative_reference_rgb_decoder():
    from ColorCorrectionPipeline.core.color_spaces import convert_to_lab
    truth = np.array([[65., 120., -110.]])
    rgb = reference_to_working_rgb(truth)
    assert rgb[0, 1] < 0 and rgb[0, 2] > 1
    canonical_working = reference_to_working_lab(truth)
    np.testing.assert_allclose(canonical_working[0, 0], 66.13847178729989, atol=1e-12, rtol=0)
    redecoded = convert_to_lab(rgb, np.array(D65), c_space="srgb")
    assert abs(redecoded[0, 0] - canonical_working[0, 0]) > 2.0
    semantics = ColorSemantics(version="explicit-v1")
    np.testing.assert_array_equal(semantics.reference_to_working_lab(truth), canonical_working)
    np.testing.assert_array_equal(truth, [[65., 120., -110.]])


def test_all_34_independent_sharma_vectors_both_directions():
    fixtures = np.loadtxt(Path(__file__).parents[1] / "fixtures/sharma_ciede2000.txt")
    assert fixtures.shape == (34, 7)
    np.testing.assert_allclose(delta_e00(fixtures[:, :3], fixtures[:, 3:6]), fixtures[:, 6], atol=.00005, rtol=0)
    np.testing.assert_allclose(delta_e00(fixtures[:, 3:6], fixtures[:, :3]), fixtures[:, 6], atol=.00005, rtol=0)
    np.testing.assert_array_equal(delta_e00(fixtures[:, :3], fixtures[:, :3]), np.zeros(34))


@pytest.mark.parametrize("values", [np.array([np.nan, 0., 0.]), np.array([np.inf, 0., 0.]), np.zeros((2, 4)), np.empty((0, 3))])
def test_bad_tristimulus_arrays_fail(values):
    with pytest.raises(ValueError):
        rgb_to_xyz(values)


@pytest.mark.parametrize("white", [[0., .3], [.3, 0.], [.6, .5], [np.nan, .3], [1, 1, 1], "D60"])
def test_bad_whites_fail(white):
    with pytest.raises((TypeError, ValueError)):
        white_xy(white)


@pytest.mark.parametrize("scale", [0, 3, True, "1", float("nan")])
def test_ambiguous_or_unsupported_scales_fail(scale):
    with pytest.raises(ValueError):
        reference_values_to_lab_d50([[.2, .3, .4]], color_space="xyz", white=D65, scale=scale)


def test_invalid_units_spaces_and_mismatched_ids_fail():
    with pytest.raises(ValueError, match="lightness"):
        reference_values_to_lab_d50([[200., 0., 0.]], color_space="lab", white=D50, scale=1)
    with pytest.raises(ValueError, match="D65"):
        reference_values_to_lab_d50([[.2, .3, .4]], color_space="srgb", white=D50, scale=1)
    with pytest.raises(ValueError, match="encoding"):
        rgb_to_xyz([[.2, .3, .4]], encoding="display-p3")
    with pytest.raises(ValueError, match="same-shape"):
        delta_e00(np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="transform"):
        adapt_xyz([[.2, .3, .4]], D50, D50, transform="typo")
    with pytest.raises(TypeError, match="real numbers"):
        srgb_to_linear([".2"])


def test_semantics_versions_roundtrip_and_validate_real_capabilities():
    legacy = ColorSemantics()
    assert legacy.version == "legacy-v1" and legacy.adaptation == "CAT02"
    explicit = ColorSemantics(version="explicit-v1", input_encoding="linear-srgb", evaluation_white="D50")
    assert explicit.adaptation == "Bradford"
    assert ColorSemantics.from_value(json.loads(json.dumps(explicit.to_dict()))) == explicit
    snapshot = explicit.to_dict()
    snapshot["working_white"][0] = 0
    assert explicit.working_white == D65
    with pytest.raises(ValueError, match="legacy-v1 retains"):
        ColorSemantics(input_encoding="linear-srgb")
    with pytest.raises(ValueError, match="encoded-rgb"):
        ColorSemantics(version="explicit-v1", fitting_domain="linear-rgb")
    with pytest.raises(ValueError, match="working D65"):
        ColorSemantics(version="explicit-v1", working_white="D50")


def test_core_legacy_adaptation_defaults_remain_unchanged():
    import colour
    from ColorCorrectionPipeline.core.color_spaces import srgb_to_cielab_D50, adapt_chart
    samples = np.array([[.2, .3, .7], [.8, .4, .2]])
    default = srgb_to_cielab_D50(samples, np.array(D65))
    selected = srgb_to_cielab_D50(samples, np.array(D65), adaptation_transform="CAT02")
    np.testing.assert_array_equal(default, selected)
    assert np.max(np.abs(default - srgb_to_cielab_D50(samples, np.array(D65), adaptation_transform="Bradford"))) > .01
    chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    default_chart = adapt_chart(chart, np.array(D65))
    explicit_chart = adapt_chart(chart, np.array(D65), adaptation_transform="CAT02")
    np.testing.assert_array_equal(list(default_chart.data.values()), list(explicit_chart.data.values()))
