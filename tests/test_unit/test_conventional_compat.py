"""Old Colour applies the saved basis/matrix directly, without refitting."""
import colour
import numpy as np
import pytest

from ColorCorrectionPipeline import MyModels
from ColorCorrectionPipeline.bundle import dumps_models, loads_models
from ColorCorrectionPipeline.conventional import apply_conventional, conventional_basis


@pytest.mark.parametrize("options", [
    {"method": "Cheung 2004", "terms": 3}, {"method": "Cheung 2004", "terms": 5},
    {"method": "Cheung 2004", "terms": 7},
    {"method": "Finlayson 2015", "degree": 1},
    {"method": "Finlayson 2015", "degree": 2},
    {"method": "Finlayson 2015", "degree": 2, "root_polynomial_expansion": False},
    {"method": "Finlayson 2015", "degree": 3, "root_polynomial_expansion": True},
    {"method": "Finlayson 2015", "degree": 3, "root_polynomial_expansion": None},
    {"method": "Vandermonde", "degree": 1}, {"method": "Vandermonde", "degree": 2},
    {"method": "Vandermonde", "degree": 3},
])
def test_old_api_fallback_is_bitwise_equal_to_current_application(monkeypatch, options):
    rng = np.random.default_rng(124)
    image = rng.uniform(.05, .95, (2, 5, 3))
    basis = conventional_basis(image.reshape(-1, 3), **options)
    matrix = rng.normal(0, .2, (3, basis.shape[1]))
    native = getattr(colour.characterisation, "apply_matrix_colour_correction", None)
    if callable(native):
        reference = lambda value: native(value, matrix, **options)
    else:
        observed = rng.uniform(.05, .95, (48, 3))
        targets = conventional_basis(observed, **options) @ matrix.T
        matrix = colour.characterisation.matrix_colour_correction(observed, targets, **options)
        def reference(value):
            array = np.asarray(value)
            if array.size == 3:
                # Old Colour's singleton expansion is itself broken. Duplicate
                # only the oracle input, then allow GEMM/GEMV rounding below.
                pair = np.repeat(array.reshape(1, 3), 2, axis=0)
                return colour.characterisation.colour_correction(pair, observed, targets, **options)[0].reshape(array.shape)
            return colour.characterisation.colour_correction(value, observed, targets, **options)
    expected, expected_single = reference(image), reference(image[0, 0])
    monkeypatch.setattr(colour.characterisation, "apply_matrix_colour_correction", None, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("a saved correction matrix must never be refitted")
    monkeypatch.setattr(colour.characterisation, "matrix_colour_correction", forbidden)
    np.testing.assert_array_equal(apply_conventional(image, matrix, **options), expected)
    if callable(native):
        np.testing.assert_array_equal(apply_conventional(image[0, 0], matrix, **options), expected_single)
    else:
        np.testing.assert_allclose(apply_conventional(image[0, 0], matrix, **options), expected_single, atol=2e-14, rtol=0)


def test_current_api_passthrough_preserves_absent_method_and_real_errors(monkeypatch):
    def current(RGB, CCM, **options):
        assert options == {"degree": 2}
        raise TypeError("real library failure")
    monkeypatch.setattr(colour.characterisation, "apply_matrix_colour_correction", current, raising=False)
    with pytest.raises(TypeError, match="real library failure"):
        apply_conventional(np.ones((2, 3)), np.eye(3), degree=2)


def test_old_default_is_cheung_and_unknown_methods_fail(monkeypatch):
    monkeypatch.setattr(colour.characterisation, "apply_matrix_colour_correction", None, raising=False)
    rgb = np.array([[.2, .4, .6]])
    np.testing.assert_array_equal(apply_conventional(rgb, np.eye(3), degree=3), rgb)
    with pytest.raises(ValueError, match="unsupported"):
        apply_conventional(rgb, np.eye(3), method="unknown")


def test_bundle_honors_missing_root_flag_default_and_old_cheung_name(monkeypatch):
    for options in ({"method": "Finlayson 2015", "degree": 2}, {"method": "Cheung 2004", "terms": 5}):
        rgb = np.array([[.17, .31, .53]])
        basis = conventional_basis(rgb, **options)
        models = MyModels()
        models.model_cc = np.ones((3, basis.shape[1]))*.1, options, "conv"
        expected = apply_conventional(rgb, models.model_cc[0], **options)
        restored = loads_models(dumps_models(models))
        np.testing.assert_array_equal(apply_conventional(rgb, restored.model_cc[0], **restored.model_cc[1]), expected)
    # Emulate the old public name for Cheung expansion on a current runtime.
    expand = getattr(colour.characterisation, "polynomial_expansion_Cheung2004", None) or colour.characterisation.matrix_augmented_Cheung2004
    monkeypatch.setattr(colour.characterisation, "polynomial_expansion_Cheung2004", None, raising=False)
    monkeypatch.setattr(colour.characterisation, "matrix_augmented_Cheung2004", expand, raising=False)
    np.testing.assert_array_equal(conventional_basis(rgb, method="Cheung 2004", terms=5), expand(rgb, terms=5))


def test_legacy_singleton_polynomial_expander_is_padded_without_refitting(monkeypatch):
    native = colour.characterisation.polynomial_expansion_Finlayson2015
    def old_expander(values, **options):
        if np.shape(values) == (1, 3):
            raise ValueError("historical singleton tstack failure")
        return native(values, **options)
    options = {"method": "Finlayson 2015", "degree": 2}
    pixel = np.array([.2, .4, .6])
    matrix = np.eye(3, 6)
    expected = apply_conventional(pixel, matrix, **options)
    monkeypatch.setattr(colour.characterisation, "apply_matrix_colour_correction", None, raising=False)
    monkeypatch.setattr(colour.characterisation, "polynomial_expansion_Finlayson2015", old_expander)
    for shape in ((3,), (1, 3), (1, 1, 3)):
        np.testing.assert_array_equal(apply_conventional(pixel.reshape(shape), matrix, **options), expected.reshape(shape))
    assert conventional_basis(pixel.reshape(1, 3), **options).shape == (1, 6)
