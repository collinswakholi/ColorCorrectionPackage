"""Safe Lab-lightness FFC rejects unusable captures rather than hiding them."""

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline.flat_field.correction import FlatFieldCorrection


def field(value=128):
    return np.full((48, 64, 3), value, dtype=np.uint8)


def safe(img=None, **kwargs):
    return FlatFieldCorrection(field() if img is None else img, safe_mode=True,
                               crop_rect=kwargs.pop("crop_rect", (0, 0, 64, 48)), bins=kwargs.pop("bins", 8), **kwargs)


@pytest.mark.parametrize("value", [0, 1, 2])
def test_dark_white_field_fails_before_division(value):
    with pytest.raises(ValueError, match="insufficient signal"):
        safe(field(value)).compute_multiplier(degree=2, fit_method="linear")


@pytest.mark.parametrize("rect", [(0,0,0,4), (-1,0,30,30), (0,0,65,48), (40,0,30,48), (0.,0,64,48), (True,0,64,48)])
def test_invalid_crop_is_not_silently_clamped(rect):
    with pytest.raises(ValueError, match="crop_rect"):
        safe(crop_rect=rect)


def test_white_plane_saturation_is_measured_on_raw_pixels():
    img = field()
    img[:2] = [255, 128, 128]
    with pytest.raises(ValueError, match="saturated"):
        safe(img).compute_multiplier(degree=2, fit_method="linear")


def test_nearblack_pixel_is_not_hidden_by_smoothing():
    img = field()
    img[20, 20] = 0
    with pytest.raises(ValueError, match="insufficient signal"):
        safe(img).compute_multiplier(degree=2, fit_method="linear")


@pytest.mark.parametrize("kwargs,match", [({"crop_rect":(0,0,8,8)},"coverage"), ({"bins":65},"bins"), ({"smooth_window":4},"odd"), ({"min_signal":0},"signal"), ({"min_gain":2,"max_gain":1},"gain"), ({"missing_plane":"silent"},"missing_plane"), ({"manual_crop":True,"crop_rect":None},"explicit crop_rect")])
def test_invalid_safe_options_are_actionable(kwargs, match):
    with pytest.raises(ValueError, match=match):
        safe(**kwargs).compute_multiplier(degree=2, fit_method="linear")


def test_missing_plane_requires_explicit_fallback(monkeypatch):
    monkeypatch.setattr(FlatFieldCorrection, "_load_detection_model", lambda self: None)
    monkeypatch.setattr(FlatFieldCorrection, "_detect_plane_opencv_dnn", lambda self, img: (np.empty((0,4)),np.empty(0)))
    with pytest.raises(ValueError, match="no white plane"):
        safe(crop_rect=None).compute_multiplier(degree=2, fit_method="linear")
    ffc = safe(crop_rect=None, missing_plane="full-image")
    gains = ffc.compute_multiplier(degree=2, fit_method="linear")
    np.testing.assert_allclose(gains, 1)
    assert ffc.quality_report["plane_selection"] == "explicit-full-image-fallback"


def test_valid_uniform_field_yields_finite_unit_gain_and_quality_metadata():
    ffc = safe()
    gains = ffc.compute_multiplier(degree=2, fit_method="linear")
    np.testing.assert_array_equal(gains, np.ones((48,64)))
    assert ffc.quality_report["status"] == "validated"
    assert ffc.quality_report["crop_fraction"] == 1
    assert ffc.quality_report["algorithm"] == "legacy-uint8-Lab-lightness"
    assert ffc.quality_report["design_rank"] == 5


@pytest.mark.parametrize("params,match", [({"degree":0},"degree"), ({"degree":9},"degree"), ({"degree":8},"degree\\+1"), ({"fit_method":"typo"},"unknown"), ({"max_iter":0},"max_iter"), ({"random_seed":-1},"random_seed"), ({"tol":float("nan")},"tol")])
def test_invalid_fitting_controls_reject_early(params, match):
    with pytest.raises(ValueError, match=match):
        safe().compute_multiplier(**dict({"degree":2,"fit_method":"linear"}, **params))


def test_nonpositive_fitted_surface_is_not_transformed_into_valid_gain(monkeypatch):
    class NegativeModel:
        def predict(self, X):
            return np.full(len(X), -100.)
    img = field()
    img[:, :32] = 100
    monkeypatch.setattr(FlatFieldCorrection, "fit_model", lambda *a, **k: NegativeModel())
    with pytest.raises(ValueError, match="fitted-grid"):
        safe(img, max_gain=4).compute_multiplier(degree=2, fit_method="linear")


def test_interpolation_overshoot_is_validated_after_resize(monkeypatch):
    original = cv2.resize
    def overshooting(values, dimensions, **kwargs):
        result = original(values, dimensions, **kwargs)
        result[0,0] = 9
        return result
    monkeypatch.setattr(cv2, "resize", overshooting)
    with pytest.raises(ValueError, match="interpolated-full-image"):
        safe().compute_multiplier(degree=2, fit_method="linear")


def test_safe_application_rejects_spatial_shape_mismatch():
    ffc = safe()
    ffc.compute_multiplier(degree=2, fit_method="linear")
    with pytest.raises(ValueError, match="dimensions"):
        ffc.apply_ffc(field()[:20])


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1, 8])
def test_safe_application_rejects_corrupt_or_excessive_external_gain(value):
    ffc = safe()
    with pytest.raises(ValueError, match="application"):
        ffc.apply_ffc(field(), np.full((48,64), value))


def test_safe_application_requires_a_calibrated_multiplier():
    with pytest.raises(ValueError, match="calibrated"):
        safe().apply_ffc(field())


def test_observed_extreme_gain_fails_before_surface_fit():
    img = field(180)
    img[:, :30] = 50
    with pytest.raises(ValueError, match="observed.*gain limits"):
        safe(img).compute_multiplier(degree=2, fit_method="linear")


def test_valid_gradient_retains_legacy_lab_algorithm_pixels():
    row = np.linspace(115, 165, 64).round().astype(np.uint8)
    img = np.broadcast_to(row[None,:,None], (48,64,3)).copy()
    legacy = FlatFieldCorrection(img, crop_rect=(0,0,64,48), bins=8)
    current = safe(img)
    old_gain = legacy.compute_multiplier(degree=2, fit_method="linear")
    new_gain = current.compute_multiplier(degree=2, fit_method="linear")
    np.testing.assert_array_equal(new_gain, old_gain)
    np.testing.assert_array_equal(current.apply_ffc(img), legacy.apply_ffc(img))


def test_legacy_crop_clamping_and_uniform_field_numerics_remain():
    ffc = FlatFieldCorrection(field(), crop_rect=(-20,-20,200,200), bins=8)
    gains = ffc.compute_multiplier(degree=2, fit_method="linear")
    assert ffc.crop_rect == [0,0,64,48]
    np.testing.assert_array_equal(gains, np.ones((48,64)))
