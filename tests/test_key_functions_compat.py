"""key_functions re-exports only the helpers that were identical to core.

The ``_legacy_*`` functions below are verbatim copies of the key_functions
implementations before they were replaced by imports from ``core.utils``; the tests
check the re-exported versions against them on representative inputs.
"""

import types
import warnings

import numpy as np
import pytest

from ColorCorrectionPipeline import key_functions as kf
from ColorCorrectionPipeline.core import utils as cu

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dependency
    cv2 = None


def _legacy_create_detector_params():
    for factory in [
        lambda: cv2.mcc.DetectorParameters.create(),
        lambda: cv2.mcc.DetectorParameters_create(),
        lambda: cv2.mcc.DetectorParameters(),
        lambda: cv2.mcc.DetectorParametersMCC(),  # OpenCV >= 4.11 / 5.x
    ]:
        try:
            return factory()
        except Exception:
            continue
    return None


def _legacy_set_param_safe(params, name, value):
    try:
        setattr(params, name, value)
    except Exception:
        pass


def _legacy_poly_func(x, coeffs):  # horner's method
    result = 0
    for c in coeffs:
        result = result * x + c
    return result


def _legacy_compute_diag(mat_ref, mat_det, rf=0.95):

    mat_ref = np.asarray(mat_ref, dtype=np.float64)
    mat_det = np.asarray(mat_det, dtype=np.float64)
    ratios = np.divide(
        mat_ref,
        mat_det,
        out=np.full_like(mat_ref, np.nan, dtype=np.float64),
        where=np.abs(mat_det) > np.finfo(np.float64).eps,
    )
    ratios[~np.isfinite(ratios)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        factors = np.nanmedian(ratios, axis=0)
    factors = np.where(np.isfinite(factors), factors, 1.0)
    diag = np.diag(rf * factors)

    return diag


@pytest.mark.parametrize(
    "name", ["_create_detector_params", "_set_param_safe", "poly_func", "compute_diag"]
)
def test_identical_helpers_are_reexported_from_core(name):
    assert getattr(kf, name) is getattr(cu, name)


@pytest.mark.parametrize(
    "x, coeffs",
    [
        (0.5, [1.0, -2.0, 3.0]),
        (np.linspace(0, 1, 11), np.array([0.2, -0.1, 1.3, 0.0])),
        (np.arange(6, dtype=np.float32).reshape(2, 3), [2.0]),
        (np.array([1, 2, 3]), [1, 0, -1]),
        (np.array([0.1, 0.9]), []),
    ],
)
def test_poly_func_matches_legacy(x, coeffs):
    new, old = kf.poly_func(x, coeffs), _legacy_poly_func(x, coeffs)
    np.testing.assert_array_equal(new, old)
    assert np.asarray(new).dtype == np.asarray(old).dtype


def test_compute_diag_matches_legacy():
    rng = np.random.default_rng(7)
    ref = rng.random((24, 3))
    det = rng.random((24, 3))
    det_with_gaps = det.copy()
    det_with_gaps[:5, 1] = 0.0
    det_with_gaps[3, 2] = np.nan
    cases = [
        (ref, det, 0.95),
        (ref, det, 1.0),
        (ref, det_with_gaps, 0.95),
        (ref, np.zeros_like(det), 0.95),
        (ref.tolist(), det.tolist(), 0.8),
    ]
    for mat_ref, mat_det, rf in cases:
        np.testing.assert_array_equal(
            kf.compute_diag(mat_ref, mat_det, rf), _legacy_compute_diag(mat_ref, mat_det, rf)
        )


def test_set_param_safe_matches_legacy():
    class ReadOnly:
        __slots__ = ()

    for target_factory in (types.SimpleNamespace, ReadOnly):
        a, b = target_factory(), target_factory()
        kf._set_param_safe(a, "minGroupSize", 7)
        _legacy_set_param_safe(b, "minGroupSize", 7)
        assert getattr(a, "minGroupSize", None) == getattr(b, "minGroupSize", None)


def test_create_detector_params_matches_legacy():
    new, old = kf._create_detector_params(), _legacy_create_detector_params()
    assert type(new) is type(old)
    if old is None:
        return
    names = [n for n in dir(old) if not n.startswith("_")]
    for name in names:
        value = getattr(old, name)
        if callable(value):
            continue
        assert getattr(new, name) == value, name


def test_module_level_detector_params_still_configured():
    if kf.CDP is None:
        pytest.skip("OpenCV build without mcc DetectorParameters")
    assert kf.CDP.minGroupSize == 7
