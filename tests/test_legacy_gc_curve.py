"""Legacy gamma correction must keep black/white fixed and be monotonic."""
import numpy as np

from ColorCorrectionPipeline.core.correction import predict_image


def _ramp(n=257):
    v = np.linspace(0.0, 1.0, n)
    return np.repeat(v[:, None, None], 3, axis=2).reshape(n, 1, 3)


def test_endpoints_are_fixed_even_when_fit_misses_them():
    # A typical over-fitted legacy curve: lifts black by 4 L* and overshoots white.
    coeffs = np.array([-2.0e-5, 0.003, 0.85, 4.0])
    out = predict_image(_ramp(), coeffs)[:, 0, :]
    np.testing.assert_allclose(out[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-6)


def test_output_is_monotonic_for_non_monotonic_polynomial():
    coeffs = np.array([8.0e-6, -1.2e-3, 1.05, 0.0])  # turns back near L*=100
    out = predict_image(_ramp(1025), coeffs)[:, 0, 0]
    assert np.all(np.diff(out) >= -1e-9)


def test_identity_coefficients_are_identity():
    img = np.random.default_rng(1).random((8, 8, 3))
    np.testing.assert_allclose(predict_image(img, np.array([1.0, 0.0])), img, atol=2e-4)
