"""Load-only application of legacy polynomial gamma (GC) models.

2.x ``legacy-v1`` fits stored GC as polynomial coefficients over Lab L*. 3.0 no
longer fits them; this module only applies saved coefficients, unchanged.
"""

import numpy as np

from ..constants import WP_DEFAULT
from ..core.accel import (
    HAS_NUMBA,
    lab_to_srgb_fast_image,
    lab_to_srgb_numba,
    srgb_to_lab_fast_image,
    srgb_to_lab_numba,
)
from ..core.utils import poly_func

__all__ = ["gc_lightness", "predict_image"]

# L* span of the ColorChecker neutral row (black ~20.5, white ~96.5).
_GC_NEUTRAL_RANGE = (20.5, 96.5)


def gc_lightness(L: np.ndarray, coeffs: np.ndarray, samples: int = 1001) -> np.ndarray:
    """Map Lab L* through a legacy gamma polynomial, pinned and monotonic.

    The least-squares polynomial does not pass exactly through its (0, 0) and
    (100, 100) anchors and may turn back near the ends, so black was lifted and
    highlights could invert. The endpoint error is removed with ramps confined
    to inputs darker than where the curve reaches the reference black neutral
    (L* 20.5) or brighter than where it reaches the white neutral (L* 96.5),
    so the fit through the chart's neutrals is unchanged; a running maximum over a dense
    [0, 100] lookup then makes the curve non-decreasing. Values outside
    [0, 100] use the pinned polynomial directly.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    p0 = float(poly_func(0.0, coeffs))
    p100 = float(poly_func(100.0, coeffs))

    grid = np.linspace(0.0, 100.0, int(samples))
    raw = poly_func(grid, coeffs)
    # Measured L* at which the curve reaches the reference black / white neutral.
    ref_low, ref_high = _GC_NEUTRAL_RANGE
    hit_low = np.nonzero(raw >= ref_low)[0]
    hit_high = np.nonzero(raw >= ref_high)[0]
    low = float(grid[hit_low[0]]) if hit_low.size and grid[hit_low[0]] > 0 else ref_low
    high = float(grid[hit_high[0]]) if hit_high.size and grid[hit_high[0]] < 100 else ref_high
    high = max(high, low)

    def pinned(x):
        x = np.asarray(x, dtype=np.float64)
        w_low = np.clip(1.0 - x / low, 0.0, 1.0)
        w_high = np.clip((x - high) / max(100.0 - high, 1e-9), 0.0, 1.0)
        return poly_func(x, coeffs) - p0 * w_low - (p100 - 100.0) * w_high

    curve = np.maximum.accumulate(pinned(grid))
    out: np.ndarray = np.interp(L, grid, curve)
    outside = (L < 0.0) | (L > 100.0)
    if np.any(outside):
        out[outside] = pinned(L[outside])
    return out


def predict_image(
    img: np.ndarray,
    coeffs: np.ndarray,
    ref_illuminant: np.ndarray = WP_DEFAULT,
    explicit_color: bool = False,
) -> np.ndarray:
    """
    Apply gamma correction to image using polynomial coefficients.

    Optimised path: uses fast sRGB↔Lab (Numba or vectorised NumPy)
    instead of colour-science, saving 50-80% time on the conversion.

    Args:
        img: RGB image (H, W, 3), float64, 0-1 range
        coeffs: Polynomial coefficients from estimate_gamma_profile
        ref_illuminant: Reference illuminant (xy coordinates)

    Returns:
        np.ndarray: Gamma-corrected image (same shape as input)
    """
    H, W, C = img.shape
    img_flat = img.reshape(-1, C)

    if explicit_color:
        from ..colorimetry import D65, lab_to_xyz, rgb_to_xyz, white_xy, xyz_to_lab, xyz_to_rgb

        if white_xy(ref_illuminant) != D65:
            raise ValueError("explicit gamma application requires working D65")
        lab = xyz_to_lab(rgb_to_xyz(img_flat, encoding="srgb"), white=D65)
        lab[:, 0] = gc_lightness(lab[:, 0], coeffs)
        rgb: np.ndarray = xyz_to_rgb(lab_to_xyz(lab, white=D65), encoding="srgb", policy="preserve")
        return rgb.reshape(H, W, C)

    # ── sRGB → Lab (fast) ──
    if HAS_NUMBA:
        img_flat_lab = srgb_to_lab_numba(img_flat)
    else:
        img_flat_lab = srgb_to_lab_fast_image(img_flat)
    img_flat_L = img_flat_lab[:, 0]

    # Apply the pinned, monotonic lightness curve. A 1-D lookup on the CPU is
    # cheaper than the former per-pixel GPU polynomial and identical on every device.
    result = gc_lightness(img_flat_L, coeffs)

    # Replace L channel in-place (no copy needed)
    img_flat_lab[:, 0] = result

    # ── Lab → sRGB (fast) ──
    if HAS_NUMBA:
        result_srgb = lab_to_srgb_numba(img_flat_lab)
    else:
        result_srgb = lab_to_srgb_fast_image(img_flat_lab)

    return result_srgb.reshape(H, W, C)
