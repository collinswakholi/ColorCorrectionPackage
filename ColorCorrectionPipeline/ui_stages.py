"""UI-compatible FFC fitting with the package's existing quality gates.

Numerical equations ported from the MIT-licensed ccp_core/ffc.rs by
Collins Wakholi (2026). Legacy FFC never selects these equations implicitly.
"""

from copy import deepcopy
import numpy as np

from .colorimetry import rgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_rgb, D65
from .ui_fitting import round_image, _integer, _solve


def apply_ffc(rgb, gain):
    """Apply an encoded-RGB lightness gain with a native float32 boundary."""
    image = round_image(rgb)
    multiplier = np.asarray(gain)
    if multiplier.shape != image.shape[:2]:
        raise ValueError("UI FFC requires the calibrated image dimensions")
    lab = xyz_to_lab(rgb_to_xyz(image, encoding="srgb"), white=D65)
    lab[..., 0] = np.clip(lab[..., 0] * multiplier, 0, 100)
    return round_image(xyz_to_rgb(lab_to_xyz(lab, white=D65), encoding="srgb", policy="clip"))


def _cubic_weights(t):
    t2, t3 = t * t, t * t * t
    return (
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1,
        -1.5 * t3 + 2 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    )


def _resize(coarse, height, width):
    """Native endpoint-aligned Catmull-Rom interpolation, in float32."""
    n = len(coarse)
    x = np.arange(width, dtype=np.float32) * np.float32(n - 1) / np.float32(max(width - 1, 1))
    y = np.arange(height, dtype=np.float32) * np.float32(n - 1) / np.float32(max(height - 1, 1))
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = _cubic_weights(x - ix.astype(np.float32)), _cubic_weights(y - iy.astype(np.float32))
    result = np.zeros((height, width), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            values = coarse[
                np.clip(iy[:, None] - 1 + row, 0, n - 1), np.clip(ix[None, :] - 1 + col, 0, n - 1)
            ]
            result += values * wy[row][:, None] * wx[col][None, :]
    return result


def fit_ffc(owner, image, parameters):
    """Fit the UI plane/polynomial surface; reject unqualified FFC solvers."""
    from .flat_field.correction import FlatFieldCorrection

    p = deepcopy(parameters)
    method = p.get("fit_method", "poly")
    if method not in ("linear", "poly"):
        raise ValueError("UI-compatible FFC currently supports linear and poly surfaces")
    allowed = {
        "model_path",
        "manual_crop",
        "show",
        "bins",
        "smooth_window",
        "crop_rect",
        "degree",
        "fit_method",
        "cache_multiplier",
        "get_deltaE",
        "safe_mode",
        "missing_plane",
        "min_signal",
        "max_saturated_fraction",
        "min_crop_fraction",
        "min_gain",
        "max_gain",
    }
    if set(p) - allowed:
        raise ValueError("unsupported UI FFC settings: %s" % sorted(set(p) - allowed))
    bins = _integer(p.get("bins", 50), "FFC bins", 2, 512)
    degree = 1 if method == "linear" else _integer(p.get("degree", 5), "FFC degree", 1, 8)
    if bins < degree + 1:
        raise ValueError("FFC needs at least degree+1 coordinate bins")
    init = {
        key: p[key]
        for key in allowed
        if key in p and key not in ("degree", "fit_method", "cache_multiplier", "get_deltaE")
    }
    init.update(bins=bins, safe_mode=True)
    guard = FlatFieldCorrection(owner.White_Image, **init)
    guard.detect_and_crop()
    guard._validate_safe_crop()
    quality = deepcopy(guard.quality_report)
    x1, y1, x2, y2 = guard.crop_rect
    source = round_image(owner.white_image_data.to_srgb())
    h, w = source.shape[:2]
    if source.shape != image.shape:
        raise ValueError("FFC white and source images must have identical dimensions")
    lab = xyz_to_lab(rgb_to_xyz(source[y1:y2, x1:x2], encoding="srgb"), white=D65)
    lightness = lab[..., 0].astype(np.float32)
    if lightness.min() / 100 < guard.min_signal:
        raise ValueError("safe UI FFC white plane has insufficient signal")
    epsilon = np.float32(1e-12)
    ratios = lightness.max() / np.maximum(lightness, epsilon)
    half = (guard.smooth_window - 1) // 2
    ch, cw = ratios.shape
    z = np.empty((bins, bins), dtype=np.float32)
    # Rust accumulates each small window in row-major float32 order.
    for i in range(bins):
        yc = int(np.float32(i) * np.float32(ch - 1) / np.float32(bins - 1))
        for j in range(bins):
            xc = int(np.float32(j) * np.float32(cw - 1) / np.float32(bins - 1))
            window = ratios[
                max(0, yc - half) : min(ch, yc + half + 1),
                max(0, xc - half) : min(cw, xc + half + 1),
            ].ravel()
            z[i, j] = np.cumsum(window, dtype=np.float32)[-1] / np.float32(len(window))
    low = z.min()
    span = np.float32(z.max() - low + epsilon)
    zs = ((z - low) / span).astype(np.float64).ravel()

    def design(x, y):
        return np.stack(
            [x**i * y ** (total - i) for total in range(degree + 1) for i in range(total, -1, -1)],
            axis=-1,
        )

    x = (
        np.float32(x1)
        + np.arange(bins, dtype=np.float32) * np.float32(x2 - x1) / np.float32(bins - 1)
    ).astype(np.float64) / (w + float(epsilon))
    y = (
        np.float32(y1)
        + np.arange(bins, dtype=np.float32) * np.float32(y2 - y1) / np.float32(bins - 1)
    ).astype(np.float64) / (h + float(epsilon))
    xx, yy = np.meshgrid(x, y)
    matrix = design(xx.ravel(), yy.ravel())
    ridge = 0 if method == "linear" else 1e-8
    coeff = _solve(matrix.T @ matrix + np.eye(matrix.shape[1]) * ridge, matrix.T @ zs)
    x = (np.arange(bins, dtype=np.float32) * np.float32(w - 1) / np.float32(bins - 1)).astype(
        np.float64
    ) / (w + float(epsilon))
    y = (np.arange(bins, dtype=np.float32) * np.float32(h - 1) / np.float32(bins - 1)).astype(
        np.float64
    ) / (h + float(epsilon))
    xx, yy = np.meshgrid(x, y)
    coarse = (design(xx, yy) @ coeff).astype(np.float32) * span + low
    high = coarse > np.float32(1.3)
    coarse[high] = np.float32(1.3) + np.float32(1.8 - 1.3) * np.tanh(
        np.float32(1.8) * (coarse[high] - np.float32(1.3))
    )
    gain = _resize(coarse, h, w).astype(np.float64)
    if not np.isfinite(gain).all() or gain.min() < guard.min_gain or gain.max() > guard.max_gain:
        raise ValueError("UI FFC gain violates the configured finite positive gain limits")
    gain = np.frombuffer(gain.tobytes(), dtype=np.float64).reshape(gain.shape)
    quality.update(
        algorithm="ui-compatible-v1/float32-Lab-D65",
        fit_method=method,
        effective_degree=degree,
        minimum_gain=float(gain.min()),
        maximum_gain=float(gain.max()),
        ridge=ridge,
        solver="float64 normal equations",
        upsampling="float32 endpoint-aligned Catmull-Rom",
    )
    owner.models.model_ffc = gain
    owner.numerical_report["FFC"] = quality
    return apply_ffc(image, gain), {}, False
