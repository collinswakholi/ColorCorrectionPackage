"""
Hardware acceleration detection and dispatch
=============================================

Detects available hardware backends (CUDA via Numba, Intel iGPU, CPU)
and provides dispatch utilities + optimized kernels.

Priority: Numba CUDA > CPU Numba JIT > NumPy fallback
"""

import functools
import numpy as np

__all__ = [
    "HAS_NUMBA",
    "HAS_NUMBA_CUDA",
    "HAS_TORCH_CUDA",
    "backend_info",
    "nb_njit",
    "nb_prange",
    "nb_cuda_jit",
    "srgb_to_linear_lut",
    "linear_to_srgb_lut",
    "apply_srgb_to_linear_image",
    "apply_linear_to_srgb_image",
    "srgb_to_xyz_matrix",
    "xyz_to_srgb_matrix",
    "xyz_to_lab_fast",
    "lab_to_xyz_fast",
    "srgb_to_lab_fast_image",
    "lab_to_srgb_fast_image",
    "trilinear_lut_apply",
    "build_3d_lut",
    "apply_ffc_float",
]

# ── Backend detection ──────────────────────────────────────────────────────

HAS_NUMBA = False
HAS_NUMBA_CUDA = False
HAS_TORCH_CUDA = False

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    try:
        from numba import cuda as numba_cuda
        HAS_NUMBA_CUDA = numba_cuda.is_available()
    except Exception:
        pass
except ImportError:
    pass

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    pass


def backend_info() -> dict:
    """Return dict describing available acceleration backends."""
    info = {
        "numba": HAS_NUMBA,
        "numba_version": getattr(numba, "__version__", None) if HAS_NUMBA else None,
        "numba_cuda": HAS_NUMBA_CUDA,
        "torch_cuda": HAS_TORCH_CUDA,
    }
    if HAS_NUMBA_CUDA:
        try:
            dev = numba_cuda.get_current_device()
            info["cuda_device"] = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
            info["cuda_compute"] = dev.compute_capability
        except Exception:
            pass
    return info


# ── JIT wrappers (graceful fallback) ──────────────────────────────────────

def _identity_decorator(*args, **kwargs):
    """No-op decorator when numba is not available."""
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda f: f

if HAS_NUMBA:
    nb_njit = njit
    nb_prange = prange
else:
    nb_njit = _identity_decorator
    nb_prange = range

if HAS_NUMBA_CUDA:
    nb_cuda_jit = numba_cuda.jit
else:
    nb_cuda_jit = _identity_decorator


# ============================================================================
# sRGB ↔ Linear LUTs (256 entries, float64)
# ============================================================================
# IEC 61966-2-1 piecewise formula

def _build_srgb_to_linear_lut() -> np.ndarray:
    """Build 256-entry LUT: sRGB [0..255] → linear [0..1]."""
    lut = np.empty(256, dtype=np.float64)
    for i in range(256):
        c = i / 255.0
        if c <= 0.04045:
            lut[i] = c / 12.92
        else:
            lut[i] = ((c + 0.055) / 1.055) ** 2.4
    return lut

def _build_linear_to_srgb_lut(size: int = 4096) -> np.ndarray:
    """Build LUT: linear [0..1] quantised to `size` steps → sRGB [0..1]."""
    lut = np.empty(size, dtype=np.float64)
    for i in range(size):
        c = i / (size - 1)
        if c <= 0.0031308:
            lut[i] = 12.92 * c
        else:
            lut[i] = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return lut

srgb_to_linear_lut = _build_srgb_to_linear_lut()        # shape (256,)
linear_to_srgb_lut = _build_linear_to_srgb_lut(4096)    # shape (4096,)

# ── sRGB ↔ XYZ matrices (D65) ─────────────────────────────────────────────
# Standard IEC 61966-2-1 / sRGB spec
srgb_to_xyz_matrix = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

xyz_to_srgb_matrix = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

# D65 reference white XYZ (Y=1)
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_D65_XYZ_inv = 1.0 / _D65_XYZ

# CIE Lab constants
_LAB_EPSILON = 216.0 / 24389.0   # 0.008856
_LAB_KAPPA = 24389.0 / 27.0      # 903.3


# ============================================================================
# Fast sRGB ↔ Lab (vectorised NumPy, no colour-science dependency)
# ============================================================================

def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Vectorised sRGB → linear (float input 0-1)."""
    out = np.empty_like(srgb)
    mask = srgb <= 0.04045
    out[mask] = srgb[mask] / 12.92
    out[~mask] = ((srgb[~mask] + 0.055) / 1.055) ** 2.4
    return out

def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Vectorised linear → sRGB (float input 0-1)."""
    out = np.empty_like(linear)
    mask = linear <= 0.0031308
    out[mask] = 12.92 * linear[mask]
    out[~mask] = 1.055 * (linear[~mask] ** (1.0 / 2.4)) - 0.055
    return out

def _f_lab(t: np.ndarray) -> np.ndarray:
    """CIE Lab forward transform helper."""
    out = np.empty_like(t)
    mask = t > _LAB_EPSILON
    out[mask] = np.cbrt(t[mask])
    out[~mask] = (_LAB_KAPPA * t[~mask] + 16.0) / 116.0
    return out

def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    """CIE Lab inverse transform helper."""
    out = np.empty_like(t)
    t3 = t ** 3
    mask = t3 > _LAB_EPSILON
    out[mask] = t3[mask]
    out[~mask] = (116.0 * t[~mask] - 16.0) / _LAB_KAPPA
    return out


def srgb_to_lab_fast_image(img: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image to CIE L*a*b* (D65) — fast, no colour-science dependency.

    Parameters
    ----------
    img : (H, W, 3) or (N, 3) float64 in [0, 1]

    Returns
    -------
    lab : same shape, float64
    """
    shape = img.shape
    flat = img.reshape(-1, 3)

    # sRGB → linear
    linear = _srgb_to_linear(flat)

    # linear → XYZ
    xyz = linear @ srgb_to_xyz_matrix.T                  # (N, 3)

    # Normalise by D65 white
    xyz_n = xyz * _D65_XYZ_inv                           # (N, 3)

    # Lab forward
    f = _f_lab(xyz_n)                                    # (N, 3)

    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0                  # L*
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])             # a*
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])             # b*

    return lab.reshape(shape)


def lab_to_srgb_fast_image(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE L*a*b* (D65) to sRGB — fast, no colour-science dependency.

    Parameters
    ----------
    lab : (H, W, 3) or (N, 3) float64

    Returns
    -------
    srgb : same shape, float64 clipped to [0, 1]
    """
    shape = lab.shape
    flat = lab.reshape(-1, 3)

    # Lab → f values
    fy = (flat[:, 0] + 16.0) / 116.0
    fx = flat[:, 1] / 500.0 + fy
    fz = fy - flat[:, 2] / 200.0
    f = np.empty((flat.shape[0], 3), dtype=flat.dtype)
    f[:, 0] = fx
    f[:, 1] = fy
    f[:, 2] = fz

    # f → XYZ (normalised)
    xyz_n = _f_lab_inv(f)                                # (N, 3)

    # Denormalise
    xyz = xyz_n * _D65_XYZ                               # (N, 3)

    # XYZ → linear sRGB
    linear = xyz @ xyz_to_srgb_matrix.T                  # (N, 3)

    # linear → sRGB
    srgb = _linear_to_srgb(np.clip(linear, 0, 1))
    srgb = np.clip(srgb, 0, 1)

    return srgb.reshape(shape)


def xyz_to_lab_fast(xyz: np.ndarray, illuminant_xy: np.ndarray = None) -> np.ndarray:
    """Fast XYZ to Lab. Default illuminant = D65."""
    if illuminant_xy is not None:
        # Convert xy to XYZ (Y=1)
        x, y = illuminant_xy[0], illuminant_xy[1]
        wp = np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)
    else:
        wp = _D65_XYZ

    shape = xyz.shape
    flat = xyz.reshape(-1, 3)
    xyz_n = flat / wp
    f = _f_lab(xyz_n)
    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])
    return lab.reshape(shape)


def lab_to_xyz_fast(lab: np.ndarray, illuminant_xy: np.ndarray = None) -> np.ndarray:
    """Fast Lab to XYZ. Default illuminant = D65."""
    if illuminant_xy is not None:
        x, y = illuminant_xy[0], illuminant_xy[1]
        wp = np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)
    else:
        wp = _D65_XYZ

    shape = lab.shape
    flat = lab.reshape(-1, 3)
    fy = (flat[:, 0] + 16.0) / 116.0
    fx = flat[:, 1] / 500.0 + fy
    fz = fy - flat[:, 2] / 200.0
    f = np.column_stack([fx, fy, fz])
    xyz_n = _f_lab_inv(f)
    xyz = xyz_n * wp
    return xyz.reshape(shape)


# ============================================================================
# Numba-accelerated kernels (CPU parallel)
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _srgb_to_lab_numba(flat_rgb, out_lab, wp_inv):
        """Parallel sRGB → Lab conversion (Numba CPU)."""
        N = flat_rgb.shape[0]
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        eps = 216.0 / 24389.0
        kappa = 24389.0 / 27.0
        for i in prange(N):
            # sRGB → linear
            r, g, b = flat_rgb[i, 0], flat_rgb[i, 1], flat_rgb[i, 2]
            lr = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
            lg = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
            lb = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
            # linear → XYZ
            x = M[0, 0] * lr + M[0, 1] * lg + M[0, 2] * lb
            y = M[1, 0] * lr + M[1, 1] * lg + M[1, 2] * lb
            z = M[2, 0] * lr + M[2, 1] * lg + M[2, 2] * lb
            # normalise
            xn = x * wp_inv[0]
            yn = y * wp_inv[1]
            zn = z * wp_inv[2]
            # f values
            fx = xn ** (1.0 / 3.0) if xn > eps else (kappa * xn + 16.0) / 116.0
            fy = yn ** (1.0 / 3.0) if yn > eps else (kappa * yn + 16.0) / 116.0
            fz = zn ** (1.0 / 3.0) if zn > eps else (kappa * zn + 16.0) / 116.0
            out_lab[i, 0] = 116.0 * fy - 16.0
            out_lab[i, 1] = 500.0 * (fx - fy)
            out_lab[i, 2] = 200.0 * (fy - fz)

    @njit(parallel=True, cache=True, fastmath=True)
    def _lab_to_srgb_numba(flat_lab, out_rgb, wp):
        """Parallel Lab → sRGB conversion (Numba CPU)."""
        N = flat_lab.shape[0]
        M = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ])
        eps = 216.0 / 24389.0
        kappa = 24389.0 / 27.0
        for i in prange(N):
            L, a, b_ = flat_lab[i, 0], flat_lab[i, 1], flat_lab[i, 2]
            fy = (L + 16.0) / 116.0
            fx = a / 500.0 + fy
            fz = fy - b_ / 200.0
            # f → XYZ_n
            fx3 = fx * fx * fx
            fy3 = fy * fy * fy
            fz3 = fz * fz * fz
            xn = fx3 if fx3 > eps else (116.0 * fx - 16.0) / kappa
            yn = fy3 if fy3 > eps else (116.0 * fy - 16.0) / kappa
            zn = fz3 if fz3 > eps else (116.0 * fz - 16.0) / kappa
            # denormalise
            x = xn * wp[0]
            y = yn * wp[1]
            z = zn * wp[2]
            # XYZ → linear
            lr = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z
            lg = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z
            lb = M[2, 0] * x + M[2, 1] * y + M[2, 2] * z
            # clip
            lr = max(0.0, min(1.0, lr))
            lg = max(0.0, min(1.0, lg))
            lb = max(0.0, min(1.0, lb))
            # linear → sRGB
            r = 12.92 * lr if lr <= 0.0031308 else 1.055 * (lr ** (1.0 / 2.4)) - 0.055
            g = 12.92 * lg if lg <= 0.0031308 else 1.055 * (lg ** (1.0 / 2.4)) - 0.055
            b = 12.92 * lb if lb <= 0.0031308 else 1.055 * (lb ** (1.0 / 2.4)) - 0.055
            out_rgb[i, 0] = max(0.0, min(1.0, r))
            out_rgb[i, 1] = max(0.0, min(1.0, g))
            out_rgb[i, 2] = max(0.0, min(1.0, b))


    def srgb_to_lab_numba(img: np.ndarray) -> np.ndarray:
        """sRGB → Lab using Numba parallel CPU. Same signature as srgb_to_lab_fast_image."""
        shape = img.shape
        flat = np.ascontiguousarray(img.reshape(-1, 3), dtype=np.float64)
        out = np.empty_like(flat)
        _srgb_to_lab_numba(flat, out, _D65_XYZ_inv)
        return out.reshape(shape)


    def lab_to_srgb_numba(lab: np.ndarray) -> np.ndarray:
        """Lab → sRGB using Numba parallel CPU. Same signature as lab_to_srgb_fast_image."""
        shape = lab.shape
        flat = np.ascontiguousarray(lab.reshape(-1, 3), dtype=np.float64)
        out = np.empty_like(flat)
        _lab_to_srgb_numba(flat, out, _D65_XYZ)
        return out.reshape(shape)

else:
    # Fallback to numpy versions
    srgb_to_lab_numba = srgb_to_lab_fast_image
    lab_to_srgb_numba = lab_to_srgb_fast_image


# ============================================================================
# 3D LUT (trilinear interpolation)
# ============================================================================

def build_3d_lut(
    model_predict_fn,
    grid_size: int = 33,
    input_range: tuple = (0.0, 1.0),
) -> np.ndarray:
    """
    Build a 3D lookup table from a prediction function.

    Parameters
    ----------
    model_predict_fn : callable
        f(rgb_array_N3) -> rgb_array_N3
    grid_size : int
        Number of grid points per axis (e.g., 33 → 33³ = 35937 entries)
    input_range : tuple
        Min/max of input RGB range

    Returns
    -------
    lut : (grid_size, grid_size, grid_size, 3) float64
    """
    lo, hi = input_range
    axis = np.linspace(lo, hi, grid_size, dtype=np.float64)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)

    # Run prediction on the grid
    predicted = model_predict_fn(grid)
    lut = predicted.reshape(grid_size, grid_size, grid_size, 3)
    return lut


if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _trilinear_interp(flat_rgb, lut, grid_size):
        """Trilinear interpolation in a 3D LUT (Numba CPU parallel)."""
        N = flat_rgb.shape[0]
        out = np.empty((N, 3), dtype=np.float64)
        step = 1.0 / (grid_size - 1)
        gs_m1 = grid_size - 1
        for i in prange(N):
            r, g, b = flat_rgb[i, 0], flat_rgb[i, 1], flat_rgb[i, 2]
            # Clamp to [0, 1]
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            # Find cell indices
            rf = r / step
            gf = g / step
            bf = b / step
            r0 = int(min(rf, gs_m1 - 1))
            g0 = int(min(gf, gs_m1 - 1))
            b0 = int(min(bf, gs_m1 - 1))
            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1
            # Fractional parts
            rd = rf - r0
            gd = gf - g0
            bd = bf - b0
            # Weights
            w1r = 1.0 - rd
            w1g = 1.0 - gd
            w1b = 1.0 - bd
            # Load all 8 corners (contiguous 3-channel reads — better cache locality)
            v000_0 = lut[r0, g0, b0, 0]; v000_1 = lut[r0, g0, b0, 1]; v000_2 = lut[r0, g0, b0, 2]
            v001_0 = lut[r0, g0, b1, 0]; v001_1 = lut[r0, g0, b1, 1]; v001_2 = lut[r0, g0, b1, 2]
            v010_0 = lut[r0, g1, b0, 0]; v010_1 = lut[r0, g1, b0, 1]; v010_2 = lut[r0, g1, b0, 2]
            v011_0 = lut[r0, g1, b1, 0]; v011_1 = lut[r0, g1, b1, 1]; v011_2 = lut[r0, g1, b1, 2]
            v100_0 = lut[r1, g0, b0, 0]; v100_1 = lut[r1, g0, b0, 1]; v100_2 = lut[r1, g0, b0, 2]
            v101_0 = lut[r1, g0, b1, 0]; v101_1 = lut[r1, g0, b1, 1]; v101_2 = lut[r1, g0, b1, 2]
            v110_0 = lut[r1, g1, b0, 0]; v110_1 = lut[r1, g1, b0, 1]; v110_2 = lut[r1, g1, b0, 2]
            v111_0 = lut[r1, g1, b1, 0]; v111_1 = lut[r1, g1, b1, 1]; v111_2 = lut[r1, g1, b1, 2]
            # Trilinear interpolation — all 3 channels unrolled
            # Channel 0
            c00 = v000_0 * w1r + v100_0 * rd
            c01 = v001_0 * w1r + v101_0 * rd
            c10 = v010_0 * w1r + v110_0 * rd
            c11 = v011_0 * w1r + v111_0 * rd
            c0 = c00 * w1g + c10 * gd
            c1 = c01 * w1g + c11 * gd
            out[i, 0] = c0 * w1b + c1 * bd
            # Channel 1
            c00 = v000_1 * w1r + v100_1 * rd
            c01 = v001_1 * w1r + v101_1 * rd
            c10 = v010_1 * w1r + v110_1 * rd
            c11 = v011_1 * w1r + v111_1 * rd
            c0 = c00 * w1g + c10 * gd
            c1 = c01 * w1g + c11 * gd
            out[i, 1] = c0 * w1b + c1 * bd
            # Channel 2
            c00 = v000_2 * w1r + v100_2 * rd
            c01 = v001_2 * w1r + v101_2 * rd
            c10 = v010_2 * w1r + v110_2 * rd
            c11 = v011_2 * w1r + v111_2 * rd
            c0 = c00 * w1g + c10 * gd
            c1 = c01 * w1g + c11 * gd
            out[i, 2] = c0 * w1b + c1 * bd
        return out
else:
    def _trilinear_interp(flat_rgb, lut, grid_size):
        """Trilinear interpolation — pure NumPy fallback."""
        from scipy.interpolate import RegularGridInterpolator
        axis = np.linspace(0, 1, grid_size)
        interps = []
        for c in range(3):
            interp = RegularGridInterpolator(
                (axis, axis, axis), lut[:, :, :, c],
                method="linear", bounds_error=False, fill_value=None,
            )
            interps.append(interp)
        out = np.column_stack([interp(flat_rgb) for interp in interps])
        return out


def trilinear_lut_apply(
    img: np.ndarray,
    lut: np.ndarray,
    grid_size: int = 33,
) -> np.ndarray:
    """
    Apply 3D LUT to image via trilinear interpolation.

    Parameters
    ----------
    img : (H, W, 3) or (N, 3) float64 in [0, 1]
    lut : (G, G, G, 3) float64
    grid_size : int

    Returns
    -------
    result : same shape as img
    """
    shape = img.shape
    flat = np.ascontiguousarray(img.reshape(-1, 3), dtype=np.float64)
    result = _trilinear_interp(flat, lut, grid_size)
    return result.reshape(shape)


# ============================================================================
# Vectorised saturation extrapolation
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def extrapolate_saturated_pixels(flat_img, scale_factor):
        """Replace saturated pixels (any channel >= 0.99) with scaled values."""
        N = flat_img.shape[0]
        out = flat_img.copy()
        for i in prange(N):
            if flat_img[i, 0] >= 0.99 or flat_img[i, 1] >= 0.99 or flat_img[i, 2] >= 0.99:
                out[i, 0] = min(1.0, flat_img[i, 0] * scale_factor)
                out[i, 1] = min(1.0, flat_img[i, 1] * scale_factor)
                out[i, 2] = min(1.0, flat_img[i, 2] * scale_factor)
        return out
else:
    def extrapolate_saturated_pixels(flat_img, scale_factor):
        """Vectorised NumPy fallback."""
        out = flat_img.copy()
        mask = np.any(flat_img >= 0.99, axis=-1)
        out[mask] = np.clip(flat_img[mask] * scale_factor, 0, 1)
        return out


# ============================================================================
# FFC binned sampling (vectorised)
# ============================================================================

def ffc_sample_bins(multiplier: np.ndarray, bins: int, smooth_window: int) -> np.ndarray:
    """
    Sample a multiplier surface at uniform grid using cv2.blur instead of
    Python double loop.

    Parameters
    ----------
    multiplier : (H, W) float array
    bins : int — grid resolution
    smooth_window : int — smoothing kernel size

    Returns
    -------
    Z_m : (bins, bins) float array — sampled multiplier
    """
    import cv2
    h_win = max(1, smooth_window)
    # Box-blur the entire multiplier once
    blurred = cv2.blur(multiplier.astype(np.float64), (h_win, h_win))
    # Subsample at uniform grid
    rows = np.linspace(0, multiplier.shape[0] - 1, bins).astype(int)
    cols = np.linspace(0, multiplier.shape[1] - 1, bins).astype(int)
    Z_m = blurred[np.ix_(rows, cols)]
    return Z_m


# ============================================================================
# FFC float-path (skip uint8 round-trip)
# ============================================================================

def apply_ffc_float(
    img_rgb: np.ndarray,
    multiplier: np.ndarray,
) -> np.ndarray:
    """
    Apply flat-field correction directly on a float64 RGB image.

    Converts sRGB → Lab, multiplies L channel by *multiplier*, then
    converts Lab → sRGB.  Avoids the uint8/BGR round-trip that the
    original ``FlatFieldCorrection.apply_ffc`` uses.

    Parameters
    ----------
    img_rgb : (H, W, 3) float64, sRGB in [0, 1]
    multiplier : (H, W) float64 — the FFC multiplier surface

    Returns
    -------
    corrected : (H, W, 3) float64, sRGB clipped to [0, 1]
    """
    import cv2

    H, W = img_rgb.shape[:2]
    mH, mW = multiplier.shape[:2]

    # Resize multiplier if sizes don't match
    if (H, W) != (mH, mW):
        multiplier = cv2.resize(
            multiplier.astype(np.float64), (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    flat = img_rgb.reshape(-1, 3)

    # sRGB → Lab
    if HAS_NUMBA:
        lab = srgb_to_lab_numba(flat)
    else:
        lab = srgb_to_lab_fast_image(flat)

    lab_2d = lab.reshape(H, W, 3)

    # Multiply L channel, clip to valid Lab L range [0, 100]
    lab_2d[:, :, 0] = np.clip(lab_2d[:, :, 0] * multiplier, 0.0, 100.0)

    # Lab → sRGB
    flat_lab = lab_2d.reshape(-1, 3)
    if HAS_NUMBA:
        result = lab_to_srgb_numba(flat_lab)
    else:
        result = lab_to_srgb_fast_image(flat_lab)

    return np.clip(result.reshape(H, W, 3), 0.0, 1.0)
