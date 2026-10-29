"""Deterministic full-precision sampling independent of detection previews."""

import math
import numpy as np


def sample_quadrilaterals(rgb, quadrilaterals, count=81):
    """Return cells×samples×RGB using bilinear interpolation on declared pixels.

    The sampling grid is defined inside the supplied quadrilaterals. Clipping
    statistics derived from these samples describe sampled pixels, not every
    physical pixel in a chart cell.
    """
    image = np.asarray(rgb, dtype=np.float64)
    quads = np.asarray(quadrilaterals, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("sampling image must be nonempty HxWx3 RGB")
    if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
        raise ValueError("sampling image must contain finite RGB in [0,1]")
    if quads.ndim != 3 or quads.shape[1:] != (4, 2) or not np.isfinite(quads).all():
        raise ValueError("sampling quadrilaterals must have shape (cells,4,2)")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError("sampling count must be an integer from 1 to 4096")
    h, w = image.shape[:2]
    if (quads[..., 0] < 0).any() or (quads[..., 0] > w - 1).any() or (quads[..., 1] < 0).any() or (quads[..., 1] > h - 1).any():
        raise ValueError("sampling geometry lies outside the image")
    side = int(math.ceil(math.sqrt(count)))
    axis = np.linspace(0, 1, side) if side > 1 else np.array([0.5])
    u, v = np.meshgrid(axis, axis)
    uv = np.stack((u.ravel(), v.ravel()), axis=-1)
    if len(uv) > count:
        uv = uv[np.linspace(0, len(uv) - 1, count).round().astype(int)]
    u, v = uv[None, :, 0:1], uv[None, :, 1:2]
    q = quads[:, None]
    points = (1-v) * ((1-u) * q[:, :, 0] + u * q[:, :, 1]) + v * ((1-u) * q[:, :, 3] + u * q[:, :, 2])
    x, y = points[..., 0], points[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = (x-x0)[..., None], (y-y0)[..., None]
    values = ((1-dx)*(1-dy)*image[y0, x0] + dx*(1-dy)*image[y0, x1]
              + (1-dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])
    # Floating interpolation can exceed a unit endpoint by a final rounding bit.
    return np.clip(values, 0, 1)
