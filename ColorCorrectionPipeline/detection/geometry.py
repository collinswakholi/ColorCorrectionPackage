"""Chart geometry recipes ported from ColorCorrect Studio.

The detector keeps each recipe's arithmetic: ``precise_homography`` solves the
UI-compatible geometry in float64, and the ``native_*`` functions, ``order_quad``
and ``anchor_center`` reproduce the native UI's letterbox and lattice equations.
"""

import math

import numpy as np


def precise_homography(source, destination):
    """Solve the UI geometry in float64 without OpenCV's float32 point cast."""
    matrix = np.zeros((8, 8), dtype=np.float64)
    rhs = np.asarray(destination, dtype=np.float64).reshape(-1)
    for index, ((x, y), (u, v)) in enumerate(zip(source, destination)):
        matrix[2 * index] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        matrix[2 * index + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
    try:
        return np.r_[np.linalg.solve(matrix, rhs), 1.0].reshape(3, 3)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan)


def native_letterbox(image, size):
    """Letterbox BGR uint8 to the detector input with the native bilinear resize."""
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(math.floor(w * gain + 0.5)), int(math.floor(h * gain + 0.5))
    left, top = (size - nw) // 2, (size - nh) // 2
    source = image[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    sx = (np.arange(nw, dtype=np.float64) + 0.5) * w / nw - 0.5
    ix = np.clip(np.floor(sx).astype(np.int64), 0, w - 1)
    jx = np.minimum(ix + 1, w - 1)
    fx = (sx - ix)[None, :, None]
    # Bound interpolation workspaces independently of source dimensions.
    for start in range(0, nh, 16):
        sy = (np.arange(start, min(start + 16, nh), dtype=np.float64) + 0.5) * h / nh - 0.5
        iy = np.clip(np.floor(sy).astype(np.int64), 0, h - 1)
        jy = np.minimum(iy + 1, h - 1)
        fy = (sy - iy)[:, None, None]
        upper = (1.0 - fx) * source[iy[:, None], ix] + fx * source[iy[:, None], jx]
        lower = (1.0 - fx) * source[jy[:, None], ix] + fx * source[jy[:, None], jx]
        values = (1.0 - fy) * upper + fy * lower
        canvas[top + start : top + start + len(iy), left : left + nw] = np.clip(
            np.floor(values * 255.0 + 0.5), 0, 255
        ).astype(np.uint8)
    return (
        np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0),
        gain,
        left,
        top,
    )


def native_homography(source, destination):
    """Solve the native UI geometry with its own Gauss-Jordan elimination order."""
    matrix = np.zeros((8, 9))
    for i, ((x, y), (u, v)) in enumerate(zip(source, destination)):
        matrix[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y, u]
        matrix[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v]
    for col in range(8):
        pivot = col + int(np.argmax(np.abs(matrix[col:, col])))
        if abs(matrix[pivot, col]) < 1e-12:
            return np.full((3, 3), np.nan)
        if pivot != col:
            matrix[[pivot, col], col:] = matrix[[col, pivot], col:]
        matrix[col, col:] /= matrix[col, col]
        for row in range(8):
            if row != col:
                matrix[row, col:] -= matrix[row, col] * matrix[col, col:]
    return np.r_[matrix[:, 8], 1.0].reshape(3, 3)


def native_warp(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    z = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return np.stack(
        [(matrix[i, 0] * x + matrix[i, 1] * y + matrix[i, 2]) / z for i in (0, 1)], axis=1
    )


def order_quad(points):
    points = [(float(x), float(y)) for x, y in points]
    cx, cy = (sum(p[k] for p in points) * 0.25 for k in (0, 1))
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    area = sum(
        points[i][0] * points[(i + 1) % 4][1] - points[i][1] * points[(i + 1) % 4][0]
        for i in range(4)
    )
    if area < 0:
        points.reverse()
    start = min(range(4), key=lambda i: points[i][0] + points[i][1])
    return np.asarray(points[start:] + points[:start])


def anchor_center(box):
    x, y, w, h, angle = map(float, box)
    vx, vy = (w / 2.0) * math.cos(angle), (w / 2.0) * math.sin(angle)
    ux, uy = (-h / 2.0) * math.sin(angle), (h / 2.0) * math.cos(angle)
    q = order_quad(
        [
            (x + vx + ux, y + vy + uy),
            (x + vx - ux, y + vy - uy),
            (x - vx - ux, y - vy - uy),
            (x - vx + ux, y - vy + uy),
        ]
    )
    return 0.25 * (q[0] + q[1] + q[2] + q[3])
