"""Explicit native UI image rules, separate from existing staged calibrations.

This runtime records the UI's uint8 decode, ordered sampling and joint FFC/GC
equations. It does not reinterpret models fitted with the staged UI policy.
Equations are ported from the MIT-licensed CCP UI color, sample and FFC modules.
"""

import math
from dataclasses import replace

import numpy as np

RUNTIME = "native-v1"
D65 = (0.31271, 0.32902)
# Native D50 coordinates differ from the canonical-reference shorthand.
D50 = (0.34567, 0.35850)
EPSILON, KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def enabled(value):
    metadata = getattr(value, "metadata", None)
    if metadata is not None:
        policy = metadata.get("fitting_policy", {})
    else:
        policy = getattr(value, "fitting_policy", {})
    if not isinstance(policy, dict):
        return policy.version == "ui-compatible-v1" and policy.ui_runtime == RUNTIME
    return policy.get("version") == "ui-compatible-v1" and policy.get("ui_runtime") == RUNTIME


def _white(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _inverse(matrix):
    """Retain the native cofactor order for the fixed 3x3 color matrices."""
    a, b, c, d, e, f, g, h, i = map(float, matrix.ravel())
    aa, bb, cc = e * i - f * h, -(d * i - f * g), d * h - e * g
    dd, ee, ff = -(b * i - c * h), a * i - c * g, -(a * h - b * g)
    gg, hh, ii = b * f - c * e, -(a * f - c * d), a * e - b * d
    factor = 1.0 / (a * aa + b * bb + c * cc)
    return np.array([[aa, dd, gg], [bb, ee, hh], [cc, ff, ii]]) * factor


def _multiply(matrix, values):
    # Keep the native left-to-right scalar additions, avoiding a BLAS tree.
    return np.stack(
        [
            values[..., 0] * row[0] + values[..., 1] * row[1] + values[..., 2] * row[2]
            for row in matrix
        ],
        axis=-1,
    )


_PRIMARY = np.array(
    [
        [0.64 / 0.33, 0.30 / 0.60, 0.15 / 0.06],
        [1.0, 1.0, 1.0],
        [(1.0 - 0.64 - 0.33) / 0.33, (1.0 - 0.30 - 0.60) / 0.60, (1.0 - 0.15 - 0.06) / 0.06],
    ]
)
RGB_TO_XYZ = _PRIMARY * _multiply(_inverse(_PRIMARY), _white(D65))
XYZ_TO_RGB = _inverse(RGB_TO_XYZ)
WHITE = _white(D65)
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
_BRADFORD_INVERSE = np.array(
    [
        [0.9869929054667123, -0.14705425642099013, 0.15996265166373125],
        [0.4323052697233945, 0.5183602715367776, 0.049291228212855594],
        [-0.008528664575177328, 0.04004282165408486, 0.9684866957875502],
    ]
)


def _adapt(xyz, source, destination):
    cone = _multiply(_BRADFORD, xyz)
    scale = _multiply(_BRADFORD, _white(destination)) / _multiply(_BRADFORD, _white(source))
    return _multiply(_BRADFORD_INVERSE, cone * scale)


def _decode(values, scalar=False):
    if scalar:
        return np.array(
            [
                v / 12.92 if v <= 0.04045 else math.pow((v + 0.055) / 1.055, 2.4)
                for v in values.ravel()
            ]
        ).reshape(values.shape)
    result = values / 12.92
    mask = values > 0.04045
    result[mask] = ((values[mask] + 0.055) / 1.055) ** 2.4
    return result


def _encode(values, scalar=False):
    values = np.clip(values, 0, 1)
    if scalar:
        result = np.array(
            [
                (
                    v * 12.92
                    if abs(v) <= 0.0031308
                    else math.copysign(1.055 * math.pow(abs(v), 1.0 / 2.4) - 0.055, v)
                )
                for v in values.ravel()
            ]
        ).reshape(values.shape)
    else:
        result = values * 12.92
        mask = np.abs(values) > 0.0031308
        result[mask] = np.sign(values[mask]) * (1.055 * np.abs(values[mask]) ** (1.0 / 2.4) - 0.055)
    return result


def _xyz_to_lab(values, white):
    scaled = values / _white(white)
    f = np.where(scaled > EPSILON, np.cbrt(scaled), (KAPPA * scaled + 16.0) / 116.0)
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def _lab_to_xyz(values, white):
    fy = (values[..., 0] + 16.0) / 116.0
    f = np.stack([values[..., 1] / 500.0 + fy, fy, fy - values[..., 2] / 200.0], axis=-1)
    return np.where(f > 6.0 / 29.0, f * f * f, (116.0 * f - 16.0) / KAPPA) * _white(white)


def lab(values, scalar=False):
    return _xyz_to_lab(
        _multiply(RGB_TO_XYZ, _decode(np.asarray(values, dtype=np.float64), scalar)), D65
    )


def reference(profile):
    """Recover physical source values when retained; validate them against truth."""
    from .colorimetry import reference_values_to_lab_d50, white_xy

    retained = profile.provenance.get("source_values")
    if retained is not None:
        if set(retained) != {"values", "color_space", "white", "scale", "adaptation"}:
            raise ValueError("invalid retained reference source values")
        converted = reference_values_to_lab_d50(
            retained["values"],
            color_space=retained["color_space"],
            white=retained["white"],
            scale=retained["scale"],
            adaptation=retained["adaptation"],
        )
        if not np.allclose(converted, profile.canonical_lab_d50, rtol=0, atol=1e-10):
            raise ValueError("retained reference source contradicts canonical Lab truth")
        # A CAT02-derived canonical reference records a different adaptation.
        # Preserve that canonical truth instead of re-adapting its source with
        # Bradford merely to recover the UI's arithmetic order.
        if retained["adaptation"] != "Bradford":
            retained = None
    if retained is not None:
        values = np.asarray(retained["values"], dtype=np.float64) / float(retained["scale"])
        white = white_xy(retained["white"])
        if retained["color_space"] == "xyz":
            xyz = values
        elif retained["color_space"] == "lab":
            xyz = _lab_to_xyz(np.asarray(retained["values"], dtype=np.float64), white)
        else:
            xyz = None
        if xyz is not None:
            if tuple(white) != D65:
                xyz = _adapt(xyz, white, D65)
        else:
            retained = None
    if retained is None:
        xyz = _adapt(
            _lab_to_xyz(profile.canonical_lab_d50, (0.3457, 0.3585)), (0.3457, 0.3585), D65
        )
    # The UI canonicalizes references to its Lab D50 before fitting any target.
    native_lab = _xyz_to_lab(_adapt(xyz, D65, D50), D50)
    working_xyz = _adapt(_lab_to_xyz(native_lab, D50), D50, D65)
    linear = _multiply(XYZ_TO_RGB, working_xyz)
    return linear, _encode(linear, scalar=True), _xyz_to_lab(working_xyz, D65)


def default_reference():
    from .ui_fitting import default_reference as staged_reference
    from .contracts import ReferenceProfile

    profile = staged_reference()
    # Preserve the original UI catalog coordinates rather than an inverse
    # round trip through the package's independently rounded D50 white.
    from .ui_fitting import _UI_CLASSIC_LAB

    return ReferenceProfile.from_values(
        _UI_CLASSIC_LAB,
        color_space="lab",
        white=D50,
        scale=1,
        profile_id="ui-native-classic24",
        revision=1,
        names=profile.names,
        cell_ids=profile.cell_ids,
        neutral_ids=profile.neutral_ids,
        retain_source=True,
    )


def validate_input(data):
    if (
        data.source_dtype != "uint8"
        or tuple(data.source_range) != (0.0, 255.0)
        or data.color_space != "srgb"
    ):
        raise ValueError("native UI runtime requires full-range uint8 sRGB input")


def normalize_codes(values):
    codes = np.rint(np.asarray(values) * 255.0).astype(np.float32)
    return (codes * np.float32(1.0 / 255.0)).astype(np.float64)


def working_input(data, values):
    validate_input(data)
    return normalize_codes(values)


def letterbox(image, size):
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


def homography(source, destination):
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


def warp(points, matrix):
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


def ordered_mean(samples):
    value = np.zeros((len(samples), 3))
    for i in range(samples.shape[1]):
        value += samples[:, i]
    return value / samples.shape[1]


def points(quads, count):
    half = count // 2
    axis = 0.5 + (np.arange(count) - half) / max(half, 1) * 0.5 * 0.70
    u, v = np.meshgrid(axis, axis)
    u = u.ravel()[None, :, None]
    v = v.ravel()[None, :, None]
    q = np.asarray(quads)[:, None]
    return (
        (1 - u) * (1 - v) * q[:, :, 0]
        + u * (1 - v) * q[:, :, 1]
        + u * v * q[:, :, 2]
        + (1 - u) * v * q[:, :, 3]
    )


def sample_ffc(owner, face):
    from .ui_fitting import sample_regions

    count = int(round(math.sqrt(face.sample_count)))
    samples = sample_regions(owner.Image, face.quadrilaterals, count)
    xy = points(face.quadrilaterals, count)
    h, w = owner.Image.shape[:2]
    x = (np.clip(xy[..., 0], 0, w - 1) * (w - 1) / max(w - 1, 1)).astype(np.float32)
    y = (np.clip(xy[..., 1], 0, h - 1) * (h - 1) / max(h - 1, 1)).astype(np.float32)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0.astype(np.float32)
    fy = y - y0.astype(np.float32)
    g = owner.models.model_ffc.astype(np.float32)
    gains = (
        g[y0, x0] * (1 - fy) * (1 - fx)
        + g[y1, x0] * fy * (1 - fx)
        + g[y0, x1] * (1 - fy) * fx
        + g[y1, x1] * fy * fx
    )
    corrected = apply_color(samples, gains, scalar=True, rounded=False)
    clipped = np.mean(np.any(corrected >= face.clipping_threshold, axis=2), axis=1)
    return replace(
        face,
        measured_rgb=ordered_mean(corrected),
        clipped_fractions=clipped,
        provenance={
            **dict(face.provenance),
            "measurement_policy": "ui-native-v1/FFC-after-bilinear",
            "sample_reduction": "row-order-f64",
        },
    )


def apply_color(image, gain=None, curve=None, scalar=False, rounded=True):
    value = lab(image, scalar=scalar)
    if gain is not None:
        value[..., 0] = np.clip(value[..., 0] * gain, 0, 100)
    if curve is not None:
        value[..., 0] = np.clip(curve.predict(value[..., 0]), 0, 100)
    result = np.clip(_encode(_multiply(XYZ_TO_RGB, _lab_to_xyz(value, D65)), scalar=scalar), 0, 1)
    return result.astype(np.float32).astype(np.float64) if rounded else result


def predict(image, models, selected, chunk_pixels=None, *, native_codes=False):
    from . import modern, ui_stages

    if (
        modern.apply_tone is not modern._REFERENCE_APPLY_TONE
        or ui_stages.apply_ffc is not ui_stages._REFERENCE_APPLY_FFC
    ):
        raise ValueError("native UI fusion does not support overridden tone/FFC hooks")
    from .ui_execution import try_execute

    if chunk_pixels is None:
        result = try_execute(image, models, selected, native_codes=native_codes)
        if result is not None:
            return result
    from .core.correction import predict_
    from .prediction import chunked_cc

    source = (
        normalize_codes(image)
        if native_codes
        else np.asarray(image, dtype=np.float32).astype(np.float64)
    )
    current = source
    result = {}
    order = ("FFC", "GC", "WB", "CC")
    stop = max(order.index(k) for k in selected)
    gain = models.model_ffc
    curve = models.model_gc
    if gain is not None and np.shape(gain) != source.shape[:2]:
        raise ValueError("native UI FFC requires the calibrated image dimensions")
    for index, name in enumerate(order[: stop + 1]):
        if name == "FFC" and gain is not None:
            if "FFC" in selected or curve is None or stop == 0:
                current = apply_color(source, gain)
        elif name == "GC" and curve is not None:
            current = apply_color(source, gain, curve)
        elif name == "WB" and models.model_wb is not None:
            current = np.clip(current @ models.model_wb, 0, 1).astype(np.float32).astype(np.float64)
        elif name == "CC" and models.model_cc is not None:
            if chunk_pixels is not None:
                current = chunked_cc(current, models.model_cc, chunk_pixels)
            else:
                current = predict_(RGB=current, M=models.model_cc[0])
            current = np.clip(current, 0, 1).astype(np.float32).astype(np.float64)
        if (
            current.shape != source.shape
            or not np.isfinite(current).all()
            or np.any(current < 0)
            or np.any(current > 1)
        ):
            raise FloatingPointError("native UI %s produced invalid image data" % name)
        if name in selected:
            result[name] = current.copy()
    return result
