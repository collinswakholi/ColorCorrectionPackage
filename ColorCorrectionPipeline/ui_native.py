"""Explicit native UI image rules, separate from existing staged calibrations.

This runtime records the UI's uint8 decode, ordered sampling and joint FFC/GC
equations. It does not reinterpret models fitted with the staged UI policy.
Equations are ported from the MIT-licensed CCP UI color, sample and FFC modules.
"""

import math

import numpy as np

RUNTIME = "native-v1"
HIGH_RUNTIME = "precision-f64-v1"
D65 = (0.31271, 0.32902)
# Native D50 coordinates differ from the canonical-reference shorthand.
D50 = (0.34567, 0.35850)
EPSILON, KAPPA = 216.0 / 24389.0, 24389.0 / 27.0
# ColorChecker Classic 24 Lab D50 values of the ColorCorrect Studio catalog.
CLASSIC24_LAB = (
    (37.54, 14.37, 14.92),
    (64.66, 19.27, 17.5),
    (49.32, -3.82, -22.54),
    (43.46, -12.74, 22.72),
    (54.94, 9.61, -24.79),
    (70.48, -32.26, -0.37),
    (62.73, 35.83, 56.5),
    (39.43, 10.75, -45.17),
    (50.57, 48.64, 16.67),
    (30.1, 22.54, -20.87),
    (71.77, -24.13, 58.19),
    (71.51, 18.24, 67.37),
    (28.37, 15.42, -49.8),
    (54.38, -39.72, 32.27),
    (42.43, 51.05, 28.62),
    (81.8, 2.67, 80.41),
    (50.63, 51.28, -14.12),
    (49.57, -29.71, -28.32),
    (95.19, -1.03, 2.93),
    (81.29, -0.57, 0.44),
    (66.89, -0.75, -0.06),
    (50.76, -0.13, 0.14),
    (35.63, -0.46, -0.48),
    (20.64, 0.07, -0.46),
)


def runtime(source):
    """Return the image runtime recorded by fitted models or named by a fitting policy.

    ``source`` is a model container (its saved ``metadata``), a ``FittingPolicy``
    or a policy dictionary. Prediction never resolves defaults here.
    """
    metadata = getattr(source, "metadata", None)
    policy = metadata.get("fitting_policy", {}) if metadata is not None else source
    if not isinstance(policy, dict):
        return policy.ui_runtime if policy.version == "ui-compatible-v1" else None
    return policy.get("ui_runtime") if policy.get("version") == "ui-compatible-v1" else None


FFC_DOMAINS = ("lightness", "linear")


def ffc_domain(models):
    """Return the saved FFC gain domain; records without one replay as ``"lightness"``."""
    domain = (getattr(models, "metadata", None) or {}).get("ffc_domain", "lightness")
    if domain not in FFC_DOMAINS:
        raise ValueError("FFC domain must be 'lightness' or 'linear'")
    return domain


def enabled(source):
    return runtime(source) in (RUNTIME, HIGH_RUNTIME)


def high_precision(source):
    return runtime(source) == HIGH_RUNTIME


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
    # Each row is written into one owned result instead of stacking temporaries.
    result = None
    for index, row in enumerate(matrix):
        channel = values[..., 0] * row[0] + values[..., 1] * row[1] + values[..., 2] * row[2]
        if result is None:
            result = np.empty(np.shape(channel) + (len(matrix),), dtype=np.result_type(channel))
        result[..., index] = channel
    return result


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
    # Both branches are evaluated per element; np.where selects exactly the
    # values the former masked assignment wrote, without gather/scatter copies.
    # Unselected bases below -0.055 would only warn; their values are discarded.
    with np.errstate(invalid="ignore"):
        return np.where(values > 0.04045, ((values + 0.055) / 1.055) ** 2.4, values / 12.92)


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
        magnitude = np.abs(values)
        result = np.where(
            magnitude > 0.0031308,
            np.sign(values) * (1.055 * magnitude ** (1.0 / 2.4) - 0.055),
            values * 12.92,
        )
    return result


def _xyz_to_lab(values, white):
    scaled = values / _white(white)
    f = np.where(scaled > EPSILON, np.cbrt(scaled), (KAPPA * scaled + 16.0) / 116.0)
    result = np.empty(f.shape, dtype=f.dtype)
    result[..., 0] = 116.0 * f[..., 1] - 16.0
    result[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    result[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return result


def _lab_to_xyz(values, white):
    fy = (values[..., 0] + 16.0) / 116.0
    f = np.empty(np.shape(fy) + (3,), dtype=np.result_type(fy, values[..., 1] / 500.0))
    f[..., 0] = values[..., 1] / 500.0 + fy
    f[..., 1] = fy
    f[..., 2] = fy - values[..., 2] / 200.0
    return np.where(f > 6.0 / 29.0, f * f * f, (116.0 * f - 16.0) / KAPPA) * _white(white)


def lab(values, scalar=False):
    return _xyz_to_lab(
        _multiply(RGB_TO_XYZ, _decode(np.asarray(values, dtype=np.float64), scalar)), D65
    )


def lightness(rgb, dtype, *, domain="lightness"):
    """Compute only the native Lab L* needed by the white-field fit, in row blocks.

    ``domain="linear"`` returns relative luminance ``100*Y`` instead, so the
    white-field ratios describe a linear-light gain.
    """
    light = np.empty(rgb.shape[:2], dtype=dtype)
    ry, gy, by = RGB_TO_XYZ[1]
    rows = max(1, 65536 // rgb.shape[1])
    for start in range(0, len(rgb), rows):
        block = _decode(rgb[start : start + rows])
        y = block[..., 0] * ry + block[..., 1] * gy + block[..., 2] * by
        if domain == "linear":
            light[start : start + rows] = 100.0 * y
            continue
        f = np.where(y > EPSILON, np.cbrt(y), (KAPPA * y + 16.0) / 116.0)
        light[start : start + rows] = 116.0 * f - 16.0
    return light


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
    """Classic24 reference in the UI catalog's own Lab D50 coordinates."""
    from .contracts import ReferenceProfile

    base = ReferenceProfile.classic24()
    # Preserve the original UI catalog coordinates rather than an inverse
    # round trip through the package's independently rounded D50 white.
    return ReferenceProfile.from_values(
        CLASSIC24_LAB,
        color_space="lab",
        white=D50,
        scale=1,
        profile_id="ui-native-classic24",
        revision=1,
        names=base.names,
        cell_ids=base.cell_ids,
        neutral_ids=base.neutral_ids,
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


def working_input(data, values, *, high=False):
    if high:
        return np.asarray(values, dtype=np.float64)
    validate_input(data)
    return normalize_codes(values)


def ordered_mean(samples):
    value = np.zeros((len(samples), 3))
    for i in range(samples.shape[1]):
        value += samples[:, i]
    return value / samples.shape[1]


def points(quads, count, random_seed=None):
    """Use the historical grid or a fixed, portable per-patch PRNG stream."""
    if random_seed is None:
        half = count // 2
        axis = 0.5 + (np.arange(count) - half) / max(half, 1) * 0.5 * 0.70
        u, v = np.meshgrid(axis, axis)
        u, v = u.ravel()[None, :, None], v.ravel()[None, :, None]
    else:
        # Unsigned LCG arithmetic is shared with the UI worker. A patch keeps
        # the same coordinate sequence across stages and requested counts.
        state = np.arange(len(quads), dtype=np.uint64) * 2654435761 + random_seed
        uv = np.empty((len(quads), count * 2))
        for i in range(count * 2):
            state = (1664525 * state + 1013904223) & np.uint64(0xFFFFFFFF)
            uv[:, i] = 0.15 + 0.70 * (state.astype(np.float64) / 4294967296.0)
        u, v = uv[:, ::2, None], uv[:, 1::2, None]
    q = np.asarray(quads)[:, None]
    return (
        (1 - u) * (1 - v) * q[:, :, 0]
        + u * (1 - v) * q[:, :, 1]
        + u * v * q[:, :, 2]
        + (1 - u) * v * q[:, :, 3]
    )


def sample_regions_float64(rgb, quads, count, *, xy=None):
    """Use the native sampling geometry against original float64 pixels."""
    image = np.asarray(rgb, dtype=np.float64)
    if xy is None:
        xy = points(quads, count)
    h, w = image.shape[:2]
    x, y = np.clip(xy[..., 0], 0, w - 1), np.clip(xy[..., 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    samples = (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy
        + image[y1, x1] * fx * fy
    )
    return np.clip(samples, 0, 1)


def face_sample_points(face):
    seed = face.provenance.get("sampling_seed")
    count = int(round(math.sqrt(face.sample_count))) if seed is None else face.sample_count
    return count, points(face.quadrilaterals, count, seed)


def apply_color(image, gain=None, curve=None, scalar=False, rounded=True, domain="lightness"):
    """Apply the joint FFC/GC equations.

    ``domain="lightness"`` (the default and the only historical rule) scales
    Lab L* by the gain. ``domain="linear"`` decodes sRGB, multiplies all three
    linear-light channels by the same spatial gain and re-encodes; GC, when
    present, then maps L* of that linear-light result.
    """
    if gain is not None and domain == "linear":
        linear = _decode(np.asarray(image, dtype=np.float64), scalar)
        linear *= np.asarray(gain)[..., None]
        if curve is None:
            result = np.clip(_encode(linear, scalar=scalar), 0, 1)
            return result.astype(np.float32).astype(np.float64) if rounded else result
        value = _xyz_to_lab(_multiply(RGB_TO_XYZ, linear), D65)
        value[..., 0] = np.clip(value[..., 0], 0, 100)
    else:
        value = lab(image, scalar=scalar)
        if gain is not None:
            value[..., 0] = np.clip(value[..., 0] * gain, 0, 100)
    if curve is not None:
        value[..., 0] = np.clip(curve.predict(value[..., 0]), 0, 100)
    result = np.clip(_encode(_multiply(XYZ_TO_RGB, _lab_to_xyz(value, D65)), scalar=scalar), 0, 1)
    return result.astype(np.float32).astype(np.float64) if rounded else result
