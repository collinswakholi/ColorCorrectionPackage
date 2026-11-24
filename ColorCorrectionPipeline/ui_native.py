"""Explicit native UI image rules, separate from existing staged calibrations.

This runtime records the UI's uint8 decode, ordered sampling and joint FFC/GC
equations. It does not reinterpret models fitted with the staged UI policy.
Equations are ported from the MIT-licensed CCP UI color, sample and FFC modules.
"""

import math
from dataclasses import replace

import numpy as np

RUNTIME = "native-v1"
HIGH_RUNTIME = "precision-f64-v1"
D65 = (0.31271, 0.32902)
# Native D50 coordinates differ from the canonical-reference shorthand.
D50 = (0.34567, 0.35850)
EPSILON, KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def runtime(value):
    """Return the persisted runtime without resolving defaults during prediction."""
    metadata = getattr(value, "metadata", None)
    if metadata is not None:
        policy = metadata.get("fitting_policy", {})
    else:
        policy = getattr(value, "fitting_policy", {})
    if not isinstance(policy, dict):
        return policy.ui_runtime if policy.version == "ui-compatible-v1" else None
    return policy.get("ui_runtime") if policy.get("version") == "ui-compatible-v1" else None


FFC_DOMAINS = ("lightness", "linear")


def ffc_domain(value):
    """Return the FFC gain domain; records without one replay as ``"lightness"``.

    A fitting owner carries the requested domain until its metadata is
    committed; fitted model containers read the persisted metadata.
    """
    if hasattr(value, "models"):
        domain = getattr(value, "_ffc_domain", "lightness")
    else:
        metadata = getattr(value, "metadata", None) or {}
        domain = metadata.get("ffc_domain", "lightness")
    if domain not in FFC_DOMAINS:
        raise ValueError("FFC domain must be 'lightness' or 'linear'")
    return domain


def enabled(value):
    return runtime(value) in (RUNTIME, HIGH_RUNTIME)


def high_precision(value):
    return runtime(value) == HIGH_RUNTIME


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


def sample_ffc(owner, face):
    from .ui_fitting import sample_regions

    count, xy = face_sample_points(face)
    high = high_precision(owner)
    samples = (sample_regions_float64 if high else sample_regions)(
        owner.Image, face.quadrilaterals, count, xy=xy
    )
    h, w = owner.Image.shape[:2]
    dtype = np.float64 if high else np.float32
    x = (np.clip(xy[..., 0], 0, w - 1) * (w - 1) / max(w - 1, 1)).astype(dtype)
    y = (np.clip(xy[..., 1], 0, h - 1) * (h - 1) / max(h - 1, 1)).astype(dtype)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0.astype(dtype)
    fy = y - y0.astype(dtype)
    g = owner.models.model_ffc.astype(dtype)
    gains = (
        g[y0, x0] * (1 - fy) * (1 - fx)
        + g[y1, x0] * fy * (1 - fx)
        + g[y0, x1] * (1 - fy) * fx
        + g[y1, x1] * fy * fx
    )
    corrected = apply_color(samples, gains, scalar=True, rounded=False, domain=ffc_domain(owner))
    clipped = np.mean(np.any(corrected >= face.clipping_threshold, axis=2), axis=1)
    return replace(
        face,
        measured_rgb=ordered_mean(corrected),
        clipped_fractions=clipped,
        provenance={
            **dict(face.provenance),
            "measurement_policy": runtime(owner) + "/FFC-after-bilinear",
            "sample_reduction": "row-order-f64",
        },
    )


def render_fit_stage(owner, image, stage):
    # Fit metadata is committed only after all stages succeed. Use the resolved
    # policy and the same guarded byte-runtime renderer as bundle prediction.
    domain = ffc_domain(owner)
    if not high_precision(owner) and image.shape[0] * image.shape[1] >= 65536:
        from .ui_execution import try_execute

        result = try_execute(
            image,
            owner.models,
            (stage,),
            fit_policy=owner.fitting_policy.to_dict(),
            ffc_domain=domain,
        )
        if result is not None:
            return result[stage]
    return apply_color(
        image,
        owner.models.model_ffc,
        owner.models.model_gc if stage == "GC" else None,
        rounded=not high_precision(owner),
        domain=domain,
    )


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


def predict(image, models, selected, chunk_pixels=None, *, native_codes=False):
    from . import modern, ui_stages

    if (
        modern.apply_tone is not modern._REFERENCE_APPLY_TONE
        or ui_stages.apply_ffc is not ui_stages._REFERENCE_APPLY_FFC
    ):
        raise ValueError("native UI fusion does not support overridden tone/FFC hooks")
    from .ui_execution import try_execute

    high = high_precision(models)
    domain = ffc_domain(models)
    if not high:
        result = try_execute(
            image, models, selected, native_codes=native_codes, chunk_pixels=chunk_pixels
        )
        if result is not None:
            return result
    if high:
        return _predict_high(image, models, selected, chunk_pixels, domain)
    from .core.correction import predict_
    from .core.correction import chunked_cc

    source = (
        np.asarray(image, dtype=np.float64)
        if high
        else (
            normalize_codes(image)
            if native_codes
            else np.asarray(image, dtype=np.float32).astype(np.float64)
        )
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
                current = apply_color(source, gain, rounded=not high, domain=domain)
        elif name == "GC" and curve is not None:
            current = apply_color(source, gain, curve, rounded=not high, domain=domain)
        elif name == "WB" and models.model_wb is not None:
            current = np.clip(current @ models.model_wb, 0, 1)
            if not high:
                current = current.astype(np.float32).astype(np.float64)
        elif name == "CC" and models.model_cc is not None:
            extended = "basis" in models.model_cc[1]
            if chunk_pixels is not None or high or extended:
                current = chunked_cc(current, models.model_cc, chunk_pixels or 65536)
            else:
                current = predict_(RGB=current, M=models.model_cc[0])
            current = np.clip(current, 0, 1)
            if not high:
                current = current.astype(np.float32).astype(np.float64)
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


# Pointwise sub-block for threaded high-precision prediction. Smaller
# temporaries than the 65536-pixel CC chunk avoid most fresh page mapping; 32768
# was the fastest and most stable size on 4K frames (see docs/performance.md).
_SUB_BLOCK = 32768


def _predict_high(image, models, selected, chunk_pixels, domain):
    """Float64 stages in bounded pixel blocks, optionally on several threads.

    Every stage is pointwise, so each block reproduces the full-frame equations
    exactly. Blocks keep the historical CC chunk boundaries (``chunk_pixels`` or
    65536 pixels) and the LUT/direct decision is still made once per image. Only
    the selected stage outputs are allocated at full frame size; intermediate
    stages and the validity checks stay within one block.
    """
    from ._cpu_workers import ranges, run_slices, worker_count
    from .core.correction import cc_block, cc_uses_lut

    source = np.asarray(image, dtype=np.float64)
    order = ("FFC", "GC", "WB", "CC")
    stop = max(order.index(k) for k in selected)
    gain = models.model_ffc
    curve = models.model_gc
    if gain is not None and np.shape(gain) != source.shape[:2]:
        raise ValueError("native UI FFC requires the calibrated image dimensions")
    flat = source.reshape(-1, 3)
    gains = None if gain is None else np.asarray(gain).reshape(-1)
    pixels = len(flat)
    chunk = chunk_pixels or 65536
    cc = models.model_cc if stop >= 3 else None
    use_lut = cc_uses_lut(cc, pixels) if cc is not None else False
    wb = models.model_wb
    fused_ffc = gain is not None and ("FFC" in selected or curve is None or stop == 0)
    outputs = {name: np.empty(flat.shape, dtype=np.float64) for name in selected}

    def valid(current, block):
        return not (
            current.shape != block.shape
            or not np.isfinite(current).all()
            or np.any(current < 0)
            or np.any(current > 1)
        )

    def run(first_block, last_block):
        # Return the earliest stage index that produced invalid data, if any.
        failed = None
        staged = np.empty((min(chunk, pixels), 3), dtype=np.float64) if stop == 3 else None
        for start in range(first_block * chunk, min(last_block * chunk, pixels), chunk):
            end = min(start + chunk, pixels)
            # Pointwise colour stages use sub-blocks small enough to reuse heap
            # memory; CC keeps the historical chunk so its matrix products see
            # exactly the same operands.
            for first in range(start, end, _SUB_BLOCK):
                last = min(first + _SUB_BLOCK, end)
                block = flat[first:last]
                block_gain = None if gains is None else gains[first:last]
                current = block
                for index, name in enumerate(order[: min(stop, 2) + 1]):
                    if name == "FFC" and fused_ffc:
                        current = apply_color(block, block_gain, rounded=False, domain=domain)
                    elif name == "GC" and curve is not None:
                        current = apply_color(
                            block, block_gain, curve, rounded=False, domain=domain
                        )
                    elif name == "WB" and wb is not None:
                        current = np.clip(current @ wb, 0, 1)
                    if not valid(current, block):
                        failed = index if failed is None else min(failed, index)
                        break
                    if name in outputs:
                        outputs[name][first:last] = current
                else:
                    if staged is not None:
                        staged[first - start : last - start] = current
                    continue
                break
            else:
                if staged is not None:
                    current = staged[: end - start]
                    if cc is not None:
                        current = np.clip(cc_block(current, cc, use_lut), 0, 1)
                    if not valid(current, flat[start:end]):
                        failed = 3 if failed is None else min(failed, 3)
                        continue
                    if "CC" in outputs:
                        outputs["CC"][start:end] = current
        return failed

    blocks = (pixels + chunk - 1) // chunk
    workers = min(blocks, worker_count(pixels, 65536))
    failures = [
        index for index in run_slices(run, ranges(blocks, workers), workers) if index is not None
    ]
    if failures:
        raise FloatingPointError("native UI %s produced invalid image data" % order[min(failures)])
    return {name: outputs[name].reshape(source.shape) for name in order if name in outputs}
