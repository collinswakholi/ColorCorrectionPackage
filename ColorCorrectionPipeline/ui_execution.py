"""Bounded-memory CPU execution of recorded UI image equations.

Staged calibrations retain each float32 boundary. Native calibrations retain
their recorded joint FFC/GC conversion. Guarded optional execution falls back
to the corresponding reference implementation when a kernel cannot run.
"""

import logging
import threading
import numpy as np

from .core import accel

_enabled = accel.HAS_NUMBA
_log = logging.getLogger(__name__)
_decode_values = None


if accel.HAS_NUMBA:
    from .core.execution import _tone_lightness

    # Numba 0.58 repeatedly deep-copies nested always-inline image IR. Keep
    # helpers separate on older compilers; this changes compilation, not math.
    try:
        _compiler_version = tuple(int(v) for v in accel.numba.__version__.split(".")[:2])
        _inline = "always" if _compiler_version >= (0, 60) else "never"
    except (AttributeError, ValueError):
        _inline = "never"

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _decode(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    @accel.nb_njit(cache=True, fastmath=False)
    def _build_decode_values():
        values = np.empty((256, 4), dtype=np.float64)
        for code in range(256):
            staged = np.float64(np.float32(code / 255.0))
            native = np.float64(np.float32(code) * np.float32(1.0 / 255.0))
            values[code, 0], values[code, 1] = staged, _decode(staged)
            values[code, 2], values[code, 3] = native, _decode(native)
        return values

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _decode_cached(v, values):
        index = int(v * 255.0 + 0.5)
        if 0 <= index < 256:
            if v == values[index, 0]:
                return values[index, 1]
            if v == values[index, 2]:
                return values[index, 3]
        return _decode(v)

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _encode(v):
        absolute = abs(v)
        value = (
            v * 12.92
            if absolute <= 0.0031308
            else (1.0 if v >= 0 else -1.0) * (1.055 * absolute ** (1.0 / 2.4) - 0.055)
        )
        return min(1.0, max(0.0, value))

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _color(r, g, b, gain, knots, coefficients, gamma, forward, inverse, white, decodes, native):
        r, g, b = _decode_cached(r, decodes), _decode_cached(g, decodes), _decode_cached(b, decodes)
        x = (r * forward[0, 0] + g * forward[0, 1] + b * forward[0, 2]) / white[0]
        y = (r * forward[1, 0] + g * forward[1, 1] + b * forward[1, 2]) / white[1]
        z = (r * forward[2, 0] + g * forward[2, 1] + b * forward[2, 2]) / white[2]
        epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
        fx = np.cbrt(x) if x > epsilon else (kappa * x + 16.0) / 116.0
        fy = np.cbrt(y) if y > epsilon else (kappa * y + 16.0) / 116.0
        fz = np.cbrt(z) if z > epsilon else (kappa * z + 16.0) / 116.0
        lightness, aa, bb = 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)
        if native:
            lightness = min(100.0, max(0.0, lightness * gain))
            if gamma:
                lightness = _tone_lightness(lightness, knots, coefficients)
        else:
            lightness = (
                _tone_lightness(lightness, knots, coefficients)
                if gamma
                else min(100.0, max(0.0, lightness * gain))
            )
        scaled, a, b = lightness / 116.0, aa / 500.0, bb / 200.0
        fy = (lightness + 16.0) / 116.0 if native else scaled + 16.0 / 116.0
        fx, fz = fy + a, fy - b
        x3, y3, z3 = fx * fx * fx, fy * fy * fy, fz * fz * fz
        x = (x3 if x3 > epsilon else (scaled + a) * 116.0 / kappa) * white[0]
        y = (y3 if lightness > kappa * epsilon else lightness / kappa) * white[1]
        z = (z3 if z3 > epsilon else (scaled - b) * 116.0 / kappa) * white[2]
        if native:
            x = (x3 if fx > 6.0 / 29.0 else (116.0 * fx - 16.0) / kappa) * white[0]
            y = (y3 if fy > 6.0 / 29.0 else (116.0 * fy - 16.0) / kappa) * white[1]
            z = (z3 if fz > 6.0 / 29.0 else (116.0 * fz - 16.0) / kappa) * white[2]
        return (
            np.float32(_encode(x * inverse[0, 0] + y * inverse[0, 1] + z * inverse[0, 2])),
            np.float32(_encode(x * inverse[1, 0] + y * inverse[1, 1] + z * inverse[1, 2])),
            np.float32(_encode(x * inverse[2, 0] + y * inverse[2, 1] + z * inverse[2, 2])),
        )

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _lookup(r, g, b, table, channel):
        size = table.shape[0]
        # Division by the stored grid step preserves the existing interpolator's
        # rounding; multiplying by size-1 can change boundary samples.
        step = 1.0 / (size - 1)
        rf, gf, bf = r / step, g / step, b / step
        r0, g0, b0 = int(min(rf, size - 2)), int(min(gf, size - 2)), int(min(bf, size - 2))
        rd, gd, bd = rf - r0, gf - g0, bf - b0
        wr, wg, wb = 1.0 - rd, 1.0 - gd, 1.0 - bd
        c00 = table[r0, g0, b0, channel] * wr + table[r0 + 1, g0, b0, channel] * rd
        c01 = table[r0, g0, b0 + 1, channel] * wr + table[r0 + 1, g0, b0 + 1, channel] * rd
        c10 = table[r0, g0 + 1, b0, channel] * wr + table[r0 + 1, g0 + 1, b0, channel] * rd
        c11 = table[r0, g0 + 1, b0 + 1, channel] * wr + table[r0 + 1, g0 + 1, b0 + 1, channel] * rd
        c0, c1 = c00 * wg + c10 * gd, c01 * wg + c11 * gd
        return np.float32(min(1.0, max(0.0, c0 * wb + c1 * bd)))

    @accel.nb_njit(cache=True, fastmath=False, inline=_inline)
    def _pixel(
        row,
        output,
        flat,
        gain,
        knots,
        coefficients,
        wb,
        table,
        stop,
        retained,
        forward,
        inverse,
        white,
        decodes,
        native,
        native_codes,
    ):
        r, g, b = (
            np.float64(np.float32(flat[row, 0])),
            np.float64(np.float32(flat[row, 1])),
            np.float64(np.float32(flat[row, 2])),
        )
        # Canonical inputs store code/255 in float64; recover the byte before
        # the native float32 reciprocal multiply, without a full-frame copy.
        if native_codes:
            r = np.float64(np.float32(np.rint(flat[row, 0] * 255.0)) * np.float32(1.0 / 255.0))
            g = np.float64(np.float32(np.rint(flat[row, 1] * 255.0)) * np.float32(1.0 / 255.0))
            b = np.float64(np.float32(np.rint(flat[row, 2] * 255.0)) * np.float32(1.0 / 255.0))
        source_r, source_g, source_b = r, g, b
        position = 0
        for stage in range(stop + 1):
            if (
                stage == 0
                and len(gain)
                and not (native and len(knots) and stop >= 1 and retained[0] != 0)
            ):
                rr, gg, bb = _color(
                    r,
                    g,
                    b,
                    gain[row],
                    knots,
                    coefficients,
                    False,
                    forward,
                    inverse,
                    white,
                    decodes,
                    native,
                )
                r, g, b = np.float64(rr), np.float64(gg), np.float64(bb)
            elif stage == 1 and len(knots):
                gamma_gain = 1.0
                if native and len(gain):
                    r, g, b = source_r, source_g, source_b
                    gamma_gain = gain[row]
                rr, gg, bb = _color(
                    r,
                    g,
                    b,
                    gamma_gain,
                    knots,
                    coefficients,
                    True,
                    forward,
                    inverse,
                    white,
                    decodes,
                    native,
                )
                r, g, b = np.float64(rr), np.float64(gg), np.float64(bb)
            elif stage == 2 and len(wb):
                rr = np.float32(min(1.0, max(0.0, r * wb[0] + g * 0.0 + b * 0.0)))
                gg = np.float32(min(1.0, max(0.0, r * 0.0 + g * wb[1] + b * 0.0)))
                bb = np.float32(min(1.0, max(0.0, r * 0.0 + g * 0.0 + b * wb[2])))
                r, g, b = np.float64(rr), np.float64(gg), np.float64(bb)
            elif stage == 3 and table.shape[0]:
                rr, gg, bb = (
                    _lookup(r, g, b, table, 0),
                    _lookup(r, g, b, table, 1),
                    _lookup(r, g, b, table, 2),
                )
                r, g, b = np.float64(rr), np.float64(gg), np.float64(bb)
            if position < len(retained) and stage == retained[position]:
                output[position, row, 0], output[position, row, 1], output[position, row, 2] = (
                    r,
                    g,
                    b,
                )
                position += 1

    @accel.nb_njit(cache=True, fastmath=False, nogil=True)
    def _run_into(
        output,
        flat,
        gain,
        knots,
        coefficients,
        wb,
        table,
        stop,
        retained,
        forward,
        inverse,
        white,
        decodes,
        native,
        native_codes,
    ):
        for row in range(len(flat)):
            _pixel(
                row,
                output,
                flat,
                gain,
                knots,
                coefficients,
                wb,
                table,
                stop,
                retained,
                forward,
                inverse,
                white,
                decodes,
                native,
                native_codes,
            )

    def _run_serial(
        flat,
        gain,
        knots,
        coefficients,
        wb,
        table,
        stop,
        retained,
        forward,
        inverse,
        white,
        decodes,
        native,
        native_codes,
    ):
        output = np.empty((len(retained), len(flat), 3), dtype=np.float64)
        _run_into(
            output,
            flat,
            gain,
            knots,
            coefficients,
            wb,
            table,
            stop,
            retained,
            forward,
            inverse,
            white,
            decodes,
            native,
            native_codes,
        )
        return output

    def _run(
        flat,
        gain,
        knots,
        coefficients,
        wb,
        table,
        stop,
        retained,
        forward,
        inverse,
        white,
        decodes,
        native,
        native_codes,
    ):
        from ._cpu_workers import worker_count, ranges, run_slices

        workers = worker_count(len(flat), 65536)
        if workers == 1:
            return _run_serial(
                flat,
                gain,
                knots,
                coefficients,
                wb,
                table,
                stop,
                retained,
                forward,
                inverse,
                white,
                decodes,
                native,
                native_codes,
            )
        output = np.empty((len(retained), len(flat), 3), dtype=np.float64)
        # Compile for the actual strides before worker threads enter the same
        # dispatcher. Every worker then owns disjoint rows of the output.
        _run_into(
            output[:, :1],
            flat[:0],
            gain[:0],
            knots,
            coefficients,
            wb,
            table,
            stop,
            retained,
            forward,
            inverse,
            white,
            decodes,
            native,
            native_codes,
        )

        def fill(start, end):
            local_gain = gain[start:end] if len(gain) else gain
            _run_into(
                output[:, start:end],
                flat[start:end],
                local_gain,
                knots,
                coefficients,
                wb,
                table,
                stop,
                retained,
                forward,
                inverse,
                white,
                decodes,
                native,
                native_codes,
            )

        run_slices(fill, ranges(len(flat), workers), workers)
        return output


def try_execute(image, models, selected, *, native_codes=False):
    """Return owned stage arrays, or None to retain the ordinary reference path."""
    global _enabled, _decode_values
    if (
        not _enabled
        or models.metadata.get("fitting_policy", {}).get("version") != "ui-compatible-v1"
    ):
        return None
    from . import modern, ui_stages
    from .core.correction import Regressor_Model, _ensure_lut
    from .core.execution import _tone_parameters
    from .core.lut_cache import inference_identity
    from .ui_fitting import UIPredictor
    from .colorimetry import _RGB_TO_XYZ, _XYZ_TO_RGB, white_xyz, D65

    if (
        modern.apply_tone is not modern._REFERENCE_APPLY_TONE
        or ui_stages.apply_ffc is not ui_stages._REFERENCE_APPLY_FFC
    ):
        return None
    order = ("FFC", "GC", "WB", "CC")
    indices = np.array([i for i, name in enumerate(order) if name in selected], dtype=np.int64)
    stop = int(indices[-1])
    gain = np.empty(0, dtype=np.float64)
    if models.model_ffc is not None:
        value = np.asarray(models.model_ffc)
        if value.shape != image.shape[:2]:
            return None
        if (
            value.dtype != np.float64
            or value.flags.writeable
            or not np.isfinite(value).all()
            or np.any(value <= 0)
        ):
            return None
        gain = value.ravel()
    knots, coefficients = np.empty(0), np.empty((0, 0))
    if stop >= 1 and models.model_gc is not None:
        parameters = _tone_parameters(models.model_gc)
        if parameters is None:
            return None
        knots, coefficients = parameters
    wb = np.empty(0)
    if stop >= 2 and models.model_wb is not None:
        matrix = np.asarray(models.model_wb)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, np.diag(np.diag(matrix)))
            or np.any(np.diag(matrix) <= 0)
        ):
            return None
        wb = np.ascontiguousarray(np.diag(matrix), dtype=np.float64)
    table = np.empty((0, 0, 0, 3))
    if stop >= 3 and models.model_cc is not None:
        model, params, method = models.model_cc
        if (
            method != "ours"
            or type(model) is not Regressor_Model
            or type(model.model) is not UIPredictor
            or not model.use_lut
            or image.shape[0] * image.shape[1] < model.lut_min_pixels
            or inference_identity(model)[0] is None
            or not _ensure_lut(model)
        ):
            return None
        table = model._lut
    from .ui_native import enabled as native_enabled

    native = native_enabled(models)
    if native:
        from .ui_native import RGB_TO_XYZ as forward, XYZ_TO_RGB as inverse, WHITE as white
    else:
        forward, inverse, white = _RGB_TO_XYZ, _XYZ_TO_RGB, white_xyz(D65)
    try:
        if _decode_values is None:
            decoded = _build_decode_values()
            _decode_values = np.frombuffer(decoded.tobytes(), dtype=np.float64).reshape(256, 4)
        # Batch workers already own concurrency; avoid nested native pools.
        kernel = _run if threading.current_thread() is threading.main_thread() else _run_serial
        values = kernel(
            image.reshape(-1, 3),
            gain,
            knots,
            coefficients,
            wb,
            table,
            stop,
            indices,
            forward,
            inverse,
            white,
            _decode_values,
            native,
            native_codes,
        )
    except Exception as error:
        _enabled = False
        _log.warning("UI image execution uses reference after CPU kernel failure: %s", error)
        return None
    if not np.isfinite(values).all():
        raise FloatingPointError("UI image stages produced nonfinite data")
    return {order[int(i)]: value.reshape(image.shape) for i, value in zip(indices, values)}
