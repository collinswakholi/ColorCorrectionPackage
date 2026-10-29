"""Explicit CPU execution helpers; legacy inference does not opt in implicitly.

Fused polynomial evaluation uses the recorded portable/reproducible CPU feature
contract. Fastmath is disabled. Diagnostics describe the backend actually used.
"""
import numpy as np

from . import accel
from ..stage_policies import MonotonicToneModel as _ToneModel

_TONE_PREDICT = _ToneModel.predict
_STANDARD_TONE = None
_STANDARD_AFFINE = None


def _tone_parameters(model):
    """Derive runtime-only PCHIP coefficients from known, unmodified models."""
    from scipy.interpolate import PchipInterpolator
    if (type(model) is not _ToneModel or model.version != "neutral-pchip-v1"
            or "predict" in vars(model) or _ToneModel.predict is not _TONE_PREDICT
            or _TONE_PREDICT.__module__ != "ColorCorrectionPipeline.stage_policies"
            or _TONE_PREDICT.__qualname__ != "MonotonicToneModel.predict"):
        return None
    curve = PchipInterpolator(model.measured_knots, model.target_knots, extrapolate=False)
    coefficients = np.ascontiguousarray(curve.c, dtype=np.float64)
    widths = np.diff(curve.x)
    # Very narrow or steep curves can amplify tiny platform-dependent libm
    # differences. Keep those valid but ill-conditioned models on reference.
    if (not np.isfinite(coefficients).all() or np.min(widths) < 1e-3
            or np.max(np.diff(model.target_knots) / widths) > 100.0):
        return None
    return np.ascontiguousarray(curve.x, dtype=np.float64), coefficients


def _tone_downstream_bound(wb, cc):
    """Conservatively bound gain before allowing tone-roundoff downstream.

    Unknown/discontinuous estimators and ill-conditioned fits retain the exact
    reference GC input. Existing estimator execution is otherwise unchanged.
    Clipping is non-expansive; each monomial derivative on [0,1] is at most its
    exponent. Bias/centering do not contribute to the derivative.
    """
    gain = 1.0 if wb is None else float(np.max(np.sum(np.abs(wb), axis=0)))
    if cc is not None:
        model, _, method = cc
        policy = bool(getattr(model, "portable_inference", False)) or getattr(model, "fitting_version", None) == "reproducible-v1"
        parameters = _affine(model.model, model.degree) if method == "ours" and policy else None
        if parameters is None:
            return None
        powers, coefficients, _, _, scale = parameters
        gain *= float(np.max(np.sum(np.abs(coefficients / scale) * powers.sum(axis=1), axis=1)))
    return gain if np.isfinite(gain) and gain <= 32.0 else None


if accel.HAS_NUMBA:
    _tone_prange = accel.nb_prange
    @accel.nb_njit(cache=True, fastmath=False, inline="always")
    def _tone_lightness(value, knots, coefficients):
        if value <= 0.0:
            return 0.0
        if value >= 100.0:
            return 100.0
        low, high = 0, knots.size - 2
        while low < high:
            middle = (low + high + 1) // 2
            if value < knots[middle]:
                high = middle - 1
            else:
                low = middle
        offset = value - knots[low]
        # PPoly evaluates ascending powers, not Horner's rule. Preserve its
        # addition/multiplication order, endpoint constraints and clipping.
        power, result = 1.0, 0.0
        for degree in range(4):
            result = result + coefficients[3-degree, low] * power
            if degree < 3:
                power *= offset
        if np.isnan(result):
            return result
        return min(100.0, max(0.0, result))

    @accel.nb_njit(cache=True, fastmath=False, inline="always")
    def _decode_srgb(value):
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    @accel.nb_njit(cache=True, fastmath=False, inline="always")
    def _encode_srgb(value):
        if not np.isfinite(value):
            return np.nan
        absolute = abs(value)
        if absolute <= 0.0031308:
            encoded = value * 12.92
        else:
            encoded = 1.055 * absolute ** (1.0 / 2.4) - 0.055
            if value < 0.0:
                encoded = -encoded
        return min(1.0, max(0.0, encoded))

    @accel.nb_njit(parallel=True, cache=True, fastmath=False)
    def _fused_tone(flat, knots, coefficients, forward, inverse, white):
        output = np.empty(flat.shape, dtype=np.float64)
        epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
        for row in _tone_prange(flat.shape[0]):
            r = _decode_srgb(flat[row, 0])
            g = _decode_srgb(flat[row, 1])
            b = _decode_srgb(flat[row, 2])
            x = (r*forward[0, 0] + g*forward[0, 1] + b*forward[0, 2]) / white[0]
            y = (r*forward[1, 0] + g*forward[1, 1] + b*forward[1, 2]) / white[1]
            z = (r*forward[2, 0] + g*forward[2, 1] + b*forward[2, 2]) / white[2]
            fx = np.cbrt(x) if x > epsilon else (kappa*x + 16.0) / 116.0
            fy = np.cbrt(y) if y > epsilon else (kappa*y + 16.0) / 116.0
            fz = np.cbrt(z) if z > epsilon else (kappa*z + 16.0) / 116.0
            lightness = _tone_lightness(116.0*fy - 16.0, knots, coefficients)
            a, bb = 500.0*(fx-fy) / 500.0, 200.0*(fy-fz) / 200.0
            scaled = lightness / 116.0
            fy = scaled + 16.0 / 116.0
            fx, fz = fy + a, fy - bb
            x3, y3, z3 = fx**3, fy**3, fz**3
            x = x3 if x3 > epsilon else (scaled+a)*116.0/kappa
            y = y3 if lightness > kappa*epsilon else lightness/kappa
            z = z3 if z3 > epsilon else (scaled-bb)*116.0/kappa
            x, y, z = x*white[0], y*white[1], z*white[2]
            output[row, 0] = _encode_srgb(x*inverse[0, 0] + y*inverse[0, 1] + z*inverse[0, 2])
            output[row, 1] = _encode_srgb(x*inverse[1, 0] + y*inverse[1, 1] + z*inverse[1, 2])
            output[row, 2] = _encode_srgb(x*inverse[2, 0] + y*inverse[2, 1] + z*inverse[2, 2])
        return output


def evaluate_tone(rgb, model, *, backend="auto", allow_fallback=True):
    """Direct float64 D65 neutral PCHIP, with reference fallback and no LUT."""
    from ..modern import apply_tone
    from ..colorimetry import _RGB_TO_XYZ, _XYZ_TO_RGB, D65, white_xyz
    values, clipped = _rgb(rgb, "reject")
    actual, reason = _backend(backend, allow_fallback)
    runtime_fallback = False
    parameters = _tone_parameters(model) if actual == "numba" else None
    if actual == "numba" and parameters is not None:
        try:
            output = _fused_tone(values.reshape(-1, 3), *parameters,
                                  _RGB_TO_XYZ, _XYZ_TO_RGB, white_xyz(D65)).reshape(values.shape)
        except Exception as error:
            if not allow_fallback:
                raise
            actual, reason = "numpy", type(error).__name__ + ": " + str(error)
            runtime_fallback = True
    else:
        unsupported = actual == "numba" and parameters is None
        if unsupported and backend == "numba" and not allow_fallback:
            raise ValueError("this tone model has no well-conditioned fused direct adapter")
        actual = "numpy"
        if unsupported:
            reason = "unsupported, overridden, or ill-conditioned tone model"
    if actual == "numpy":
        output = apply_tone(values.reshape(1, -1, 3), model).reshape(values.shape)
    if output.shape != values.shape or not np.isfinite(output).all():
        raise FloatingPointError("direct tone evaluation produced invalid/nonfinite output")
    return output, {"actual_backend": "cpu-" + actual, "requested_backend": backend,
                    "fallback_reason": reason, "precision": "float64", "LUT": False,
                    "runtime_fallback": runtime_fallback,
                    "method": "fused-D65-neutral-PCHIP" if actual == "numba" else "reference-D65-neutral-PCHIP"}


def _rgb(value, domain):
    array = np.asarray(value)
    if array.ndim not in (2, 3) or array.shape[-1] != 3 or not array.size:
        raise ValueError("RGB must have nonempty (N,3) or (H,W,3) shape")
    if array.dtype.kind not in "fiu":
        raise ValueError("RGB must contain finite numeric values")
    low, high = array.min(), array.max()
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("RGB must contain finite numeric values")
    if domain not in ("reject", "clip"):
        raise ValueError("domain must be reject or clip")
    outside = bool(low < 0 or high > 1)
    if outside and domain == "reject":
        raise ValueError("encoded RGB lies outside [0,1]")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if outside:
        array = np.clip(array, 0, 1)
    return array, outside


def _backend(requested, allow_fallback):
    if requested not in ("auto", "numpy", "numba"):
        raise ValueError("backend must be auto, numpy or numba")
    if type(allow_fallback) is not bool:
        raise TypeError("allow_fallback must be boolean")
    if requested == "numpy":
        return "numpy", None
    if accel.HAS_NUMBA:
        return "numba", None
    if requested == "numba" and not allow_fallback:
        raise RuntimeError("Numba was requested but is unavailable")
    return "numpy", "Numba unavailable" if requested == "numba" else None


def _numpy_lut(flat, lut, size, chunk=65536):
    output = np.empty(flat.shape, dtype=np.float64)
    step = 1.0 / (size - 1)
    for start in range(0, len(flat), chunk):
        coordinates = flat[start:start+chunk] / step
        lower = np.minimum(coordinates.astype(np.int64), size - 2)
        fraction = coordinates - lower
        r, g, b = lower.T
        dr, dg, db = fraction.T
        dr, dg, db = dr[:, None], dg[:, None], db[:, None]
        c00 = lut[r,g,b]*(1-dr) + lut[r+1,g,b]*dr
        c01 = lut[r,g,b+1]*(1-dr) + lut[r+1,g,b+1]*dr
        c10 = lut[r,g+1,b]*(1-dr) + lut[r+1,g+1,b]*dr
        c11 = lut[r,g+1,b+1]*(1-dr) + lut[r+1,g+1,b+1]*dr
        output[start:start+len(coordinates)] = (c00*(1-dg)+c10*dg)*(1-db) + (c01*(1-dg)+c11*dg)*db
    return output


def apply_lut(rgb, lut, *, backend="auto", domain="reject", allow_fallback=True):
    """Apply the same clamped/rejected-domain trilinear contract on both CPUs."""
    values, clipped = _rgb(rgb, domain)
    table = np.asarray(lut)
    if table.ndim != 4 or table.shape[-1] != 3 or not 2 <= table.shape[0] <= 129 or table.shape[:3] != (table.shape[0],)*3:
        raise ValueError("LUT must have shape (G,G,G,3), with G in 2..129")
    if table.dtype.kind != "f" or not np.isfinite(table).all():
        raise ValueError("LUT must contain finite floating values")
    table = np.ascontiguousarray(table, dtype=np.float64)
    actual, reason = _backend(backend, allow_fallback)
    flat = values.reshape(-1, 3)
    if actual == "numba":
        try:
            output = accel._trilinear_interp(flat, table, table.shape[0])
        except Exception as exc:
            if not allow_fallback:
                raise
            actual, reason = "numpy", type(exc).__name__+": "+str(exc)
    if actual == "numpy":
        output = _numpy_lut(flat, table, table.shape[0])
    if not np.isfinite(output).all():
        raise FloatingPointError("LUT evaluation produced nonfinite values")
    return output.reshape(values.shape), {"requested_backend": backend, "actual_backend": "cpu-"+actual,
        "fallback_reason": reason, "precision": "float64", "domain_policy": domain,
        "input_was_clipped": clipped, "method": "trilinear", "grid_size": table.shape[0]}


def _affine(model, degree):
    from sklearn.linear_model import LinearRegression
    from ..portable_inference import AffinePredictor
    from sklearn.preprocessing import PolynomialFeatures
    if type(model) not in (LinearRegression, AffinePredictor) or any(name in vars(model) for name in ("predict", "_decision_function", "_forward_pass_fast")):
        return None
    if type(degree) is not int or not 0 <= degree <= 8:
        return None
    powers = PolynomialFeatures(degree=degree, include_bias=True).fit(np.zeros((1,3))).powers_.astype(np.int64)
    count = len(powers)
    if type(model) is LinearRegression and "predict" not in vars(model):
        coefficients = np.asarray(model.coef_)
        intercept = np.asarray(model.intercept_)
        center = np.zeros(count)
        scale = np.ones(count)
    elif type(model) is AffinePredictor and "predict" not in vars(model):
        coefficients = np.asarray(model.coefficients)
        intercept = np.asarray(model.intercept)
        center = np.zeros(count) if model.center is None else np.asarray(model.center)
        scale = np.ones(count) if model.scale is None else np.asarray(model.scale)
    else:
        return None
    if intercept.ndim == 0:
        intercept = np.full(3, intercept, dtype=np.float64)
    if coefficients.shape != (3,count) or intercept.shape != (3,) or center.shape != (count,) or scale.shape != (count,):
        raise ValueError("affine parameters contradict polynomial feature dimensions")
    if any(value.dtype.kind not in "fiu" for value in (coefficients,intercept,center,scale)):
        raise ValueError("affine parameters must contain real numeric values")
    if not all(np.isfinite(value).all() for value in (coefficients,intercept,center,scale)) or np.any(scale <= 0):
        raise ValueError("affine parameters must be finite with positive scale")
    return (powers, np.ascontiguousarray(coefficients,dtype=np.float64), np.ascontiguousarray(intercept,dtype=np.float64),
            np.ascontiguousarray(center,dtype=np.float64), np.ascontiguousarray(scale,dtype=np.float64))


if accel.HAS_NUMBA:
    _prange = accel.nb_prange
    @accel.nb_njit(parallel=True, cache=True, fastmath=False)
    def _fused_affine(flat, powers, coefficients, intercept, center, scale):
        output = np.empty((flat.shape[0], 3), dtype=np.float64)
        for row in _prange(flat.shape[0]):
            a, b, c = intercept[0], intercept[1], intercept[2]
            for term in range(powers.shape[0]):
                value = 1.0
                for channel in range(3):
                    for exponent in range(powers[term,channel]):
                        value *= flat[row,channel]
                value = (value-center[term])/scale[term]
                a += value*coefficients[0,term]
                b += value*coefficients[1,term]
                c += value*coefficients[2,term]
            output[row,0],output[row,1],output[row,2] = a,b,c
        return output

    @accel.nb_njit(parallel=True, cache=True, fastmath=False)
    def _affine_inplace(flat, powers, coefficients, intercept, center, scale):
        # Each row is read completely before its three destination writes.
        # Rows are independent; only the private GC working buffer is mutable.
        for row in _prange(flat.shape[0]):
            a, b, c = intercept[0], intercept[1], intercept[2]
            for term in range(powers.shape[0]):
                value = 1.0
                for channel in range(3):
                    for exponent in range(powers[term,channel]):
                        value *= flat[row,channel]
                value = (value-center[term])/scale[term]
                a += value*coefficients[0,term]
                b += value*coefficients[1,term]
                c += value*coefficients[2,term]
            flat[row,0],flat[row,1],flat[row,2] = a,b,c

    _STANDARD_TONE = _fused_tone
    _STANDARD_AFFINE = _fused_affine


def bulk_direct_plan(models, *, backend, stop):
    """Admit only pure pointwise stages; preserve extension/override routes."""
    if backend not in ("auto", "numba") or not accel.HAS_NUMBA:
        return None
    if (_STANDARD_TONE is None or globals().get("_fused_tone") is not _STANDARD_TONE
            or globals().get("_fused_affine") is not _STANDARD_AFFINE):
        return None
    if _tone_parameters(models.model_gc) is None:
        return None
    wb = models.model_wb if stop != "GC" else None
    if wb is not None and (wb.shape != (3,3) or not np.array_equal(wb,np.diag(np.diag(wb))) or np.any(np.diag(wb) <= 0)):
        return None
    cc = models.model_cc if stop == "CC" else None
    gain = _tone_downstream_bound(wb, cc)
    if gain is None:
        return None
    parameters = None if cc is None else _affine(cc[0].model, cc[0].degree)
    return {"wb_gains": None if wb is None else np.diag(wb), "affine": parameters, "gain": gain}


def execute_bulk_direct(rgb, models, plan, *, backend, retain, stop, allow_fallback):
    """No extra full frame beside requested outputs; no serialized runtime plan."""
    result, tone = evaluate_tone(rgb, models.model_gc, backend=backend, allow_fallback=allow_fallback)
    tone.update(downstream_rgb_gain_bound=plan["gain"],backends_used=[tone["actual_backend"]],
                calls_by_backend={tone["actual_backend"]:1},runtime_fallback_occurred=tone["runtime_fallback"],
                fallback_reasons=[] if tone["fallback_reason"] is None else [tone["fallback_reason"]])
    stages = {}
    info = {"GC":"explicit-D65-neutral-PCHIP", "GC_execution":tone, "WB":"disabled", "CC":"disabled",
            "pointwise_execution":"owned-buffer-bulk-direct"}
    if "GC" in retain and stop != "GC": stages["GC"] = result.copy()
    if stop != "GC" and plan["wb_gains"] is not None:
        np.multiply(result, plan["wb_gains"], out=result)
        np.clip(result,0,1,out=result)
        info.update(WB="numpy-diagonal/matrix",WB_execution="numpy-channel-gains")
    if "WB" in retain and stop != "WB": stages["WB"] = result.copy()
    if stop == "CC" and plan["affine"] is not None:
        _affine_inplace(result.reshape(-1,3),*plan["affine"])
        if not np.isfinite(result).all():
            raise FloatingPointError("pointwise direct model produced nonfinite output")
        np.clip(result,0,1,out=result)
        info["CC"] = {"actual_backend":"cpu-numba", "requested_backend":backend, "precision":"float64",
            "method":"owned-buffer-polynomial-affine", "fallback_reason":None,"runtime_fallback":False,
            "domain_policy":"reject", "input_was_clipped":False,
            "feature_policy":"portable/reproducible CPU float64", "polynomial_degree":models.model_cc[0].degree,
            "feature_count":len(plan["affine"][0]), "runtime_fallback_occurred":False,
            "backends_used":["cpu-numba"],"calls_by_backend":{"cpu-numba":1},"fallback_reasons":[]}
    if not np.isfinite(result).all():
        raise FloatingPointError("pointwise stages produced nonfinite output")
    if stop in retain: stages[stop] = result
    return result, stages, info


def evaluate_direct(rgb, model, *, backend="auto", domain="reject", allow_fallback=True):
    """Opt-in fused affine evaluation; unsupported models retain exact reference.

    No LUT or output clipping is applied. Unknown estimators are evaluated in one
    reference call because their prediction may depend on the whole batch.
    """
    from .correction import _predict_direct
    from sklearn.preprocessing import PolynomialFeatures
    values, clipped = _rgb(rgb, domain)
    actual, reason = _backend(backend, allow_fallback)
    runtime_fallback = False
    policy = bool(getattr(model,"portable_inference",False)) or getattr(model,"fitting_version",None) == "reproducible-v1"
    parameters = _affine(model.model, model.degree) if policy else None
    diagnostic = {"requested_backend": backend, "precision": "float64", "domain_policy": domain,
                  "input_was_clipped": clipped, "feature_policy": "portable/reproducible CPU float64" if policy else "legacy runtime-defined"}
    if parameters is None:
        if backend in ("numpy","numba") and not allow_fallback:
            raise ValueError("this estimator/feature policy has no fused NumPy/Numba adapter")
        # Preserve the caller dtype for legacy feature generation. Validation's
        # float64 working buffer must not change reference numerical behavior.
        reference_values = np.asarray(rgb)
        if clipped:
            reference_values = np.clip(reference_values,0,1)
        output = _predict_direct(reference_values, model)
        diagnostic.update(actual_backend="reference-direct", fallback_reason="unsupported estimator or legacy feature policy",
                          precision="reference estimator dtype", output_dtype=str(output.dtype), method="reference")
    else:
        flat = values.reshape(-1,3)
        if actual == "numba":
            try:
                output = _fused_affine(flat, *parameters)
            except Exception as exc:
                if not allow_fallback:
                    raise
                actual = "numpy"
                reason = type(exc).__name__+": "+str(exc)
                runtime_fallback = True
        if actual == "numpy":
            _, coefficients, intercept, center, scale = parameters
            transformer = PolynomialFeatures(degree=model.degree,include_bias=True).fit(np.zeros((1,3)))
            output = np.empty(flat.shape,dtype=np.float64)
            for start in range(0,len(flat),65536):
                features = transformer.transform(flat[start:start+65536])
                features -= center
                features /= scale
                output[start:start+len(features)] = features @ coefficients.T + intercept
        output = output.reshape(values.shape)
        diagnostic.update(actual_backend="cpu-"+actual, fallback_reason=reason,
                          runtime_fallback=runtime_fallback,
                          method="fused-polynomial-affine" if actual == "numba" else "chunked-polynomial-affine",
                          polynomial_degree=model.degree, feature_count=len(parameters[0]))
    if output.shape != values.shape or not np.isfinite(output).all():
        raise FloatingPointError("direct evaluation produced invalid/nonfinite output")
    return output, diagnostic
