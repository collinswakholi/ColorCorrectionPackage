"""Opt-in, empirically qualified pointwise execution over an owned calibration.

Spatial FFC is always separate. Qualification samples are evidence, not a global
interpolation-error proof. Existing pipeline calls never enable this implicitly.
"""
from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from types import MappingProxyType

import numpy as np

from .colorimetry import D65, delta_e00, rgb_to_xyz, srgb_to_linear, xyz_to_lab
from .results import _owned_mapping, _plain


class AccuracyBudgetError(ValueError):
    """Requested sampled interpolation/backend error budget was not met."""


@dataclass(frozen=True)
class ExecutionOptions:
    mode: str = "auto"
    backend: str = "auto"
    grid_sizes: tuple = (17, 33, 65, 129)
    max_rgb_error: float = .002
    max_delta_e00: float = .5
    validation_samples: int = 4096
    stress_grid_size: int = 67
    random_state: int = 92026
    chunk_pixels: int = 65536
    allow_fallback: bool = True

    def __post_init__(self):
        if self.mode not in ("direct", "auto", "combined-lut") or self.backend not in ("auto", "numpy", "numba"):
            raise ValueError("unsupported execution mode/backend")
        sizes = tuple(self.grid_sizes)
        if not sizes or any(type(value) is not int or not 2 <= value <= 129 for value in sizes) or tuple(sorted(set(sizes))) != sizes:
            raise ValueError("grid_sizes must be distinct increasing integers in 2..129")
        object.__setattr__(self, "grid_sizes", sizes)
        for name in ("max_rgb_error", "max_delta_e00"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if type(self.validation_samples) is not int or not 64 <= self.validation_samples <= 65536:
            raise ValueError("validation_samples must be in 64..65536")
        if type(self.stress_grid_size) is not int or not 3 <= self.stress_grid_size <= 67:
            raise ValueError("stress_grid_size must be an integer in 3..67")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32-1:
            raise ValueError("random_state must be in [0,2**32-1)")
        if type(self.chunk_pixels) is not int or self.chunk_pixels < 1 or type(self.allow_fallback) is not bool:
            raise ValueError("invalid chunk_pixels/allow_fallback")

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("execution options must be ExecutionOptions or a dictionary")

    def to_dict(self):
        result = asdict(self)
        result["grid_sizes"] = list(self.grid_sizes)
        return result


@dataclass(frozen=True)
class AcceleratedResult:
    images: object
    diagnostics: object

    def __post_init__(self):
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))
        object.__setattr__(self, "diagnostics", _owned_mapping(self.diagnostics))

    def to_dict(self):
        return {"images": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}
                            for key, value in self.images.items()}, "diagnostics": _plain(self.diagnostics)}


class LutCache:
    """Byte/entry-bounded shared immutable tables; construction is serialized."""
    def __init__(self, max_bytes=64*1024*1024, max_entries=16):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries < 1:
            raise ValueError("invalid cache capacity")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._lock = threading.RLock()
        self._entries, self._bytes = OrderedDict(), 0
        self._builds = self._hits = 0

    def get_or_build(self, key, builder):
        with self._lock:
            if key in self._entries:
                self._hits += 1
                table, report = self._entries.pop(key)
                self._entries[key] = table, report
                return table, deepcopy(report), True
            self._builds += 1
            table, report = builder()
            if table is not None:
                values = np.ascontiguousarray(table, dtype=np.float64)
                table = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            size = 0 if table is None else table.nbytes
            if size <= self.max_bytes:
                while self._entries and (self._bytes+size > self.max_bytes or len(self._entries) >= self.max_entries):
                    old, _ = self._entries.popitem(last=False)[1]
                    self._bytes -= 0 if old is None else old.nbytes
                self._entries[key] = table, deepcopy(report)
                self._bytes += size
            return table, report, False

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    @property
    def info(self):
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes, "max_bytes": self.max_bytes,
                    "max_entries": self.max_entries, "builds": self._builds, "hits": self._hits}


_DEFAULT_CACHE = LutCache()


def _points(count, seed, *, audit=False):
    rng = np.random.default_rng(seed)
    random = rng.random((count, 3))
    axis = np.linspace(0, 1, 67 if audit else 65)
    edges = []
    for channel in range(3):
        for a in (0., 1.):
            for b in (0., 1.):
                values = np.full((len(axis), 3), a)
                values[:, (channel+1) % 3] = b
                values[:, channel] = axis
                edges.append(values)
    neutral = np.repeat(axis[:, None], 3, axis=1)
    return np.concatenate([random, neutral] + edges)


def _measure(actual, expected, output_encoding):
    if output_encoding == "linear-srgb":
        a, b = srgb_to_linear(actual), srgb_to_linear(expected)
    else:
        a, b = actual, expected
    difference = np.abs(a-b)
    errors = delta_e00(xyz_to_lab(rgb_to_xyz(actual), white=D65), xyz_to_lab(rgb_to_xyz(expected), white=D65))
    return {"samples": len(actual), "max_rgb_error": float(difference.max()),
            "p95_rgb_error": float(np.percentile(difference.max(axis=1), 95)),
            "max_delta_e00": float(errors.max()), "p95_delta_e00": float(np.percentile(errors, 95)),
            "rgb_error_encoding": output_encoding, "delta_e_white": "D65"}


def _stress_points(options):
    size = options.stress_grid_size
    axis = (np.arange(size, dtype=np.float64)+.5)/size
    centers = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    levels = min(17, max(4, size))
    edge = np.arange(levels, dtype=np.float64)/255
    dark = np.stack(np.meshgrid(edge, edge, edge, indexing="ij"), axis=-1).reshape(-1, 3)
    count = min(262144, options.validation_samples*64)
    quantized = np.random.default_rng(options.random_state+2).integers(0, 256, (count, 3), dtype=np.uint8).astype(np.float64)/255
    values = np.concatenate((centers, dark, 1-dark, quantized))
    return values, {"center_grid": size, "center_points": len(centers),
                    "dark_and_white_byte_levels": levels, "dark_and_white_points": 2*len(dark),
                    "independent_random_byte_points": count}


def _within(error, options):
    return error["max_rgb_error"] <= options.max_rgb_error and error["max_delta_e00"] <= options.max_delta_e00


class PreparedExecution:
    """Owned calibration with a reusable direct or qualified combined LUT path."""
    def __init__(self, owner, options=None, *, cache=None):
        from .ui_native import enabled as native_enabled
        if native_enabled(owner.models):
            raise ValueError("native UI models use predict_image/predict_batch; prepared execution cannot change their recorded fusion rules")
        from .bundle import digest_models, snapshot_models
        from .models import MyModels
        from .pipeline import ColorCorrection
        from .core import accel
        self.options = ExecutionOptions.from_value(options)
        self.cache = _DEFAULT_CACHE if cache is None else cache
        if not isinstance(self.cache, LutCache):
            raise TypeError("cache must be LutCache")
        started = time.perf_counter()
        with owner._state_lock:
            models = snapshot_models(owner.models)
            illuminant = deepcopy(owner.REF_ILLUMINANT)
        if models.metadata:
            illuminant = deepcopy(models.metadata.get("legacy_reference_illuminant", D65))
        if illuminant is None:
            illuminant = D65
        if all(getattr(models, "model_"+stage) is None for stage in ("ffc", "gc", "wb", "cc")):
            raise ValueError("prepared execution requires at least one fitted stage")
        self.calibration_id = models.bundle_digest
        self._pipeline = ColorCorrection()
        self._pipeline.models, self._pipeline.REF_ILLUMINANT = models, illuminant
        self.semantics = self._pipeline._active_semantics()
        pointwise = MyModels()
        pointwise.model_gc, pointwise.model_wb, pointwise.model_cc = models.model_gc, models.model_wb, models.model_cc
        if pointwise.model_cc is not None and pointwise.model_cc[2] == "ours":
            original, params, method = pointwise.model_cc
            isolated = copy(original)
            isolated.clear_lut_cache()
            isolated._lut_predict_fn = None
            isolated.use_lut = False
            pointwise.model_cc = isolated, deepcopy(params), method
            original.clear_lut_cache()
            original._lut_predict_fn = None
        # UI predictors require their complete versioned metadata. Retain the
        # validated calibration identity instead of inventing a metadata-free
        # model for this cache key; this conservatively scopes reuse to one fit.
        from .ui_fitting import is_ui
        point_identity = models.bundle_digest if is_ui(models) else digest_models(pointwise)
        policy = {"pointwise": point_identity, "semantics": self.semantics.to_dict(),
                  "reference_illuminant": list(illuminant),
                  "options": self.options.to_dict(), "numpy": np.__version__,
                  "numba": accel.backend_info().get("numba_version"), "engine": "qualified-pointwise-v4-bulk-direct"}
        self.pointwise_id = hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self._condition = threading.Condition(threading.RLock())
        self._active, self._closed = 0, False
        try:
            self._lut, proof, hit = self.cache.get_or_build(self.pointwise_id, self._qualify)
        except BaseException:
            self.close()
            raise
        self._proof = proof
        self._direct_backend = proof["direct_backend_selected"]
        self._preparation = {"seconds": time.perf_counter()-started, "cache_hit": hit,
                             "calibration_id": self.calibration_id, "pointwise_id": self.pointwise_id,
                             "qualification": deepcopy(proof), "cache": self.cache.info}
        if self._lut is None and self.options.mode == "combined-lut" and not self.options.allow_fallback:
            self.close()
            raise AccuracyBudgetError(proof["reason"])

    def _direct(self, rgb, *, backend=None, retain=(), stop="CC"):
        from .core.correction import _predict_direct, predict_image
        from .core.execution import evaluate_direct, evaluate_tone, _tone_downstream_bound
        from .stage_policies import MonotonicToneModel
        import colour
        models = self._pipeline.models
        if backend is not None and models.metadata.get("fitting_policy",{}).get("version") != "ui-compatible-v1":
            from .core.execution import bulk_direct_plan, execute_bulk_direct
            plan = bulk_direct_plan(models, backend=backend, stop=stop)
            if plan is not None:
                try:
                    return execute_bulk_direct(rgb, models, plan, backend=backend, retain=retain,
                                               stop=stop, allow_fallback=self.options.allow_fallback)
                except Exception as error:
                    if not self.options.allow_fallback:
                        raise
                    # An in-place backend may have partially written its private
                    # working result. Recompute from the untouched source, never
                    # feed a partially corrected buffer to another estimator.
                    result, stages, info = self._direct(rgb, backend=None, retain=retain, stop=stop)
                    reason = type(error).__name__+": "+str(error)
                    info["GC_execution"] = {"actual_backend":"reference-direct", "precision":"float64",
                        "requested_backend":backend, "LUT":False, "runtime_fallback":True,
                        "fallback_reason":reason,"runtime_fallback_occurred":True,"fallback_reasons":[reason],
                        "backends_used":["reference-direct"],"calls_by_backend":{"reference-direct":1}}
                    info["pointwise_execution"] = "reference-after-bulk-runtime-failure"
                    return result, stages, info
        from .ui_fitting import is_ui, round_image
        ui_policy = is_ui(models)
        current = round_image(rgb) if ui_policy else np.asarray(rgb, dtype=np.float64)
        shape = current.shape
        flat = current.reshape(-1, 3)
        out = np.empty(flat.shape, dtype=np.float64)
        stages = {key: np.empty(flat.shape, dtype=np.float64) for key in retain if key != stop}
        diagnostics = {"GC": "disabled", "WB": "disabled", "CC": "disabled"}
        usage = {name: {"counts": {}, "runtime": False, "reasons": []} for name in ("GC_execution", "CC")}
        tone_gain = (_tone_downstream_bound(models.model_wb if stop != "GC" else None,
            models.model_cc if stop == "CC" else None) if backend is not None
            and isinstance(models.model_gc, MonotonicToneModel) else None)
        wb = models.model_wb
        wb_gains = (np.diag(wb) if wb is not None and np.array_equal(wb, np.diag(np.diag(wb)))
                    and np.all(np.diag(wb) > 0) else None)
        for start in range(0, len(flat), self.options.chunk_pixels):
            block = flat[start:start+self.options.chunk_pixels]
            if models.model_gc is not None:
                if isinstance(models.model_gc, MonotonicToneModel):
                    if backend is None or tone_gain is None:
                        from .modern import apply_tone
                        block = apply_tone(block.reshape(1, -1, 3), models.model_gc).reshape(-1, 3)
                        diagnostics["GC"] = "explicit-D65-neutral-PCHIP"
                        if backend is not None:
                            diagnostics["GC_execution"] = {"actual_backend": "reference-direct",
                                "fallback_reason": "downstream sensitivity is unbounded or exceeds 32",
                                "precision": "float64", "LUT": False}
                    else:
                        block, diagnostics["GC_execution"] = evaluate_tone(block, models.model_gc,
                            backend=backend, allow_fallback=self.options.allow_fallback)
                        diagnostics["GC"] = "explicit-D65-neutral-PCHIP"
                        diagnostics["GC_execution"]["downstream_rgb_gain_bound"] = tone_gain
                else:
                    block = predict_image(img=block.reshape(1, -1, 3), coeffs=models.model_gc,
                        ref_illuminant=self._pipeline.REF_ILLUMINANT,
                        explicit_color=self.semantics.version == "explicit-v1").reshape(-1, 3)
                    diagnostics["GC"] = "saved-polynomial-stage"
                block = np.clip(block, 0, 1)
            if ui_policy:
                block = round_image(block)
            if "GC" in stages:
                stages["GC"][start:start+len(block)] = block
            if stop != "GC" and models.model_wb is not None:
                if wb_gains is not None and diagnostics.get("GC_execution", {}).get("actual_backend") == "cpu-numba":
                    # A bounded, positive diagonal matrix is an exact channel
                    # scale. Avoid a tiny BLAS multiply beside parallel GC.
                    block = np.clip(block * wb_gains, 0, 1)
                    diagnostics["WB_execution"] = "numpy-channel-gains"
                else:
                    block = np.clip(block @ models.model_wb, 0, 1)
                diagnostics["WB"] = "numpy-diagonal/matrix"
            if ui_policy:
                block = round_image(block)
            if "WB" in stages:
                stages["WB"][start:start+len(block)] = block
            if stop == "CC" and models.model_cc is not None:
                model, params, method = models.model_cc
                if method == "conv":
                    from .conventional import apply_conventional
                    block = apply_conventional(block, model, **params)
                    diagnostics["CC"] = "colour-conventional-direct"
                elif backend is None:
                    block = _predict_direct(block, model)
                    diagnostics["CC"] = "recorded-estimator-direct"
                else:
                    block, diagnostics["CC"] = evaluate_direct(block, model, backend=backend,
                                                               allow_fallback=self.options.allow_fallback)
                if not np.isfinite(block).all():
                    raise FloatingPointError("pointwise direct model produced nonfinite output")
                block = np.clip(block, 0, 1)
            if not np.isfinite(block).all():
                raise FloatingPointError("pointwise stages produced nonfinite output")
            if ui_policy:
                block = round_image(block)
            for name, aggregate in usage.items():
                info = diagnostics.get(name)
                if isinstance(info, dict):
                    actual = info["actual_backend"]
                    aggregate["counts"][actual] = aggregate["counts"].get(actual, 0) + 1
                    aggregate["runtime"] |= info.get("runtime_fallback", False)
                    reason = info.get("fallback_reason")
                    if reason and reason not in aggregate["reasons"] and len(aggregate["reasons"]) < 8:
                        aggregate["reasons"].append(reason)
            out[start:start+len(block)] = block
        for name, aggregate in usage.items():
            if aggregate["counts"]:
                diagnostics[name].update(backends_used=sorted(aggregate["counts"]),
                    calls_by_backend=aggregate["counts"], runtime_fallback_occurred=aggregate["runtime"],
                    fallback_reasons=aggregate["reasons"])
        if "WB_execution" in diagnostics and len(usage["GC_execution"]["counts"]) > 1:
            diagnostics["WB_execution"] = "mixed numpy-channel-gains/matrix"
        result = out.reshape(shape)
        if stop in retain:
            stages[stop] = out
        return result, {key: value.reshape(shape) for key, value in stages.items()}, diagnostics

    def _qualify(self):
        from .core.execution import apply_lut
        from .stage_policies import MonotonicToneModel
        options = self.options
        qualification = _points(options.validation_samples, options.random_state)
        audit = _points(options.validation_samples, options.random_state+1, audit=True)
        proof = {"method": "sampled qualification and separate audit; not all-domain proof",
                 "reason": None, "candidates": [], "selected_mode": "direct",
                 "budgets": {"max_rgb_error": options.max_rgb_error, "max_delta_e00": options.max_delta_e00},
                 "qualification_sha256": hashlib.sha256(qualification.tobytes()).hexdigest(),
                 "audit_sha256": hashlib.sha256(audit.tobytes()).hexdigest()}
        retained = ("GC",) if isinstance(self._pipeline.models.model_gc, MonotonicToneModel) else ()
        expected, expected_stages, _ = self._direct(qualification, retain=retained)
        direct, direct_stages, backend = self._direct(qualification, backend=options.backend, retain=retained)
        proof["direct_backend"] = backend
        proof["direct_backend_error"] = _measure(direct, expected, self.semantics.output_encoding)
        expected_audit, expected_audit_stages, _ = self._direct(audit, retain=retained)
        direct_audit, direct_audit_stages, audit_backend = self._direct(audit, backend=options.backend, retain=retained)
        proof["direct_backend_audit"] = _measure(direct_audit, expected_audit, self.semantics.output_encoding)
        proof["direct_backend_selected"] = options.backend
        direct_keys = ("direct_backend_error", "direct_backend_audit")
        direct_ok = all(_within(proof[key], options) for key in direct_keys)
        if isinstance(self._pipeline.models.model_gc, MonotonicToneModel):
            # Direct tone fusion changes only floating-point evaluation, not the
            # model. Loose interpolation budgets must never permit material
            # direct error after a sensitive downstream CC model amplifies it.
            proof["direct_roundoff_limits"] = {"max_rgb_error": 5e-12, "max_delta_e00": 1e-8}
            direct_ok = direct_ok and all(proof[key]["max_rgb_error"] <= 5e-12
                and proof[key]["max_delta_e00"] <= 1e-8 for key in direct_keys)
            proof["direct_GC_error"] = {
                "qualification_max_encoded_rgb_error": float(np.max(np.abs(direct_stages["GC"]-expected_stages["GC"]))),
                "audit_max_encoded_rgb_error": float(np.max(np.abs(direct_audit_stages["GC"]-expected_audit_stages["GC"]))),
                "max_encoded_rgb_limit": 5e-12}
            direct_ok = direct_ok and proof["direct_GC_error"]["qualification_max_encoded_rgb_error"] <= 5e-12
            direct_ok = direct_ok and proof["direct_GC_error"]["audit_max_encoded_rgb_error"] <= 5e-12
        runtime_failed = any(isinstance(info.get(stage), dict) and info[stage].get("runtime_fallback_occurred", False)
                             for info in (backend, audit_backend) for stage in ("GC_execution", "CC"))
        if not direct_ok or runtime_failed:
            if not options.allow_fallback:
                raise AccuracyBudgetError("direct backend exceeds sampled error budget")
            proof["direct_backend_selected"] = None
            proof["direct_backend_fallback"] = ("direct backend had a runtime failure during qualification; recorded reference pinned"
                if runtime_failed else "recorded direct reference; requested CPU backend exceeded budget")
        if options.mode == "direct":
            proof["reason"] = "direct execution requested"
            return None, proof
        for size in options.grid_sizes:
            axis = np.linspace(0, 1, size)
            grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
            values, _, _ = self._direct(grid.reshape(-1, 3))
            table = values.reshape(size, size, size, 3)
            actual, backend = apply_lut(qualification, table, backend=options.backend, allow_fallback=options.allow_fallback)
            actual = np.clip(actual, 0, 1)
            error = _measure(actual, expected, self.semantics.output_encoding)
            proof["candidates"].append({"grid_size": size, "error": error, "application_backend": backend})
            if _within(error, options):
                actual_audit, audit_backend = apply_lut(audit, table, backend=options.backend, allow_fallback=options.allow_fallback)
                actual_audit = np.clip(actual_audit, 0, 1)
                proof["audit"] = _measure(actual_audit, expected_audit, self.semantics.output_encoding)
                if _within(proof["audit"], options):
                    # Centers alone can miss near-black subcell transitions.
                    # Add byte-lattice endpoints and independent quantized RGB.
                    # This is still empirical, never an all-domain error bound.
                    stress, components = _stress_points(options)
                    expected_stress, _, _ = self._direct(stress)
                    actual_stress, _ = apply_lut(stress, table, backend=options.backend, allow_fallback=options.allow_fallback)
                    proof["stress_grid_size"] = options.stress_grid_size
                    proof["stress_components"] = components
                    proof["stress_sha256"] = hashlib.sha256(stress.tobytes()).hexdigest()
                    proof["stress"] = _measure(np.clip(actual_stress, 0, 1), expected_stress, self.semantics.output_encoding)
                    if not _within(proof["stress"], options):
                        proof["reason"] = "selected candidate failed dense stress audit; using direct without retuning on audit data"
                        return None, proof
                    proof.update(selected_mode="combined-lut", grid_size=size, application_backend=audit_backend)
                    return table, proof
                proof["reason"] = "selected candidate failed separate audit; using direct without retuning on audit data"
                return None, proof
        proof["reason"] = "no candidate meets sampled interpolation budgets"
        return None, proof

    @property
    def preparation(self):
        return deepcopy(self._preparation)

    def predict_image(self, image, *, stages="final"):
        from .core.accel import apply_ffc_float
        from .core.execution import apply_lut
        from .prediction import normalize_stages
        selected = normalize_stages(stages)
        if self._lut is not None and selected != ("CC",):
            raise ValueError("combined LUT execution requires stages='final'; use direct execution for intermediates")
        with self._condition:
            if self._closed:
                raise RuntimeError("prepared execution is closed")
            self._active += 1
        try:
            started = time.perf_counter()
            pipeline = self._pipeline
            data, current = pipeline._prepare_input(image, self.semantics, pipeline.models.metadata.get("image_options", {}),
                                                    _borrow_readonly=selected == ("CC",))
            from .ui_fitting import is_ui, round_image
            ui_policy = is_ui(pipeline.models)
            if ui_policy:
                current = round_image(current)
            source = data.to_metadata()
            del data
            decode = time.perf_counter()
            if pipeline.models.model_ffc is not None:
                if self.semantics.version == "explicit-v1" and current.shape[:2] != pipeline.models.model_ffc.shape:
                    raise ValueError("FFC requires the calibrated image dimensions")
                if ui_policy:
                    from .ui_stages import apply_ffc
                    current = apply_ffc(current, pipeline.models.model_ffc)
                else:
                    current = apply_ffc_float(current, pipeline.models.model_ffc)
            if ui_policy:
                current = round_image(current)
            ffc = time.perf_counter()
            outputs = {"FFC": current} if "FFC" in selected else {}
            if self._lut is not None:
                current, backend = apply_lut(current, self._lut, backend=self.options.backend, allow_fallback=self.options.allow_fallback)
                current = np.clip(current, 0, 1)
                outputs["CC"] = current
            elif selected != ("FFC",):
                retained_stages = tuple(key for key in selected if key != "FFC")
                current, retained, backend = self._direct(current, backend=self._direct_backend, retain=retained_stages, stop=retained_stages[-1])
                outputs.update(retained)
            outputs = pipeline._convert_outputs(outputs)
            if any(not np.isfinite(value).all() for value in outputs.values()):
                raise FloatingPointError("prepared execution produced nonfinite output")
            report = {"calibration_id": self.calibration_id, "pointwise_id": self.pointwise_id,
                      "requested": self.options.to_dict(), "selected_mode": self._proof["selected_mode"],
                      "backend": backend if selected != ("FFC",) else "spatial-FFC-only",
                      "qualification": self._proof, "input_image": source,
                      "seconds": {"decode": decode-started, "FFC": ffc-decode, "pointwise_and_output": time.perf_counter()-ffc,
                                  "total": time.perf_counter()-started},
                      "FFC_spatial": pipeline.models.model_ffc is not None}
            return AcceleratedResult(outputs, report)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    apply = predict_image

    def export_cube(self, path, *, exclude_ffc=False, working_space=False, title="ColorCorrectionPipeline GC WB CC"):
        with self._condition:
            if self._closed:
                raise RuntimeError("prepared execution is closed")
            self._active += 1
        try:
            return self._export_cube(path, exclude_ffc=exclude_ffc, working_space=working_space, title=title)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _export_cube(self, path, *, exclude_ffc, working_space, title):
        if self._lut is None:
            raise AccuracyBudgetError("cube export requires a qualified combined LUT; prepare with a suitable candidate grid/budget")
        if self._pipeline.models.model_ffc is not None and exclude_ffc is not True:
            raise ValueError("a .cube cannot encode spatial FFC; explicitly choose exclude_ffc=True")
        if (self.semantics.input_encoding != "srgb" or self.semantics.output_encoding != "srgb") and working_space is not True:
            raise ValueError("cube uses working encoded sRGB; explicitly choose working_space=True and handle declared boundary conversions separately")
        if not isinstance(title, str) or any(character in title for character in '\r\n"') or not title.isascii():
            raise ValueError("cube title must be single-line ASCII without quotes")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = self._lut.shape[0]
        lines = ['TITLE "'+title+'"', '# Input/output: encoded sRGB, domain [0,1].',
                 '# Spatial FFC is excluded; apply it separately before this cube.',
                 '# Pointwise identity: '+self.pointwise_id, 'LUT_3D_SIZE '+str(size),
                 'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0']
        temporary = None
        digest, byte_count = hashlib.sha256(), 0
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix="."+destination.name+".", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                def write(line):
                    nonlocal byte_count
                    payload = (line+'\n').encode('ascii')
                    file.write(payload)
                    digest.update(payload)
                    byte_count += len(payload)
                for line in lines:
                    write(line)
                # .cube rows use red as the fastest varying input coordinate.
                for blue in range(size):
                    for green in range(size):
                        for red in range(size):
                            write(' '.join(format(float(value), '.17g') for value in self._lut[red, green, blue]))
                file.flush(); os.fsync(file.fileno())
            check, read_bytes = hashlib.sha256(), 0
            with temporary.open('rb') as file:
                for block in iter(lambda: file.read(1024*1024), b''):
                    check.update(block)
                    read_bytes += len(block)
            if read_bytes != byte_count or check.digest() != digest.digest():
                raise OSError("cube readback validation failed")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"path": str(destination), "sha256": digest.hexdigest(), "bytes": byte_count,
                "grid_size": size, "domain": "encoded-sRGB", "FFC_excluded": True, "pointwise_id": self.pointwise_id,
                "qualification": deepcopy(self._proof)}

    def close(self):
        with self._condition:
            self._closed = True
            while self._active:
                self._condition.wait()
            self._pipeline = None
            self._lut = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
