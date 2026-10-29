"""Bounded prediction orchestration; legacy list prediction stays available."""

import copy
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np

from .contracts import ImageData


def normalize_stages(stages=None):
    order = ("FFC", "GC", "WB", "CC")
    if stages is None or isinstance(stages, str) and stages == "all":
        return order
    if isinstance(stages, str):
        stages = ("CC",) if stages == "final" else (stages,)
    try:
        selected = tuple(stages)
    except TypeError as exc:
        raise TypeError("stages must be all, final, a stage name, or a sequence") from exc
    if not selected or any(not isinstance(key, str) or key not in order for key in selected) or len(set(selected)) != len(selected):
        raise ValueError("stages must contain distinct names from FFC, GC, WB, CC")
    return tuple(key for key in order if key in selected)


def _identity(item, index):
    if isinstance(item, (str, os.PathLike)):
        return {"kind": "file", "path": os.fspath(item), "index": index}
    if isinstance(item, ImageData):
        return {"kind": "ImageData", "digest": item.digest, "index": index}
    if isinstance(item, np.ndarray):
        return {"kind": "ndarray", "shape": list(item.shape), "dtype": str(item.dtype), "index": index}
    return {"kind": type(item).__name__, "index": index}


def chunked_cc(image, cc, chunk_pixels):
    """Bound pointwise feature workspace while selecting LUT/direct once per image."""
    from .core.correction import _ensure_lut, _predict_direct
    from .core.accel import trilinear_lut_apply
    import colour
    model, options, method = cc
    flat = image.reshape(-1, 3)
    use_lut = method == "ours" and len(flat) >= model.lut_min_pixels and _ensure_lut(model)
    output = None
    for start in range(0, len(flat), chunk_pixels):
        block = flat[start:start+chunk_pixels]
        if method == "conv":
            from .conventional import apply_conventional
            predicted = apply_conventional(block, model, **options)
        elif method == "ours":
            predicted = (trilinear_lut_apply(block, model._lut, model._lut_grid_size) if use_lut
                         else _predict_direct(block, model))
        else:
            raise ValueError("unsupported saved CC method")
        if output is None:
            output = np.empty(flat.shape, dtype=predicted.dtype)
        output[start:start+len(block)] = predicted
    return output.reshape(image.shape)


def _prepare(item, index):
    if isinstance(item, np.ndarray):
        return np.array(item, copy=True)
    if isinstance(item, ImageData):
        return item
    if isinstance(item, (str, os.PathLike)):
        return Path(item)
    raise TypeError("batch inputs must be paths, ImageData, or numpy arrays")


class PredictionStream:
    """Own one batch iterator, its cancellation token and optional summary file.

    Closing waits for the bounded running tasks. It cannot interrupt native
    kernels. A streamed report retains aggregate outcomes, not image arrays.
    File mode additionally saves a manifest beside each completed item's files.
    """

    def __init__(self, stream, event, *, setup, output_directory=None, report_path=None, release=None):
        self._stream = stream
        self._event = event
        self._setup = copy.deepcopy(setup)
        self.output_directory = output_directory
        self.report_path = Path(report_path) if report_path is not None else None
        self._report = {"status": "pending" if self.report_path is not None else "not_requested"}
        self._finished = False
        self._release = release

    @property
    def summary(self):
        return dict(self._stream.summary, setup=copy.deepcopy(self._setup),
                    output_directory=str(self.output_directory) if self.output_directory is not None else None,
                    report=copy.deepcopy(self._report))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        except BaseException:
            self.close()
            raise

    def cancel(self):
        self._event.set()

    def _finalize(self):
        if self._finished:
            return
        self._finished = True
        release, self._release = self._release, None
        if release is not None:
            release()
        if self.report_path is not None:
            from .reporting import save_report
            self._report = {"status": "written", "path": str(self.report_path)}
            try:
                save_report(self.report_path, self.summary)
            except Exception as exc:
                self._report = {"status": "failed", "path": str(self.report_path),
                                "type": type(exc).__name__, "message": str(exc)}

    def close(self):
        try:
            self._stream.close()
        finally:
            self._finalize()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def iter_predict(owner, inputs, *, stages="final", max_workers=None, max_in_flight=None,
                 ordered=False, on_error="continue", cancel_event=None, on_progress=None,
                 callback_error="record", output_dir=None, output_ext=".tiff",
                 export_options=None, report_path=None, allow_identity=False, bundle_limits=None,
                 chunk_pixels=65536):
    from .batch import execute_batch, BatchItemResult
    from .bundle import snapshot_models, digest_models
    from .io import write_image_data
    from .reporting import save_report

    selected = normalize_stages(stages)
    if type(chunk_pixels) is not int or chunk_pixels < 1:
        raise ValueError("chunk_pixels must be a positive integer")
    if type(allow_identity) is not bool:
        raise TypeError("allow_identity must be boolean")
    options = copy.deepcopy({} if export_options is None else export_options)
    if not isinstance(options, dict) or set(options) - {"dtype", "output_encoding", "allow_precision_loss", "quality", "png_profile"}:
        raise ValueError("unsupported batch export_options")
    if output_dir is None and options:
        raise ValueError("export_options require output_dir")
    if output_ext not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        raise ValueError("output_ext must be .tif, .tiff, .png, .jpg or .jpeg")
    event = cancel_event if cancel_event is not None else threading.Event()
    if not callable(getattr(event, "is_set", None)) or not callable(getattr(event, "set", None)):
        raise TypeError("cancel_event must provide is_set and set")
    started = time.perf_counter()
    with owner._state_lock:
        if not owner.models.has_models() and not owner.models.metadata and not allow_identity:
            raise ValueError("batch prediction requires calibration; use allow_identity=True explicitly")
        snapshot = copy.copy(owner)
        snapshot.models = snapshot_models(owner.models, limits=bundle_limits)
        snapshot.REF_ILLUMINANT = copy.deepcopy(owner.REF_ILLUMINANT)
    lut_report = {"requested": False}
    if snapshot.models.model_cc is not None and snapshot.models.model_cc[2] == "ours":
        model = snapshot.models.model_cc[0]
        if model.use_lut and not event.is_set():
            from .core.correction import _ensure_lut
            lut_report = {"requested": True, "ready": bool(_ensure_lut(model)),
                          "policy": "prepare once on the owned snapshot before parallel workers"}
            if not lut_report["ready"]:
                model.use_lut = False
                model._lut_predict_fn = None
                lut_report["fallback"] = "direct estimator; LUT preparation failed"
    calibration_id = digest_models(snapshot.models, limits=bundle_limits)
    semantics = snapshot._active_semantics()
    run_directory = None

    def prepare(item, index):
        pixels = item.pixels if isinstance(item, ImageData) else item if isinstance(item, np.ndarray) else None
        if pixels is not None and pixels.ndim >= 2:
            facts = snapshot.models.metadata.get("image_options", {})
            count = int(pixels.shape[0])*int(pixels.shape[1])
            if count > facts.get("max_pixels", 100_000_000) or int(pixels.nbytes) + count*24 > facts.get("max_decoded_bytes", 1_073_741_824):
                raise ValueError("batch array exceeds declared source/canonical image limits before copying")
        return _prepare(item, index)

    def cancelled(index, item, timings=None):
        return BatchItemResult(index=index, input_identity=_identity(item, index), status="cancelled",
                               calibration_id=calibration_id, timings=timings or {})

    def worker(item, index):
        if event.is_set():
            return cancelled(index, item)
        output, context = snapshot._predict_image(item, stages=selected, chunk_pixels=chunk_pixels,
                                                 _return_context=True, _quiet=True)
        timings = context["timings"]
        if event.is_set():
            return cancelled(index, item, timings)
        exports = {}
        identity = _identity(item, index)
        if run_directory is not None:
            export_started = time.perf_counter()
            destination = run_directory / ("item-%08d" % index)
            temporary = Path(tempfile.mkdtemp(prefix=".item-%08d-" % index, dir=run_directory))
            export_error = cleanup_error = None
            cancelled_export = False
            try:
                for stage, image in output.items():
                    if event.is_set():
                        cancelled_export = True
                        break
                    record = write_image_data(temporary / (stage + output_ext), image,
                                              input_encoding=semantics.output_encoding,
                                              overwrite=False, **options).to_dict()
                    record["path"] = str(destination / (stage + output_ext))
                    exports[stage] = record
                if not cancelled_export:
                    timings["export_seconds"] = time.perf_counter() - export_started
                    manifest = BatchItemResult(index=index, input_identity=identity, status="succeeded",
                        calibration_id=calibration_id, exports=exports, timings=timings, provenance=context)
                    save_report(temporary / "item.json", manifest)
                    if event.is_set():
                        cancelled_export = True
                    else:
                        # A unique run directory and index prevent basename collisions.
                        # Single-directory publication hides incomplete stage sets.
                        os.rename(temporary, destination)
            except BaseException as exc:
                export_error = exc
            finally:
                if temporary.exists():
                    try:
                        if temporary.resolve().parent != run_directory.resolve():
                            raise RuntimeError("refusing cleanup outside the batch output directory")
                        shutil.rmtree(temporary)
                    except Exception as exc:
                        cleanup_error = exc
            if export_error is not None and not isinstance(export_error, Exception):
                raise export_error
            if export_error is not None or cleanup_error is not None:
                error = export_error if export_error is not None else cleanup_error
                detail = {"type": type(error).__name__, "message": str(error), "phase": "export"}
                warnings = ()
                if cleanup_error is not None:
                    detail["recovery_directory"] = str(temporary)
                    warnings = ({"type": type(cleanup_error).__name__, "message": str(cleanup_error), "phase": "cleanup"},)
                return BatchItemResult(index=index, input_identity=identity, status="failed",
                    calibration_id=calibration_id, error=detail, warnings=warnings, timings=timings, provenance=context)
            if cancelled_export:
                return cancelled(index, item, timings)
            output = {}
        return BatchItemResult(index=index, input_identity=identity, status="succeeded",
                               calibration_id=calibration_id, images=output, exports=exports,
                               timings=timings, provenance=context)

    # Constructing the engine validates options before any input is consumed or
    # output directory is created. Its generator starts on the first next().
    stream = execute_batch(inputs, worker, calibration_id=calibration_id, max_workers=max_workers,
        max_in_flight=max_in_flight, ordered=ordered, on_error=on_error, cancel_event=event,
        on_progress=on_progress, callback_error=callback_error, identity=_identity, prepare=prepare)
    if output_dir is not None:
        parent = Path(output_dir)
        parent.mkdir(parents=True, exist_ok=True)
        run_directory = parent / ("batch-" + uuid.uuid4().hex)
        run_directory.mkdir()
        if report_path is None:
            report_path = run_directory / "batch.json"
    setup = {"calibration_id": calibration_id, "seconds": time.perf_counter() - started,
             "identity_requested": allow_identity, "metadata_available": bool(snapshot.models.metadata),
             "stages": list(selected), "lut_preparation": lut_report,
             "chunk_pixels": chunk_pixels,
             "result_retention": "file manifests only" if run_directory is not None else "selected image stages"}
    def release():
        if snapshot.models.model_cc is not None and snapshot.models.model_cc[2] == "ours":
            snapshot.models.model_cc[0]._lut_predict_fn = None
    return PredictionStream(stream, event, setup=setup, output_directory=run_directory,
                            report_path=report_path, release=release)
