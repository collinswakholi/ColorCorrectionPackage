"""Bounded batch scheduling with truthful partial-success records.

The coordinator consumes at most max_in_flight accepted inputs ahead of emitted
records, including ordered reordering. Running native kernels cannot be forcibly
interrupted; cancellation stops consumption and cancels queued work.
"""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import replace
import os
import threading
import time
import traceback

from .results import BatchItemResult, BatchResult


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def _identity(item, index):
    if isinstance(item, (str, os.PathLike)):
        return {"kind": "path", "path": os.fsdecode(os.fspath(item))}
    if hasattr(item, "digest") and isinstance(item.digest, str):
        return {"kind": "typed-image", "digest": item.digest}
    return {"kind": "array-or-object", "label": "input_%06d" % index}


def _error(exc, phase):
    try:
        message = str(exc)
    except Exception:
        message = "exception message could not be rendered"
    return {"type": type(exc).__name__, "message": message, "phase": phase}


class BatchStream:
    def __init__(self, inputs, worker, *, calibration_id, max_workers=None,
                 max_in_flight=None, on_error="continue", cancel_event=None,
                 on_progress=None, callback_error="record", ordered=False,
                 identity=None, prepare=None, expected_count=None):
        if not callable(worker):
            raise TypeError("worker must be callable")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError("calibration_id must be nonempty text")
        max_workers = min(os.cpu_count() or 1, 4) if max_workers is None else max_workers
        self.max_workers = _integer(max_workers, "max_workers")
        self.max_in_flight = _integer(max_in_flight if max_in_flight is not None else self.max_workers*2, "max_in_flight")
        if on_error not in ("continue", "stop") or callback_error not in ("record", "stop"):
            raise ValueError("on_error must be continue/stop and callback_error record/stop")
        if type(ordered) is not bool:
            raise TypeError("ordered must be boolean")
        for hook, name in ((on_progress, "on_progress"), (identity, "identity"), (prepare, "prepare")):
            if hook is not None and not callable(hook):
                raise TypeError(name + " must be callable")
        if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
            raise TypeError("cancel_event must expose is_set()")
        if expected_count is None and isinstance(inputs, (list, tuple)):
            expected_count = len(inputs)
        if expected_count is not None:
            _integer(expected_count, "expected_count", 0)
        self._inputs, self._worker = inputs, worker
        self._cancel_event = cancel_event or threading.Event()
        self._on_progress, self._callback_error = on_progress, callback_error
        self._on_error, self._ordered = on_error, ordered
        self._identity, self._prepare = identity or _identity, prepare
        self._calibration_id = calibration_id
        self._closed = False
        self._iterator = None
        self._stats = {"schema_version": 1, "calibration_id": calibration_id,
                       "expected_count": expected_count, "consumed_count": 0,
                       "yielded_count": 0, "succeeded": 0, "failed": 0,
                       "cancelled": 0, "abandoned_count": 0,
                       "unsubmitted_count": expected_count,
                       "source_exhausted": False, "source_error": None,
                       "source_close_error": None,
                       "stop_reason": None, "callback_errors": 0,
                       "cancellation_requested": False, "closed_early": False,
                       "max_workers": self.max_workers, "max_in_flight": self.max_in_flight,
                       "peak_in_flight": 0, "ordered": ordered, "seconds": 0.0}

    @property
    def summary(self):
        result = deepcopy(self._stats)
        expected = result["expected_count"]
        result["unsubmitted_count"] = None if expected is None else max(0, expected-result["consumed_count"])
        result["source_count_mismatch"] = (expected != result["consumed_count"]
                                           if expected is not None and result["source_exhausted"] else None)
        return result

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._iterator is None:
            self._iterator = self._run()
        try:
            return next(self._iterator)
        except BaseException as exc:
            self._closed = True
            self._release_sources()
            # Future exceptions retain completed worker frames and their input /
            # model locals. Preserve the exception and stack locations without
            # keeping those buffers alive through traceback cycles.
            try:
                traceback.clear_frames(exc.__traceback__)
            except Exception:
                pass
            raise

    def _release_sources(self):
        # A completed iterator is useful for its summary. It must not thereby
        # retain the caller's image list, preparation/callback closures or the
        # worker's owned calibration snapshot.
        self._inputs = self._worker = self._prepare = None
        self._on_progress = self._identity = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stats["closed_early"] = not self._stats["source_exhausted"] or self._stats["consumed_count"] != self._stats["yielded_count"]
        try:
            if self._iterator is not None:
                self._iterator.close()
        finally:
            self._release_sources()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _result(self, index, identity, status, *, error=None, timings=None):
        return BatchItemResult(index, identity, status, self._calibration_id,
                               error=error, timings=timings or {})

    def _invoke(self, item, index, identity):
        started = time.perf_counter()
        try:
            output = self._worker(item, index)
            if isinstance(output, BatchItemResult):
                result = replace(output, index=index, input_identity=identity,
                                 calibration_id=self._calibration_id)
                timings = dict(result.timings)
                timings.setdefault("worker_seconds", time.perf_counter()-started)
                return replace(result, timings=timings)
            if not isinstance(output, Mapping):
                raise TypeError("batch worker must return stage arrays or BatchItemResult")
            return BatchItemResult(index, identity, "succeeded", self._calibration_id,
                                   images=output, timings={"worker_seconds": time.perf_counter()-started})
        except Exception as exc:
            return self._result(index, identity, "failed", error=_error(exc, "worker"),
                                timings={"worker_seconds": time.perf_counter()-started})

    def _run(self):
        started = time.perf_counter()
        pending, ready = {}, {}
        identities = {}
        try:
            source = iter(self._inputs)
        except Exception as exc:
            self._stats["source_error"] = _error(exc, "input_iterator")
            self._stats["stop_reason"] = "input_error"
            self._stats["seconds"] = time.perf_counter()-started
            self._release_sources()
            return
        next_yield = 0
        stopped = False
        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ccp-batch")
        try:
            while True:
                if self._cancel_event.is_set():
                    stopped = True
                    self._stats["cancellation_requested"] = True
                    self._stats["stop_reason"] = self._stats["stop_reason"] or "cancelled"
                while not stopped and not self._cancel_event.is_set() and not self._stats["source_exhausted"] and len(pending)+len(ready) < self.max_in_flight:
                    try:
                        item = next(source)
                    except StopIteration:
                        self._stats["source_exhausted"] = True
                        break
                    except Exception as exc:
                        self._stats["source_error"] = _error(exc, "input_iterator")
                        self._stats["stop_reason"] = "input_error"
                        stopped = True
                        break
                    index = self._stats["consumed_count"]
                    self._stats["consumed_count"] += 1
                    identity = {"kind": "unknown", "label": "input_%06d" % index}
                    try:
                        candidate_identity = self._identity(item, index)
                        # Validate compact identity before starting side effects.
                        identity = self._result(index, candidate_identity, "cancelled").input_identity
                        if self._cancel_event.is_set():
                            stopped = True
                            self._stats["cancellation_requested"] = True
                            self._stats["stop_reason"] = "cancelled"
                            ready[index] = self._result(index, identity, "cancelled", error={"type": "Cancelled", "message": "cancelled before preparation", "phase": "prepare"})
                            self._stats["peak_in_flight"] = max(self._stats["peak_in_flight"], len(pending)+len(ready))
                            continue
                        prepared = item if self._prepare is None else self._prepare(item, index)
                        if self._cancel_event.is_set():
                            stopped = True
                            self._stats["cancellation_requested"] = True
                            self._stats["stop_reason"] = "cancelled"
                            ready[index] = self._result(index, identity, "cancelled", error={"type": "Cancelled", "message": "cancelled before submission", "phase": "prepare"})
                            self._stats["peak_in_flight"] = max(self._stats["peak_in_flight"], len(pending)+len(ready))
                            continue
                        future = executor.submit(self._invoke, prepared, index, identity)
                        pending[future] = index
                        identities[index] = identity
                    except Exception as exc:
                        ready[index] = self._result(index, identity, "failed", error=_error(exc, "prepare"))
                    self._stats["peak_in_flight"] = max(self._stats["peak_in_flight"], len(pending)+len(ready))
                    if index in ready and self._on_error == "stop":
                        stopped = True
                        self._stats["stop_reason"] = "item_failed"
                if stopped:
                    for future in tuple(pending):
                        future.cancel()
                # Complete all already-ready futures without growing input demand.
                completed = [future for future in pending if future.done()]
                for future in completed:
                    index = pending.pop(future)
                    if future.cancelled():
                        ready[index] = self._result(index, identities.pop(index), "cancelled",
                                                   error={"type": "Cancelled", "message": "queued work cancelled before execution", "phase": "queue"})
                    else:
                        ready[index] = future.result()  # propagate BaseException after cleanup
                        identities.pop(index, None)
                    if ready[index].status == "failed" and self._on_error == "stop":
                        stopped = True
                        self._stats["stop_reason"] = self._stats["stop_reason"] or "item_failed"
                available = next_yield if self._ordered and next_yield in ready else (min(ready) if ready and not self._ordered else None)
                if available is not None:
                    result = ready.pop(available)
                    next_yield += 1
                    if self._on_progress is not None:
                        try:
                            self._on_progress(result)
                        except Exception as exc:
                            self._stats["callback_errors"] += 1
                            warning = {"type": type(exc).__name__, "message": str(exc), "phase": "progress_callback"}
                            result = replace(result, warnings=tuple(result.warnings)+(warning,))
                            if self._callback_error == "stop":
                                stopped = True
                                self._stats["stop_reason"] = "callback_failed"
                    self._stats[result.status] += 1
                    self._stats["yielded_count"] += 1
                    yield result
                    continue
                if not pending:
                    if not ready and (stopped or self._stats["source_exhausted"]):
                        break
                    continue
                wait(tuple(pending), return_when=FIRST_COMPLETED, timeout=0.05)
        finally:
            try:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True)
                # A closed iterator cannot emit these records. Do not invent
                # cancellation or success outcomes for abandoned native work.
                self._stats["abandoned_count"] = self._stats["consumed_count"]-self._stats["yielded_count"]
                self._stats["seconds"] = time.perf_counter()-started
                close_source = getattr(source, "close", None)
                if callable(close_source):
                    try:
                        close_source()
                    except Exception as exc:
                        self._stats["source_close_error"] = _error(exc, "input_close")
                        self._stats["stop_reason"] = self._stats["stop_reason"] or "input_close_error"
            finally:
                self._release_sources()


def execute_batch(inputs, worker, **options):
    return BatchStream(inputs, worker, **options)


def collect_batch(stream):
    if not isinstance(stream, BatchStream):
        raise TypeError("collect_batch requires BatchStream")
    try:
        items = list(stream)
    finally:
        stream.close()
    return BatchResult(items, stream.summary)
