import json
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ColorCorrectionPipeline.batch import execute_batch, collect_batch
from ColorCorrectionPipeline.results import BatchItemResult
from ColorCorrectionPipeline.reporting import save_report


def pixel(value):
    return {"CC": np.full((2, 3, 3), value, dtype=float)}


def run(values, worker=None, **options):
    return execute_batch(values, worker or (lambda item, index: pixel(item)),
                         calibration_id="fixed-model", **options)


def test_duplicates_keep_index_identity_and_snapshot():
    result = collect_batch(run(["same.jpg", "same.jpg"], lambda item, index: pixel(index)))
    assert [record.index for record in result.items] == [0, 1]
    assert result.items[0].input_identity == result.items[1].input_identity
    assert all(record.calibration_id == "fixed-model" for record in result.items)
    assert result.summary["succeeded"] == result.summary["yielded_count"] == 2
    assert result.summary["source_exhausted"]


def test_mixed_success_errors_preserve_all_records():
    def worker(item, index):
        if item == "bad":
            raise ValueError("broken input")
        return pixel(index)
    stream = run(["first", "bad", "last"], worker, ordered=True)
    result = collect_batch(stream)
    assert [item.status for item in result.items] == ["succeeded", "failed", "succeeded"]
    assert result.items[1].error["phase"] == "worker"
    assert result.summary["consumed_count"] == 3
    assert result.summary["succeeded"] == 2 and result.summary["failed"] == 1
    assert result.summary["abandoned_count"] == 0


def test_prepare_owns_mutable_inputs_before_next_consumption():
    shared = np.zeros((2, 3, 3))
    def inputs():
        for i in range(4):
            shared.fill(i)
            yield shared
    stream = run(inputs(), lambda item, index: {"CC": item},
                 prepare=lambda item, index: item.copy(), ordered=True, max_in_flight=4)
    records = list(stream)
    assert [record.images["CC"][0, 0, 0] for record in records] == list(range(4))


def test_unknown_generator_count_is_never_invented():
    stream = run((item for item in [1, 2, 3]), max_in_flight=1)
    assert stream.summary["expected_count"] is None
    result = collect_batch(stream)
    assert result.summary["expected_count"] is None
    assert result.summary["unsubmitted_count"] is None
    assert result.summary["consumed_count"] == 3


def test_unordered_stream_does_not_wait_for_first_input():
    release = threading.Event()
    entered = threading.Event()
    def worker(item, index):
        if index == 0:
            entered.set()
            assert release.wait(5)
        return pixel(index)
    stream = run([0, 1, 2], worker, max_workers=2, max_in_flight=2)
    try:
        first = next(stream)
        assert entered.is_set()
        assert first.index == 1
    finally:
        release.set()
        stream.close()
    assert stream.summary["abandoned_count"] == 1


def test_ordered_ready_buffer_counts_toward_bound():
    release, prefetched, second_done = threading.Event(), threading.Event(), threading.Event()
    consumed = []
    def inputs():
        for i in range(20):
            consumed.append(i)
            if len(consumed) == 3:
                prefetched.set()
            yield i
    def worker(item, index):
        if index == 0:
            assert release.wait(5)
        else:
            second_done.set()
        return pixel(index)
    stream = run(inputs(), worker, max_workers=2, max_in_flight=3, ordered=True)
    with ThreadPoolExecutor(max_workers=1) as observer:
        future = observer.submit(next, stream)
        try:
            assert prefetched.wait(5) and second_done.wait(5)
            assert len(consumed) == 3
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=5).index == 0
    remaining = list(stream)
    assert [item.index for item in remaining] == list(range(1, 20))
    assert stream.summary["peak_in_flight"] == 3


def test_iterator_close_joins_workers_and_accounts_abandoned():
    completed = threading.Event()
    def worker(item, index):
        completed.set()
        return pixel(item)
    stream = run([1, 2, 3, 4], worker, max_workers=1, max_in_flight=2)
    next(stream)
    stream.close()
    assert completed.is_set()
    summary = stream.summary
    assert summary["closed_early"]
    assert summary["yielded_count"] + summary["abandoned_count"] == summary["consumed_count"]
    assert summary["unsubmitted_count"] == 2
    assert not any(thread.name.startswith("ccp-batch") for thread in threading.enumerate())
    assert list(stream) == []


def test_callback_errors_record_warning_without_losing_success():
    def broken_callback(item):
        raise RuntimeError("UI callback failed")
    stream = run([1, 2], on_progress=broken_callback)
    results = collect_batch(stream)
    assert all(result.status == "succeeded" for result in results.items)
    assert results.items[0].warnings[0]["phase"] == "progress_callback"
    assert results.summary["callback_errors"] == 2
    assert results.summary["failed"] == 0


def test_callback_stop_prevents_new_consumption():
    def broken_callback(item):
        raise RuntimeError("UI callback failed")
    stream = run(list(range(10)), on_progress=broken_callback, callback_error="stop", max_in_flight=1)
    records = list(stream)
    assert len(records) == 1 and records[0].status == "succeeded"
    assert stream.summary["stop_reason"] == "callback_failed"
    assert stream.summary["unsubmitted_count"] == 9


def test_stop_on_item_error_is_explicit_and_bounded():
    def worker(item, index):
        raise ValueError("bad")
    stream = run(list(range(10)), worker, on_error="stop", max_in_flight=1)
    assert [item.status for item in stream] == ["failed"]
    assert stream.summary["stop_reason"] == "item_failed"
    assert stream.summary["unsubmitted_count"] == 9


def test_cancelled_before_start_never_consumes_generator():
    event = threading.Event()
    event.set()
    consumed = []
    def values():
        consumed.append(1)
        yield 1
    stream = run(values(), cancel_event=event)
    assert list(stream) == []
    assert not consumed
    assert stream.summary["cancellation_requested"]
    assert stream.summary["stop_reason"] == "cancelled"


def test_cancel_running_work_does_not_claim_kernel_interruption():
    event = threading.Event()
    def worker(item, index):
        event.set()
        return pixel(item)
    stream = run(list(range(10)), worker, cancel_event=event, max_workers=1, max_in_flight=1)
    records = list(stream)
    assert len(records) == 1 and records[0].status == "succeeded"
    assert stream.summary["cancellation_requested"]
    assert stream.summary["unsubmitted_count"] == 9


def test_prepare_failure_is_a_result_and_can_continue():
    def prepare(item, index):
        if index == 1:
            raise TypeError("input ownership failed")
        return item
    result = collect_batch(run([1, 2, 3], prepare=prepare, ordered=True))
    assert [item.status for item in result.items] == ["succeeded", "failed", "succeeded"]
    assert result.items[1].error["phase"] == "prepare"


def test_source_failure_is_separate_from_input_results():
    def inputs():
        yield 1
        raise RuntimeError("source broke")
    stream = run(inputs())
    assert [item.status for item in stream] == ["succeeded"]
    summary = stream.summary
    assert summary["source_error"]["phase"] == "input_iterator"
    assert summary["stop_reason"] == "input_error"
    assert summary["source_exhausted"] is False


def test_worker_base_exception_propagates_and_cleans_up():
    def worker(item, index):
        raise KeyboardInterrupt()
    stream = run([1], worker, max_workers=1)
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert stream.summary["abandoned_count"] == 1
    assert not any(thread.name.startswith("ccp-batch") for thread in threading.enumerate())


def test_progress_runs_in_coordinator_thread():
    caller = threading.get_ident()
    called = []
    list(run([1, 2, 3], on_progress=lambda item: called.append(threading.get_ident())))
    assert called == [caller]*3


def test_result_worker_cannot_swap_index_identity_or_calibration():
    def worker(item, index):
        return BatchItemResult(99, "fake", "succeeded", "other-model", images=pixel(index),
                               exports={"CC": {"path": "000000_CC.png", "dtype": "uint16"}})
    record = list(run(["source.jpg"], worker))[0]
    assert record.index == 0 and record.calibration_id == "fixed-model"
    assert record.input_identity["path"] == "source.jpg"
    assert record.exports["CC"]["dtype"] == "uint16"


def test_summary_and_report_are_owned_and_never_embed_pixels(tmp_path):
    collected = collect_batch(run([.1]))
    summary = collected.summary
    summary["succeeded"] = 99
    assert collected.summary["succeeded"] == 1
    saved = save_report(tmp_path / "report.json", collected)
    data = json.loads(saved.read_text())
    assert data["items"][0]["images"]["CC"] == {"shape": [2, 3, 3], "dtype": "float64", "bytes": 144}
    assert "pixels" not in saved.read_text()


def test_report_encoding_error_preserves_previous_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("previous")
    with pytest.raises(TypeError):
        save_report(path, {"image": np.zeros((2, 3, 3))})
    assert path.read_text() == "previous"
    with pytest.raises(TypeError):
        save_report(path, {"nonfinite": float("nan")})
    assert path.read_text() == "previous"


def test_report_replace_failure_preserves_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    import ColorCorrectionPipeline.reporting as reporting
    path = tmp_path / "report.json"
    path.write_text("previous")
    monkeypatch.setattr(reporting.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("failed replace")))
    with pytest.raises(OSError, match="failed replace"):
        save_report(path, {"ok": True})
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("option,value", [("max_workers", 0), ("max_workers", True), ("max_in_flight", 0),
    ("on_error", "ignore"), ("callback_error", "ignore"), ("ordered", 1), ("expected_count", -1)])
def test_invalid_options_fail_before_consuming_inputs(option, value):
    consumed = []
    def inputs():
        consumed.append(True)
        yield 1
    with pytest.raises((ValueError, TypeError)):
        run(inputs(), **{option: value})
    assert not consumed


def test_empty_input_has_zero_counts():
    result = collect_batch(run([]))
    assert result.items == ()
    assert result.summary["expected_count"] == result.summary["consumed_count"] == result.summary["yielded_count"] == 0
    assert result.summary["source_exhausted"]


def test_context_manager_closes_stream_even_on_consumer_error():
    stream = run([1, 2, 3], max_in_flight=1)
    with pytest.raises(RuntimeError):
        with stream:
            next(stream)
            raise RuntimeError("consumer failed")
    assert stream.summary["closed_early"]
    assert list(stream) == []


def test_invalid_identity_is_a_prepare_failure_with_fallback_identity():
    stream = run([1, 2], identity=lambda item, index: np.zeros((2, 3, 3)))
    records = list(stream)
    assert all(record.status == "failed" and record.error["phase"] == "prepare" for record in records)
    assert [record.input_identity["label"] for record in records] == ["input_000000", "input_000001"]


def test_records_own_nested_metadata_and_expose_it_read_only():
    exports = {"CC": {"path": "000000_CC.png", "settings": [16, "srgb"]}}
    record = BatchItemResult(0, {"path": ["a", "b"]}, "succeeded", "id", exports=exports)
    exports["CC"]["path"] = "changed"
    assert record.exports["CC"]["path"] == "000000_CC.png"
    with pytest.raises(TypeError):
        record.exports["CC"]["path"] = "mutated"
    assert record.to_dict()["input_identity"]["path"] == ["a", "b"]


def test_non_iterable_source_is_structured_source_error():
    stream = run(42)
    assert list(stream) == []
    assert stream.summary["source_error"]["type"] == "TypeError"
    assert stream.summary["stop_reason"] == "input_error"


def test_source_close_error_does_not_mask_consumer_error():
    def source():
        try:
            yield 1
            yield 2
        finally:
            raise ValueError("source close failed")
    stream = run(source(), max_in_flight=1)
    with pytest.raises(RuntimeError, match="consumer"):
        with stream:
            next(stream)
            raise RuntimeError("consumer")
    assert stream.summary["source_close_error"]["type"] == "ValueError"


def test_changed_list_length_is_reported_without_fabricated_records():
    inputs = [1, 2]
    stream = run(inputs, max_in_flight=1)
    inputs.append(3)
    assert len(list(stream)) == 3
    assert stream.summary["expected_count"] == 2
    assert stream.summary["consumed_count"] == 3
    assert stream.summary["source_count_mismatch"] is True


def test_cancel_during_preparation_never_runs_worker():
    event = threading.Event()
    ran = []
    def prepare(item, index):
        event.set()
        return item
    stream = run([1, 2], lambda item, index: ran.append(index) or pixel(item),
                 prepare=prepare, cancel_event=event, max_in_flight=2)
    records = list(stream)
    assert not ran
    assert len(records) == 1 and records[0].status == "cancelled"
    assert stream.summary["unsubmitted_count"] == 1
    assert stream.summary["peak_in_flight"] == 1


def test_identity_is_frozen_before_worker_and_later_generator_mutation():
    shared = {"name": "first"}
    def source():
        yield 1
        shared["name"] = "second"
        yield 2
    records = list(run(source(), identity=lambda item, index: shared, max_in_flight=2, ordered=True))
    assert [record.input_identity["name"] for record in records] == ["first", "second"]


@pytest.mark.parametrize("finish", ["exhaust", "close", "unstarted", "failure"])
def test_closed_stream_releases_input_and_worker_graphs_without_losing_summary(finish):
    class Inputs(list):
        pass
    class Worker:
        def __init__(self):
            self.model = np.ones((512, 512))
        def __call__(self, item, index):
            if finish == "failure":
                raise KeyboardInterrupt()
            return pixel(index)
    inputs, worker = Inputs([1, 2, 3]), Worker()
    references = [weakref.ref(inputs), weakref.ref(worker), weakref.ref(worker.model)]
    stream = run(inputs, worker, max_in_flight=1)
    del inputs, worker
    if finish == "exhaust":
        list(stream)
    elif finish == "close":
        next(stream)
        stream.close()
    elif finish == "failure":
        with pytest.raises(KeyboardInterrupt):
            next(stream)
    else:
        stream.close()
    assert all(reference() is None for reference in references)
    assert stream.summary["expected_count"] == 3
    assert stream._inputs is stream._worker is stream._prepare is stream._on_progress is stream._identity is None
