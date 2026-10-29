"""Public patch 7–8 integration, including compatibility and failure ownership."""
import json
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData
from ColorCorrectionPipeline.core.correction import fit_model


def calibrated(gain=.8):
    pipeline = ColorCorrection()
    pipeline.models.model_wb = np.eye(3) * gain
    return pipeline


def image():
    return np.random.default_rng(78).uniform(.02, .8, (12, 17, 3))


def test_default_predictions_keep_all_stages_and_selected_stages_match():
    cc = calibrated()
    source = image()
    original = cc.predict_image(source)
    assert tuple(original) == ("FFC", "GC", "WB", "CC")
    for stages in ("final", ["GC", "CC"], "FFC"):
        selected = cc.predict_image(source, stages=stages)
        for stage, values in selected.items():
            np.testing.assert_array_equal(values, original[stage])
    assert tuple(cc.predict_image(source, stages="final")) == ("CC",)


@pytest.mark.parametrize("stages", [[], "wrong", ["CC", "CC"], ["WB", 2], False])
def test_invalid_retention_is_rejected(stages):
    with pytest.raises((ValueError, TypeError)):
        calibrated().predict_image(image(), stages=stages)


def test_batch_preserves_successful_middle_and_duplicate_inputs(tmp_path):
    cc = calibrated()
    source = image()
    result = cc.predict_batch([source, tmp_path / "missing.png", source], ordered=True, max_workers=2, max_in_flight=2)
    assert [item.index for item in result.items] == [0, 1, 2]
    assert [item.status for item in result.items] == ["succeeded", "failed", "succeeded"]
    assert result.summary["succeeded"] == 2 and result.summary["failed"] == 1
    assert result.summary["peak_in_flight"] <= 2
    assert len({item.calibration_id for item in result.items}) == 1
    np.testing.assert_array_equal(result.items[0].images["CC"], source * .8)


def test_snapshot_is_captured_before_input_consumption_and_refit():
    cc = calibrated()
    entered = []
    def inputs():
        entered.append(True)
        yield image()
    stream = cc.iter_predict(inputs(), max_workers=1, max_in_flight=1)
    assert not entered
    cc.models.model_wb[:] = np.eye(3) * .2
    with stream:
        result = list(stream)
    np.testing.assert_array_equal(result[0].images["CC"], image() * .8)
    np.testing.assert_array_equal(cc.predict_image(image())["CC"], image() * .2)


def test_new_batch_identity_requires_explicit_opt_in():
    cc = ColorCorrection()
    with pytest.raises(ValueError, match="requires calibration"):
        cc.predict_batch([image()])
    result = cc.predict_batch([image()], allow_identity=True)
    np.testing.assert_array_equal(result.items[0].images["CC"], image())
    assert result.summary["setup"]["metadata_available"] is False


def test_batch_callback_failure_records_warning_without_losing_success():
    def callback(result):
        raise RuntimeError("progress sink failed")
    result = calibrated().predict_batch([image(), image()], on_progress=callback)
    assert result.summary["callback_errors"] == 2
    assert result.summary["succeeded"] == 2
    assert all(item.warnings for item in result.items)


def test_early_close_writes_honest_compact_report(tmp_path):
    report = tmp_path / "stream.json"
    stream = calibrated().iter_predict((image() for _ in range(30)), max_workers=1, max_in_flight=2, report_path=report)
    with stream:
        assert next(stream).status == "succeeded"
    saved = json.loads(report.read_text())
    assert saved["closed_early"]
    assert saved["consumed_count"] <= 2
    assert saved["yielded_count"] == 1
    assert saved["consumed_count"] == saved["yielded_count"] + saved["abandoned_count"]
    assert '"pixels"' not in report.read_text()


def test_explicit_cancellation_stops_new_consumption():
    event = threading.Event()
    event.set()
    consumed = []
    def inputs():
        consumed.append(True)
        yield image()
    result = calibrated().predict_batch(inputs(), cancel_event=event)
    assert not result.items and not consumed
    assert result.summary["cancellation_requested"]


def test_file_outputs_are_unique_manifested_and_precision_preserving(tmp_path):
    import tifffile
    cc = calibrated()
    result = cc.predict_batch([image(), image()], output_dir=tmp_path, stages=["WB", "CC"], max_workers=2)
    assert result.summary["succeeded"] == 2, result.to_dict()
    assert all(not item.images for item in result.items)
    paths = [Path(item.exports["CC"]["path"]) for item in result.items]
    assert paths[0] != paths[1]
    for item, path in zip(result.items, paths):
        assert path.exists() and (path.parent / "item.json").exists()
        np.testing.assert_array_equal(tifffile.imread(path), image() * .8)
        assert item.exports["CC"]["dtype"] == "float64"
    saved = json.loads((paths[0].parent.parent / "batch.json").read_text())
    assert saved["succeeded"] == 2
    another = cc.predict_batch([image()], output_dir=tmp_path)
    assert Path(another.items[0].exports["CC"]["path"]).parent.parent != paths[0].parent.parent


def test_multi_stage_export_failure_never_publishes_a_partial_item(tmp_path, monkeypatch):
    import ColorCorrectionPipeline.io as module
    original = module.write_image_data
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("encoder failed on second stage")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "write_image_data", fail_second)
    result = calibrated().predict_batch([image()], output_dir=tmp_path, stages=["WB", "CC"], max_workers=1)
    assert result.summary["failed"] == 1
    run = Path(result.summary["output_directory"])
    assert not list(run.glob("item-*")) and not list(run.glob(".item-*"))
    assert "second stage" in result.items[0].error["message"]


@pytest.mark.parametrize("options", [[], False, 0, ""])
def test_invalid_falsy_export_options_are_not_silently_ignored(options):
    with pytest.raises(ValueError, match="export_options"):
        calibrated().iter_predict([image()], export_options=options)


def test_report_failure_is_visible_and_preserves_results_and_destination(tmp_path, monkeypatch):
    import ColorCorrectionPipeline.reporting as module
    path = tmp_path / "report.json"
    path.write_text("previous")
    monkeypatch.setattr(module, "save_report", lambda *args: (_ for _ in ()).throw(OSError("report write failed")))
    result = calibrated().predict_batch([image()], report_path=path)
    assert result.summary["succeeded"] == 1
    assert result.summary["report"]["status"] == "failed"
    assert path.read_text() == "previous"


def test_failed_lut_setup_uses_one_reported_direct_fallback(monkeypatch):
    import ColorCorrectionPipeline.core.correction as correction
    cc = ColorCorrection()
    x = np.random.default_rng(3).random((24, 3))
    model = fit_model(x, x, {"mtd": "linear", "degree": 1, "lut_min_pixels": 1})
    cc.models.model_cc = (model, {}, "ours")
    calls = []
    def ensure(candidate):
        if candidate.use_lut:
            calls.append(True)
        return False
    monkeypatch.setattr(correction, "_ensure_lut", ensure)
    result = cc.predict_batch([image(), image()], max_workers=2)
    assert result.summary["succeeded"] == 2
    assert len(calls) == 1
    assert "direct" in result.summary["setup"]["lut_preparation"]["fallback"]


@pytest.mark.parametrize("use_lut", [False, True])
def test_chunking_preserves_whole_image_inference_path_even_for_short_tail(use_lut):
    cc = ColorCorrection()
    x = np.random.default_rng(3).random((24, 3))
    model = fit_model(x, x**.7, {"mtd": "linear", "degree": 2, "use_lut": use_lut, "lut_min_pixels": 20})
    cc.models.model_cc = (model, {}, "ours")
    source = image()
    expected = cc.predict_image(source)["CC"]
    actual = cc.predict_image(source, stages="final", chunk_pixels=50)["CC"]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    result = cc.predict_batch([source], chunk_pixels=50)
    np.testing.assert_allclose(result.items[0].images["CC"], expected, atol=1e-12, rtol=1e-12)


def test_bundle_public_activation_and_failed_load_preserve_previous_context(tmp_path):
    cc = calibrated()
    path = tmp_path / "calibration.ccpb"
    cc.save_bundle(path)
    restored = calibrated(.2)
    restored.load_bundle(path)
    np.testing.assert_array_equal(restored.predict_image(image())["CC"], image() * .8)
    assert restored.last_run["operation"] == "load_bundle"
    before = restored.models
    path.write_bytes(b"not a bundle")
    with pytest.raises(ValueError):
        restored.load_bundle(path)
    assert restored.models is before
    assert restored.last_run["operation"] == "load_bundle"


def test_legacy_predict_images_return_contract_is_unchanged():
    result = calibrated().predict_images([image(), image()], max_workers=1)
    assert isinstance(result, list) and len(result) == 2
    assert all(tuple(item) == ("FFC", "GC", "WB", "CC") for item in result)


def test_new_image_options_require_explicit_semantics():
    with pytest.raises(ValueError, match="require explicit"):
        Config(image_options={"grayscale_policy": "replicate"}).validate()
    Config(color_semantics={"version": "explicit-v1"}, image_options={"grayscale_policy": "replicate", "max_pixels": 10000}).validate()


def test_stream_releases_prior_arrays_instead_of_retaining_batch_history():
    references = []
    def inputs():
        for _ in range(100):
            source = image()
            references.append(weakref.ref(source))
            yield source
    outputs = []
    with calibrated().iter_predict(inputs(), max_workers=2, max_in_flight=3) as stream:
        for item in stream:
            outputs.append(weakref.ref(item.images["CC"]))
            assert sum(reference() is not None for reference in references) <= 4
            assert sum(reference() is not None for reference in outputs) <= 5
    assert stream.summary["consumed_count"] == 100
    assert stream.summary["peak_in_flight"] <= 3


def test_mutable_arrays_are_copied_before_the_generator_resumes():
    original = image()
    expected = original.copy() * .8
    def inputs():
        yield original
        original[:] = .1
        yield image()
    result = calibrated().predict_batch(inputs(), ordered=True, max_workers=1, max_in_flight=2)
    np.testing.assert_array_equal(result.items[0].images["CC"], expected)


def test_cancellation_during_export_leaves_no_published_partial_item(tmp_path, monkeypatch):
    import ColorCorrectionPipeline.io as module
    event = threading.Event()
    original = module.write_image_data
    def cancel_after_first(*args, **kwargs):
        record = original(*args, **kwargs)
        event.set()
        return record
    monkeypatch.setattr(module, "write_image_data", cancel_after_first)
    result = calibrated().predict_batch([image()], stages=["WB", "CC"], output_dir=tmp_path,
                                       cancel_event=event, max_workers=1, max_in_flight=1)
    assert result.items[0].status == "cancelled"
    run = Path(result.summary["output_directory"])
    assert not list(run.glob("item-*")) and not list(run.glob(".item-*"))


def test_closed_stream_does_not_keep_its_owned_model_or_lut_alive(monkeypatch):
    import ColorCorrectionPipeline.bundle as bundle
    original = bundle.snapshot_models
    references = []
    def snapshot(*args, **kwargs):
        candidate = original(*args, **kwargs)
        if candidate.model_cc is not None:
            references.append(weakref.ref(candidate.model_cc[0]))
        return candidate
    monkeypatch.setattr(bundle, "snapshot_models", snapshot)
    cc = ColorCorrection()
    x = np.random.default_rng(3).random((24, 3))
    cc.models.model_cc = (fit_model(x, x, {"degree": 1, "lut_min_pixels": 1}), {}, "ours")
    stream = cc.iter_predict([image()], max_workers=1)
    with stream:
        assert len(list(stream)) == 1
    assert references and all(reference() is None for reference in references)


def test_nonfinite_actual_inference_is_a_failed_item_in_array_mode(monkeypatch):
    import ColorCorrectionPipeline.prediction as prediction
    cc = ColorCorrection()
    x = np.random.default_rng(3).random((24, 3))
    cc.models.model_cc = (fit_model(x, x, {"degree": 1, "use_lut": False}), {}, "ours")
    monkeypatch.setattr(prediction, "chunked_cc", lambda source, *args: np.full_like(source, np.nan))
    result = cc.predict_batch([image()], chunk_pixels=10)
    assert result.items[0].status == "failed" and not result.items[0].images
    assert result.items[0].error["type"] == "FloatingPointError"
