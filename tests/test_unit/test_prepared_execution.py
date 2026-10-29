"""Prepared execution is opt-in, owned, bounded, and error-qualified."""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ColorCorrectionPipeline import (ColorCorrection, ExecutionOptions, LutCache,
                                     AccuracyBudgetError)
from ColorCorrectionPipeline.acceleration import PreparedExecution, _measure
from ColorCorrectionPipeline.colorimetry import srgb_to_linear
from ColorCorrectionPipeline.fitting import fit_observations


def calibration(nonlinear=False):
    cc = ColorCorrection()
    cc.models.model_wb = np.array([[.8, .02, .03], [.01, .75, .01], [.02, .03, .85]])
    if nonlinear:
        x = np.random.default_rng(14).uniform(.01, .99, (40, 3))
        cc.models.model_cc = (fit_observations(x, x*x, ["cell%d" % i for i in range(40)],
            {"degree": 2, "use_lut": False}, "reproducible-v1"), {}, "ours")
    return cc


def options(**updates):
    values = dict(mode="auto", backend="numpy", grid_sizes=(3,), validation_samples=64,
                  stress_grid_size=3,
                  max_rgb_error=1e-10, max_delta_e00=1e-8)
    values.update(updates)
    return values


@pytest.mark.parametrize("bad", [dict(mode="gpu"), dict(backend="vulkan"), dict(grid_sizes=(3, 3)),
    dict(grid_sizes=(1,)), dict(grid_sizes=(130,)), dict(validation_samples=63),
    dict(stress_grid_size=2), dict(stress_grid_size=68), dict(stress_grid_size=True),
    dict(max_rgb_error=float("nan")), dict(max_delta_e00=True), dict(chunk_pixels=0),
    dict(random_state=-1), dict(allow_fallback=1)])
def test_options_reject_invalid(bad):
    with pytest.raises((ValueError, TypeError)):
        ExecutionOptions(**bad)


def test_unfitted_preparation_is_rejected():
    with pytest.raises(ValueError, match="fitted stage"):
        ColorCorrection().prepare_execution(options())


def test_white_semantics_change_cache_identity():
    cc, cache = calibration(), LutCache()
    with cc.prepare_execution(options(), cache=cache) as first:
        identity = first.pointwise_id
    cc.REF_ILLUMINANT = np.array([.3457, .3585])
    with cc.prepare_execution(options(), cache=cache) as second:
        assert second.pointwise_id != identity


def test_explicit_boundary_encoding_and_metadata_white_are_preserved(monkeypatch, tmp_path):
    from tests.test_unit.test_modern_pipeline import chart, config
    from ColorCorrectionPipeline import ImageData
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    cfg = config(reference_profile=reference, do_gc=False, do_wb=True, do_cc=False,
                 color_semantics={"version": "explicit-v1", "input_encoding": "linear-srgb", "output_encoding": "linear-srgb"})
    _, _, failed = cc.run(ImageData.from_array(image), config=cfg)
    assert not failed, cc.last_run
    cc.REF_ILLUMINANT = np.array([.3457, .3585])  # metadata must take precedence
    linear = srgb_to_linear(image)
    expected = cc.predict_image(linear)["CC"]
    cc.REF_ILLUMINANT = np.array([.3457, .3585])
    with cc.prepare_execution(options(max_rgb_error=1e-8, max_delta_e00=1e-7), cache=LutCache()) as prepared:
        np.testing.assert_allclose(prepared.predict_image(linear).images["CC"], expected, atol=5e-16)
        np.testing.assert_array_equal(prepared._pipeline.REF_ILLUMINANT, [.31271, .32902])
        with pytest.raises(ValueError, match="working encoded sRGB"):
            prepared.export_cube(tmp_path / "no.cube")
        prepared.export_cube(tmp_path / "working.cube", working_space=True)


def test_combined_asymmetric_affine_matches_direct_and_cube_order(tmp_path):
    cc = calibration()
    image = np.random.default_rng(15).random((17, 13, 3))
    expected = cc.predict_image(image)["CC"]
    with cc.prepare_execution(options()) as prepared:
        assert prepared.preparation["qualification"]["selected_mode"] == "combined-lut"
        result = prepared.predict_image(image)
        np.testing.assert_allclose(result.images["CC"], expected, atol=4e-16, rtol=0)
        assert result.diagnostics["backend"]["actual_backend"] == "cpu-numpy"
        report = prepared.export_cube(tmp_path / "model.cube")
        rows = np.array([[float(v) for v in line.split()] for line in
                         (tmp_path / "model.cube").read_text().splitlines() if line and line[0].isdigit()])
        coordinates = np.array([[r, g, b] for b in (0., .5, 1.) for g in (0., .5, 1.) for r in (0., .5, 1.)])
        np.testing.assert_allclose(rows, coordinates @ cc.models.model_wb, atol=1e-16, rtol=0)
        assert report["grid_size"] == 3 and report["FFC_excluded"]
        assert not prepared._lut.flags.writeable
        with pytest.raises(ValueError):
            prepared._lut.setflags(write=True)
        with pytest.raises(ValueError, match="intermediates"):
            prepared.predict_image(image, stages="all")
    with pytest.raises(RuntimeError, match="closed"):
        prepared.predict_image(image)


def test_snapshot_survives_owner_mutation_and_new_identity_invalidates():
    cc, cache = calibration(), LutCache()
    image = np.random.default_rng(16).random((4, 5, 3))
    with cc.prepare_execution(options(), cache=cache) as first:
        before = first.predict_image(image).images["CC"].copy()
        cc.models.model_wb *= .5
        np.testing.assert_array_equal(first.predict_image(image).images["CC"], before)
        with cc.prepare_execution(options(), cache=cache) as second:
            assert first.pointwise_id != second.pointwise_id
            np.testing.assert_allclose(second.predict_image(image).images["CC"], before*.5)
    assert cache.info["builds"] == 2


def test_cache_hit_restores_direct_backend_and_owns_diagnostics():
    cc, cache = calibration(True), LutCache()
    with cc.prepare_execution(options(), cache=cache) as first:
        assert first.preparation["qualification"]["selected_mode"] == "direct"
        image = np.random.default_rng(17).random((5, 4, 3))
        expected = first.predict_image(image).images["CC"]
    with cc.prepare_execution(options(), cache=cache) as second:
        assert second.preparation["cache_hit"]
        np.testing.assert_array_equal(second.predict_image(image).images["CC"], expected)
        report = second.preparation
        report["qualification"]["reason"] = "tampered"
        assert second.preparation["qualification"]["reason"] != "tampered"


def test_unqualified_forced_lut_has_honest_fallback_or_raises():
    cc = calibration(True)
    with cc.prepare_execution(options(mode="combined-lut"), cache=LutCache()) as prepared:
        assert prepared.preparation["qualification"]["selected_mode"] == "direct"
        assert "no candidate" in prepared.preparation["qualification"]["reason"]
    with pytest.raises(AccuracyBudgetError, match="no candidate"):
        cc.prepare_execution(options(mode="combined-lut", allow_fallback=False), cache=LutCache())


def test_direct_intermediate_stops_before_cc_and_shares_final_storage(monkeypatch):
    with calibration(True).prepare_execution(options(mode="direct"), cache=LutCache()) as prepared:
        import ColorCorrectionPipeline.core.execution as execution
        monkeypatch.setattr(execution, "evaluate_direct", lambda *a, **k: pytest.fail("CC must not run"))
        image = np.full((4, 3, 3), .25)
        result = prepared.predict_image(image, stages=("FFC", "WB"))
        np.testing.assert_array_equal(result.images["FFC"], image)
        np.testing.assert_allclose(result.images["WB"], image @ prepared._pipeline.models.model_wb)


def test_spatial_ffc_is_separate_from_table_cache_and_requires_export_consent(tmp_path):
    cc, cache = calibration(), LutCache()
    image = np.full((4, 5, 3), .25)
    with cc.prepare_execution(options(), cache=cache) as plain:
        identity = plain.pointwise_id
    cc.models.model_ffc = np.full((4, 5), 1.1)
    with cc.prepare_execution(options(), cache=cache) as spatial:
        assert spatial.pointwise_id == identity and spatial.preparation["cache_hit"]
        np.testing.assert_allclose(spatial.predict_image(image).images["CC"], cc.predict_image(image)["CC"], atol=5e-16)
        with pytest.raises(ValueError, match="spatial FFC"):
            spatial.export_cube(tmp_path / "no.cube")
        assert not (tmp_path / "no.cube").exists()
        spatial.export_cube(tmp_path / "yes.cube", exclude_ffc=True)


def test_separate_audit_failure_does_not_tune_on_audit(monkeypatch):
    import ColorCorrectionPipeline.acceleration as module
    real_measure, calls = module._measure, []
    def fail_audit(*args):
        error = real_measure(*args)
        calls.append(error)
        if len(calls) == 4:
            error["max_rgb_error"] = 1.
        return error
    monkeypatch.setattr(module, "_measure", fail_audit)
    with calibration().prepare_execution(options(grid_sizes=(3, 5)), cache=LutCache()) as prepared:
        proof = prepared.preparation["qualification"]
        assert len(proof["candidates"]) == 1
        assert proof["qualification_sha256"] != proof["audit_sha256"]
        assert proof["selected_mode"] == "direct" and "without retuning" in proof["reason"]


def test_rgb_budget_uses_declared_output_encoding():
    a, b = np.full((2, 3), .2), np.full((2, 3), .21)
    error = _measure(a, b, "linear-srgb")
    assert error["max_rgb_error"] == pytest.approx(float((srgb_to_linear(b)-srgb_to_linear(a)).max()))
    assert error["max_delta_e00"] > 0


def test_dense_stress_failure_falls_back_without_retuning(monkeypatch):
    import ColorCorrectionPipeline.acceleration as module
    original, calls = module._measure, []
    def stress_failure(*args):
        error = original(*args)
        calls.append(error)
        if len(calls) == 5:
            error["max_rgb_error"] = 1.
        return error
    monkeypatch.setattr(module, "_measure", stress_failure)
    with calibration().prepare_execution(options(grid_sizes=(3, 5)), cache=LutCache()) as prepared:
        proof = prepared.preparation["qualification"]
        assert len(proof["candidates"]) == 1 and proof["selected_mode"] == "direct"
        assert proof["stress_components"]["center_points"] == 27
        assert proof["stress"]["samples"] == 4251
        assert "dense stress" in proof["reason"]


def test_cache_lru_bytes_and_single_concurrent_build():
    cache, entered, release = LutCache(max_bytes=200, max_entries=2), threading.Event(), threading.Event()
    table = np.zeros((2, 2, 2, 3))
    calls = []
    def builder():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return table, {"owned": []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get_or_build, "same", builder) for _ in range(4)]
        assert entered.wait(5)
        release.set()
        results = [f.result(timeout=5) for f in futures]
    assert len(calls) == 1 and cache.info["hits"] == 3
    results[0][1]["owned"].append("external")
    assert cache.get_or_build("same", builder)[1]["owned"] == []
    cache.get_or_build("other", lambda: (table, {}))
    assert cache.info["entries"] == 1 and cache.info["bytes"] == table.nbytes
    cache.get_or_build("huge", lambda: (np.zeros((3, 3, 3, 3)), {}))
    assert cache.info["bytes"] == table.nbytes
    cache.clear()
    assert cache.info["entries"] == cache.info["bytes"] == 0


@pytest.mark.parametrize("operation", ["predict", "export"])
def test_close_waits_for_running_work(monkeypatch, tmp_path, operation):
    prepared = calibration().prepare_execution(options(), cache=LutCache())
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    if operation == "predict":
        import ColorCorrectionPipeline.core.execution as module
        original = module.apply_lut
        def block(*args, **kwargs):
            entered.set(); assert release.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(module, "apply_lut", block)
        action = lambda: prepared.predict_image(np.full((2, 3, 3), .5))
    else:
        original = prepared._export_cube
        def block(*args, **kwargs):
            entered.set(); assert release.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(prepared, "_export_cube", block)
        action = lambda: prepared.export_cube(tmp_path / "ok.cube")
    def close():
        closing.set()
        prepared.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(action)
        assert entered.wait(5)
        done = pool.submit(close)
        assert closing.wait(5) and not done.done()
        release.set()
        work.result(timeout=5); done.result(timeout=5)
    assert prepared._pipeline is prepared._lut is None


def test_atomic_cube_failure_preserves_existing_file(monkeypatch, tmp_path):
    import ColorCorrectionPipeline.acceleration as module
    destination = tmp_path / "prior.cube"
    destination.write_bytes(b"prior")
    with calibration().prepare_execution(options(), cache=LutCache()) as prepared:
        def denied(*args):
            raise OSError("injected replace failure")
        monkeypatch.setattr(module.os, "replace", denied)
        with pytest.raises(OSError, match="injected"):
            prepared.export_cube(destination)
    assert destination.read_bytes() == b"prior"
    assert list(tmp_path.iterdir()) == [destination]
