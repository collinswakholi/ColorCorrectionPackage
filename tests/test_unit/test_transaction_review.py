"""Independent adversarial checks of the complete-run transaction boundary."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config


def test_prediction_during_fit_uses_the_complete_previous_calibration(monkeypatch):
    pipeline = ColorCorrection()
    pipeline.get_reference_values()
    previous_illuminant = pipeline.REF_ILLUMINANT.copy()
    previous_models = pipeline.models
    previous_models.model_wb = np.eye(3) * 1.5
    entered = threading.Event()
    release = threading.Event()

    def delayed_white_balance(candidate, Image, do_wb, wb_kwargs):
        candidate.models.model_wb = np.eye(3) * 2.0
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("test did not release candidate fit")
        return Image @ candidate.models.model_wb, {}, False

    monkeypatch.setattr(ColorCorrection, "do_white_balance", delayed_white_balance)
    image = np.full((2, 2, 3), 0.4)
    config = Config(
        do_ffc=False,
        do_gc=False,
        do_wb=True,
        do_cc=False,
        check_saturation=False,
        REF_ILLUMINANT=np.array([0.34567, 0.35850]),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.run, image, config=config)
        try:
            assert entered.wait(timeout=10), "candidate WB stage did not start"
            assert pipeline.models is previous_models
            np.testing.assert_array_equal(pipeline.REF_ILLUMINANT, previous_illuminant)
            np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.6)
        finally:
            release.set()
        _, _, failed = future.result(timeout=10)

    assert failed is False
    assert pipeline.models is not previous_models
    np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.8)
    np.testing.assert_array_equal(pipeline.REF_ILLUMINANT, config.REF_ILLUMINANT)


def test_save_failure_preserves_context_and_caller_owned_data(tmp_path, monkeypatch):
    pipeline = ColorCorrection()
    pipeline.get_reference_values()
    pipeline.models.model_wb = np.eye(3) * 1.5
    pipeline.Image = np.full((2, 2, 3), 0.2)
    pipeline.White_Image = np.full((2, 2, 3), 240, dtype=np.uint8)
    pipeline.Models_path = "previous_models.pkl"
    old_context = {
        name: getattr(pipeline, name)
        for name in (
            "models",
            "Image",
            "White_Image",
            "Models_path",
            "REF_ILLUMINANT",
            "REFERENCE_CHART",
            "REFERENCE_RGB_PD",
            "REFERENCE_NEUTRAL_PATCHES_PD",
        )
    }

    def mutating_gamma(candidate, Image, do_gc, gc_kwargs):
        Image[0, 0] = 0.8
        gc_kwargs["nested"]["values"].append(2)
        candidate.models.model_gc = np.array([1.0, 0.0])
        return Image, {}, False

    def failed_export(candidate, metrics, name_, config):
        raise OSError("injected export failure")

    monkeypatch.setattr(ColorCorrection, "do_gamma_correction", mutating_gamma)
    monkeypatch.setattr(ColorCorrection, "_save_run_outputs", failed_export)
    image = np.full((2, 2, 3), 0.4)
    config = Config(
        do_ffc=False,
        do_gc=True,
        do_wb=False,
        do_cc=False,
        check_saturation=False,
        save=True,
        save_path=str(tmp_path),
        REF_ILLUMINANT=np.array([0.34567, 0.35850]),
        GC_kwargs={"nested": {"values": [1]}},
    )

    with pytest.raises(OSError, match="injected export failure"):
        pipeline.run(image, name_="candidate", config=config)

    for name, value in old_context.items():
        assert getattr(pipeline, name) is value
    np.testing.assert_array_equal(image, np.full((2, 2, 3), 0.4))
    assert config.GC_kwargs == {"nested": {"values": [1]}}
    assert pipeline.last_run["success"] is False
    assert any("injected export failure" in error["message"] for error in pipeline.last_run["errors"])
    np.testing.assert_allclose(pipeline.predict_image(image)["CC"], 0.6)


def test_incomplete_export_rollback_retains_recoverable_backups(tmp_path, monkeypatch):
    import ColorCorrectionPipeline.pipeline as pipeline_module

    pipeline = ColorCorrection()
    pipeline.get_reference_values()
    previous_models = pipeline.models
    previous_models.model_wb = np.eye(3) * 1.5
    old_metrics = b"previous metrics must remain recoverable"
    old_model = b"previous model must remain unchanged"
    metrics_path = tmp_path / "case_metrics.csv"
    model_path = tmp_path / "case_models.pkl"
    metrics_path.write_bytes(old_metrics)
    model_path.write_bytes(old_model)

    def fitted_gamma(candidate, Image, do_gc, gc_kwargs):
        candidate.models.model_gc = np.array([1.0, 0.0])
        return Image.copy(), {"value": 1.0}, False

    replace = pipeline_module.os.replace

    def failed_publish_and_restore(source, destination):
        if Path(destination) == model_path:
            raise OSError("injected model publication failure")
        if Path(destination) == metrics_path and Path(source).name.startswith("backup-"):
            raise OSError("injected metrics restoration failure")
        return replace(source, destination)

    monkeypatch.setattr(ColorCorrection, "do_gamma_correction", fitted_gamma)
    monkeypatch.setattr(pipeline_module.os, "replace", failed_publish_and_restore)
    config = Config(
        do_ffc=False,
        do_gc=True,
        do_wb=False,
        do_cc=False,
        check_saturation=False,
        save=True,
        save_path=str(tmp_path),
    )

    with pytest.raises(OSError, match="recovery files retained"):
        pipeline.run(np.full((2, 2, 3), 0.4), name_="case", config=config)

    recovery = pipeline.last_run["recovery"]
    recovery_directory = Path(recovery["directory"])
    assert recovery_directory.is_dir()
    assert Path(recovery["backups"]["case_metrics.csv"]).read_bytes() == old_metrics
    assert model_path.read_bytes() == old_model
    assert recovery["failed"][0]["file"] == "case_metrics.csv"
    assert pipeline.last_run["success"] is False
    assert pipeline.last_run["artifacts"] == []
    assert pipeline.models is previous_models


def test_cleanup_failure_does_not_undo_a_fully_published_run(tmp_path, monkeypatch):
    import shutil

    pipeline = ColorCorrection()

    def fitted_gamma(candidate, Image, do_gc, gc_kwargs):
        candidate.models.model_gc = np.array([1.0, 0.0])
        return Image.copy(), {"value": 1.0}, False

    remove_tree = shutil.rmtree

    def failed_cleanup(path, *args, **kwargs):
        if Path(path).parent == tmp_path and Path(path).name.startswith(".ccp-run-"):
            raise OSError("injected temporary cleanup failure")
        return remove_tree(path, *args, **kwargs)

    monkeypatch.setattr(ColorCorrection, "do_gamma_correction", fitted_gamma)
    monkeypatch.setattr(shutil, "rmtree", failed_cleanup)
    config = Config(
        do_ffc=False,
        do_gc=True,
        do_wb=False,
        do_cc=False,
        check_saturation=False,
        save=True,
        save_path=str(tmp_path),
    )

    _, _, failed = pipeline.run(np.full((2, 2, 3), 0.4), name_="case", config=config)

    assert failed is False
    assert pipeline.last_run["success"] is True
    assert (tmp_path / "case_models.pkl").is_file()
    assert (tmp_path / "case_metrics.csv").is_file()
    warning = pipeline.last_run["cleanup_warnings"][0]
    assert "injected temporary cleanup failure" in warning["message"]
    assert Path(warning["directory"]).is_dir()
