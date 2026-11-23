"""Diagnostics go through the ``ColorCorrectionPipeline`` logger, not stdout."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

import ColorCorrectionPipeline
from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import read_batch, write_image

IMAGES = Path(__file__).resolve().parents[1] / "Images"
CHART, WHITE, SCENE = (str(IMAGES / n) for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))


def test_package_logger_has_null_handler():
    root = logging.getLogger("ColorCorrectionPipeline")
    assert ColorCorrectionPipeline.__name__ == root.name
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)


@pytest.fixture(scope="module")
def fitted():
    cc = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc.fit(CHART, name="chart", config=Config.from_preset("ccm", white=False, device="cpu"))
    return cc


def test_default_predict_writes_nothing_to_stdout(fitted, capsys):
    capsys.readouterr()
    fitted.predict(SCENE)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_default_fit_with_flat_field_writes_nothing_to_stdout(capsys):
    # Flat-field fitting used to print the detected white plane and fit method.
    cc = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc.fit(CHART, WHITE, name="chart", config=Config.from_preset("ccm", device="cpu"))
    cc.predict(SCENE)
    assert capsys.readouterr().out == ""


def test_prediction_timing_is_logged(fitted, caplog):
    with caplog.at_level(logging.INFO, logger="ColorCorrectionPipeline"):
        fitted.predict(SCENE)
    records = [r for r in caplog.records if r.name == "ColorCorrectionPipeline.pipeline"]
    assert any(r.getMessage().startswith("Prediction done in") for r in records)
    assert all(r.levelno == logging.INFO for r in records)


def test_read_batch_logs_progress_and_failures(tmp_path, caplog, capsys):
    good = tmp_path / "ok.png"
    assert write_image(good, np.zeros((4, 4, 3)))
    with caplog.at_level(logging.INFO, logger="ColorCorrectionPipeline.io"):
        images, failed = read_batch([good, tmp_path / "missing.png"])
    assert len(images) == 1 and len(failed) == 1
    messages = {(r.levelno, r.getMessage().split(":")[0]) for r in caplog.records}
    assert (logging.INFO, "Successfully read 1/2 images") in messages
    assert (logging.WARNING, "Failed to read 1 images") in messages
    assert capsys.readouterr().out == ""
