"""Unit tests for pipeline configuration validation."""

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config


def test_run_validates_config_before_processing():
    pipeline = ColorCorrection()
    image = np.zeros((8, 8, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="save_path"):
        pipeline.run(image, config=Config(save=True, save_path=None))


def test_run_saves_models_to_expected_pickle_path(tmp_path, monkeypatch):
    pipeline = ColorCorrection()
    image = np.zeros((8, 8, 3), dtype=np.float64)

    monkeypatch.setattr(
        pipeline,
        "do_flat_field_correction",
        lambda Image, do_ffc, ffc_kwargs: (Image, {"stage": "ffc"}, False),
    )
    monkeypatch.setattr(
        pipeline,
        "do_gamma_correction",
        lambda Image, do_gc, gc_kwargs: (Image, {"stage": "gc"}, False),
    )
    monkeypatch.setattr(
        pipeline,
        "do_white_balance",
        lambda Image, do_wb, wb_kwargs: (Image, {"stage": "wb"}, False),
    )
    monkeypatch.setattr(
        pipeline,
        "do_color_correction",
        lambda Image, do_cc, cc_method, cc_kwargs: (Image, {"stage": "cc"}, False),
    )

    pipeline.run(
        image,
        name_="case",
        config=Config(save=True, save_path=str(tmp_path), check_saturation=False),
    )

    assert (tmp_path / "case_models.pkl").is_file()
    assert not (tmp_path / "case_models.pkl" / "models.pkl").exists()