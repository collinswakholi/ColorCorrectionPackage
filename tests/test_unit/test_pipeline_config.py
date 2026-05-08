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


def test_color_correction_does_not_store_missing_custom_model(monkeypatch):
    pipeline = ColorCorrection()
    pipeline.get_reference_values()
    image = np.zeros((8, 8, 3), dtype=np.float64)

    monkeypatch.setattr(
        "ColorCorrectionPipeline.pipeline.color_correction",
        lambda **kwargs: (None, kwargs["img_rgb"], None, {}),
    )

    img_out, metrics, err = pipeline.do_color_correction(
        image,
        do_cc=True,
        cc_method="ours",
        cc_kwargs={"get_deltaE": False},
    )

    assert err is True
    assert pipeline.models.model_cc is None
    np.testing.assert_array_equal(img_out, image)
    assert metrics == {}


def test_flat_field_multiplier_cache_reuses_matching_white_image(monkeypatch):
    import ColorCorrectionPipeline.pipeline as pipeline_module

    calls = {"compute": 0}

    class FakeFlatFieldCorrection:
        def __init__(self, img, **kwargs):
            self.img = img

        def compute_multiplier(self, **kwargs):
            calls["compute"] += 1
            return np.ones(self.img.shape[:2], dtype=np.float64)

        def show_results(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        pipeline_module,
        "FlatFieldCorrection",
        FakeFlatFieldCorrection,
    )

    pipeline = ColorCorrection()
    pipeline.White_Image = np.full((8, 8, 3), 255, dtype=np.uint8)
    image = np.full((8, 8, 3), 0.5, dtype=np.float64)
    kwargs = {"get_deltaE": False, "bins": 8, "fit_method": "linear"}

    first, _, err_first = pipeline.do_flat_field_correction(image, True, kwargs)
    second, _, err_second = pipeline.do_flat_field_correction(image, True, kwargs)

    assert err_first is False
    assert err_second is False
    assert calls["compute"] == 1
    np.testing.assert_allclose(first, second)
