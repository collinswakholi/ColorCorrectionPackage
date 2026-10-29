"""Patch 5–6 contracts must hold through fit, prediction and persistence."""
import copy
import pickle
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData, ReferenceProfile


def config(**updates):
    settings = dict(do_ffc=False, color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1",
                    fitting_policy="reproducible-v1", chart_detection={"device": "cpu"},
                    CC_kwargs={"degree": 1})
    settings.update(updates)
    return Config(**settings)


def chart(monkeypatch, *, colors=None, faces=1):
    from ColorCorrectionPipeline.detection import charts
    if colors is None:
        colors = np.random.default_rng(28).uniform(.12, .8, (24, 3))
        colors[-6:] = np.repeat(np.linspace(.85, .15, 6)[:, None], 3, axis=1)
    source = np.zeros((80, 120 * faces, 3))
    geometry = []
    for face in range(faces):
        quads = []
        for i, rgb in enumerate(colors):
            row, col = divmod(i, 6)
            left, top = face * 120 + col * 20, row * 20
            source[top:top + 20, left:left + 20] = rgb * (1 - .1 * face)
            quads.append([[left+3., top+3.], [left+16., top+3.], [left+16., top+16.], [left+3., top+16.]])
        geometry.append({"type": 0, "score": .99 - face*.01, "patches": np.array(quads),
                         "corners": np.array([[face*120., 0], [face*120+119., 0], [face*120+119., 79], [face*120., 79]]), "dims": [20, 20]})
    monkeypatch.setattr(charts, "_forward", lambda *args: (geometry, {"provider": "fixture", "sha256": "0"*64}))
    profile = ReferenceProfile.from_values(colors, color_space="srgb", white="D65", scale=1,
                 profile_id="modern-fixture", revision=1, names=tuple("cell%d" % i for i in range(24)),
                 neutral_ids=tuple("r4c%d" % i for i in range(1, 7)))
    return source, profile


def test_modern_complete_fit_apply_reload_and_validation(monkeypatch, tmp_path):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    cfg = config(reference_profile=reference, stage_policy={"version": "modern-v1", "wb_headroom": 1},
                 save=True, save_path=str(tmp_path))
    metrics, images, failed = cc.run(image, name_="modern", config=cfg)
    assert not failed, cc.last_run
    prediction = cc.predict_image(image)
    for stage in ("GC", "WB", "CC"):
        np.testing.assert_allclose(prediction[stage], images["modern_"+stage], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(prediction["CC"], image, atol=1e-10, rtol=0)
    validation = cc.numerical_report["CC"]["validation"]
    assert validation["status"] == "available", validation
    assert validation["expected_groups"] == validation["scored_groups"] == 24
    assert validation["whole_pipeline"] is validation["independent_capture"] is False
    assert "CIEDE2000" in validation["metric"]
    assert validation["mean"] < 1e-9
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "modern_models")
    np.testing.assert_array_equal(restored.predict_image(image)["CC"], prediction["CC"])
    assert restored.models.metadata["stage_policy"]["wb_headroom"] == 1
    assert cfg.to_dict()["stage_policy"]["version"] == "modern-v1"
    assert Config.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert metrics["modern_CC"]["validation"] == validation


def test_saturation_never_darkens_modern_pixels_and_exclusion_uses_source(monkeypatch):
    image, reference = chart(monkeypatch)
    image[:20, :10] = 1
    cc = ColorCorrection()
    _, images, failed = cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False))
    assert not failed, cc.last_run
    report = cc.numerical_report
    assert report["saturation"]["pixel_transform"] == "identity"
    assert report["CC"]["selection"]["excluded_cell_ids"] == ["r1c1"]
    assert report["CC"]["validation"]["excluded_observations"] == 1
    source = cc.observations.observations[0]
    assert 0 < source.clipped_fractions[0] < 1
    assert np.all(source.measured_rgb[0] < .99)
    assert report["source_clipping"]["fractions_per_channel"][0][0] > 0
    np.testing.assert_allclose(cc.predict_image(image)["CC"], images["_CC"], atol=1e-12)


def test_modern_saturation_only_is_identity(monkeypatch):
    image, reference = chart(monkeypatch)
    image[:20, :20] = 1
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False, do_cc=False))[2]
    np.testing.assert_array_equal(cc.predict_image(image)["CC"], image)
    assert cc.numerical_report["source_clipping"]["fractions_any_channel"][0] == 1


def test_named_neutrals_only_gamma_ignores_colored_cells(monkeypatch):
    image, reference = chart(monkeypatch)
    cfg = config(reference_profile=reference, do_wb=False, do_cc=False)
    a = ColorCorrection()
    assert not a.run(image, config=cfg)[2], a.last_run
    changed = image.copy()
    changed[:60] = .23
    b = ColorCorrection()
    assert not b.run(changed, config=cfg)[2], b.last_run
    np.testing.assert_array_equal(a.models.model_gc.measured_knots, b.models.model_gc.measured_knots)
    np.testing.assert_array_equal(a.models.model_gc.target_knots, b.models.model_gc.target_knots)
    assert len(a.numerical_report["GC"]["used_cell_ids"]) == 6


def test_clipped_neutrals_do_not_become_usable_after_preprocessing(monkeypatch):
    image, reference = chart(monkeypatch)
    image[60:80, 0:80] = 1  # four of six neutrals source-clipped
    cc = ColorCorrection()
    old = cc.models
    assert cc.run(image, config=config(reference_profile=reference))[2]
    assert cc.models is old
    assert cc.last_run["stages"]["GC"]["status"] == "failed"
    assert "usable measured neutral" in cc.last_run["errors"][0]["message"]


def test_missing_chart_reports_failure_and_preserves_active_models(monkeypatch):
    from ColorCorrectionPipeline.detection import charts
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference))[2]
    old = cc.models
    monkeypatch.setattr(charts, "_forward", lambda *args: ([], {}))
    assert cc.run(image, config=config(reference_profile=reference))[2]
    assert cc.models is old
    assert cc.last_run["errors"]


def test_selected_second_face_is_used_by_all_new_fits(monkeypatch):
    image, reference = chart(monkeypatch, faces=2)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference, chart_detection={"device": "cpu", "chart_index": 1}))[2], cc.last_run
    assert cc._source_face.face_id == "classic-01"
    np.testing.assert_array_equal(cc.observations.observations[1].quadrilaterals, cc._source_face.quadrilaterals)


def test_modern_canonical_metric_uses_working_srgb_even_for_linear_input(monkeypatch):
    from ColorCorrectionPipeline.colorimetry import srgb_to_linear
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(srgb_to_linear(image), config=config(reference_profile=reference, do_gc=False, do_wb=False,
                  color_semantics={"version": "explicit-v1", "input_encoding": "linear-srgb", "output_encoding": "linear-srgb"}))[2], cc.last_run
    assert cc.numerical_report["CC"]["validation"]["mean"] < 1e-9


@pytest.mark.parametrize("updates,match", [
    ({"color_semantics": {"version": "legacy-v1"}}, "require explicit"),
    ({"stage_policy": "legacy-v1"}, "requires modern"),
    ({"fitting_policy": "legacy-v1"}, "requires reproducible"),
    ({"GC_kwargs": {"max_degree": 3}}, "polynomial degree"),
    ({"validation_options": {"n_splits": True}}, "integer"),
    ({"validation_options": {"enabled": 1}}, "boolean"),
    ({"validation_options": {"unknown": 3}}, "unknown"),
])
def test_new_policies_reject_ambiguous_or_ignored_settings(updates, match):
    with pytest.raises((ValueError, TypeError), match=match):
        config(**updates).validate()


def test_unsupported_legacy_sampling_control_fails_without_commit(monkeypatch):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert cc.run(image, config=config(reference_profile=reference, CC_kwargs={"n_samples": 50}))[2]
    assert not cc.models.has_models()
    assert "legacy sampling option" in cc.last_run["errors"][0]["message"]


def test_load_rejects_policy_model_mismatch_without_changing_active_state(monkeypatch, tmp_path):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference))[2]
    corrupt = copy.deepcopy(cc.models)
    corrupt.model_gc = np.array([1., 0.])
    with open(tmp_path / "bad.pkl", "wb") as stream:
        pickle.dump(corrupt, stream)
    before = cc.models.model_gc
    with pytest.raises(OSError, match="neutral-pchip"):
        cc.models.load(str(tmp_path), "bad")
    assert cc.models.model_gc is before


def test_modern_conv_exposes_root_polynomial_and_rejects_high_degree_rank(monkeypatch):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False,
        CC_kwargs={"cc_method": "conv", "degree": 2, "root_polynomial_expansion": True}))[2], cc.last_run
    assert cc.models.model_cc[1]["root_polynomial_expansion"] is True
    assert cc.numerical_report["CC"]["validation"]["status"] == "available"
    before = cc.models
    assert cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False,
        CC_kwargs={"degree": 4}))[2]
    assert cc.models is before


def test_explicit_weighted_anchor_assumptions_are_train_only(monkeypatch):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False,
        fitting_policy={"version": "reproducible-v1", "anchor_weight": .2}))[2], cc.last_run
    report = cc.numerical_report["CC"]
    assert report["fit"]["synthetic_training_groups"] == 2
    assert report["validation"]["synthetic_rows_train_only"] == 2
    assert report["validation"]["expected_groups"] == 24
    for fold in report["validation"]["folds"]:
        assert all(i < 24 for i in fold["test_indices"])
        assert 24 in fold["training_indices"] and 25 in fold["training_indices"]


def test_underidentified_validation_folds_do_not_invent_an_aggregate(monkeypatch):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    assert not cc.run(image, config=config(reference_profile=reference, do_gc=False, do_wb=False,
        CC_kwargs={"degree": 3}))[2], cc.last_run
    validation = cc.numerical_report["CC"]["validation"]
    assert validation["status"] == "unavailable"
    assert validation["mean"] is None
    assert validation["failed_groups"] == 24
    assert all("rank deficient" in fold["reason"] for fold in validation["folds"])


def white_gradient(shape):
    h, w = shape[:2]
    plane = np.rint(160 + 25 * np.linspace(0, 1, h)[:, None] + 15 * np.linspace(0, 1, w)[None, :]).astype(np.uint8)
    return np.repeat(plane[..., None], 3, axis=2)


def test_safe_ffc_pipeline_reuses_quality_cache_and_roundtrips(tmp_path):
    image = np.full((80, 120, 3), .4)
    white = white_gradient(image.shape)
    cfg = config(do_ffc=True, do_gc=False, do_wb=False, do_cc=False, check_saturation=False,
                 FFC_kwargs={"crop_rect": [0, 0, 120, 80], "bins": 8, "smooth_window": 3, "degree": 2},
                 save=True, save_path=str(tmp_path))
    cc = ColorCorrection()
    assert not cc.run(image, white, "ffc", cfg)[2], cc.last_run
    assert cc.numerical_report["FFC"]["safe_mode"] is True
    assert not cc.numerical_report["FFC"]["multiplier_cache_reused"]
    assert not cc.models.model_ffc.flags.writeable
    assert not cc.run(image, white, "ffc", cfg)[2], cc.last_run
    assert cc.numerical_report["FFC"]["multiplier_cache_reused"]
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "ffc_models")
    np.testing.assert_array_equal(restored.predict_image(image)["FFC"], cc.predict_image(image)["FFC"])
    with pytest.raises(ValueError, match="calibrated image dimensions"):
        restored.predict_image(np.full((40, 60, 3), .4))


@pytest.mark.parametrize("corruption", ["missing", "nan", "reversed", "shape", "negative"])
def test_safe_ffc_metadata_cannot_bypass_load_bounds(corruption, tmp_path):
    from ColorCorrectionPipeline import MyModels
    image = np.full((80, 120, 3), .4)
    cc = ColorCorrection()
    cfg = config(do_ffc=True, do_gc=False, do_wb=False, do_cc=False, check_saturation=False,
                 FFC_kwargs={"crop_rect": [0, 0, 120, 80], "bins": 8, "smooth_window": 3, "degree": 2})
    assert not cc.run(image, white_gradient(image.shape), config=cfg)[2], cc.last_run
    broken = copy.deepcopy(cc.models)
    quality = broken.metadata["numerical_report"]["FFC"]
    if corruption == "missing":
        quality.pop("gain_limits")
    elif corruption == "nan":
        quality["gain_limits"] = [float("nan"), float("nan")]
    elif corruption == "reversed":
        quality["gain_limits"] = [2, .5]
    elif corruption == "shape":
        broken.model_ffc = broken.model_ffc[:20]
    else:
        broken.model_ffc = -np.ones(image.shape[:2])
    with open(tmp_path / "corrupt.pkl", "wb") as stream:
        pickle.dump(broken, stream)
    models = MyModels()
    with pytest.raises(OSError, match="FFC"):
        models.load(str(tmp_path), "corrupt")
    assert not models.has_models()
