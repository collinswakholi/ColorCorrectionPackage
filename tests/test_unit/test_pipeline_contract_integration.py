"""Contracts must govern actual fitting, application and persisted interpretation."""

import copy
import pickle

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from ColorCorrectionPipeline import ColorCorrection, Config, ImageData, ReferenceProfile, ColorSemantics
from ColorCorrectionPipeline.colorimetry import D65, linear_to_srgb
from ColorCorrectionPipeline.io import read_image_data
from ColorCorrectionPipeline.models import MyModels


def explicit(**kwargs):
    values = dict(do_ffc=False, do_gc=False, do_wb=False, do_cc=False, check_saturation=False,
                  color_semantics={"version": "explicit-v1"}, chart_detection={"device": "cpu"})
    values.update(kwargs)
    return Config(**values)


def synthetic_chart(monkeypatch, colors=None, faces=1):
    from ColorCorrectionPipeline.detection import charts
    colors = np.random.default_rng(8).uniform(0.1, 0.8, (24, 3)) if colors is None else colors
    source = np.zeros((120, 180 * faces, 3), dtype=np.float64)
    geometry = []
    for face in range(faces):
        quads = []
        scale = 1 if faces == 1 else (0.7 if face == 0 else 0.9)
        for index, rgb in enumerate(colors):
            row, col = divmod(index, 6)
            left, top = face * 180 + col * 30, row * 30
            source[top:top+30, left:left+30] = rgb * scale
            quads.append([[left+5., top+5.], [left+24., top+5.], [left+24., top+24.], [left+5., top+24.]])
        geometry.append({"type": 0, "score": 0.99-face*0.01, "patches": np.asarray(quads),
                         "corners": np.array([[face*180., 0], [face*180+179., 0], [face*180+179., 119], [face*180., 119]]),
                         "dims": [30, 30]})
    monkeypatch.setattr(charts, "_forward", lambda *a: (geometry, {"provider": "fixture", "sha256": "0"*64}))
    return source, colors


def profile_from_rgb(rgb, neutral_ids=("r4c1", "r4c2", "r4c3", "r4c4", "r4c5", "r4c6")):
    return ReferenceProfile.from_values(rgb, color_space="srgb", white="D65", scale=1,
                                       profile_id="test-profile", revision=1, names=tuple("cell-%d" % i for i in range(24)),
                                       neutral_ids=neutral_ids)


def test_uint16_image_data_runs_and_predicts_without_quantizing():
    values = np.arange(48, dtype=np.uint16).reshape(4, 4, 3) + 12345
    image = ImageData.from_array(values)
    cc = ColorCorrection()
    assert cc.run(image, config=explicit())[2] is False
    np.testing.assert_array_equal(cc.Image, values / 65535)
    np.testing.assert_array_equal(cc.predict_image(image)["CC"], values / 65535)
    assert cc.image_data.source_dtype == "uint16"
    assert "pixels" not in cc.models.metadata["input_image"]
    assert cc.observations.unavailable_reason


def test_explicit_plain_integer_input_is_normalized_but_legacy_rejects():
    values = np.full((4, 4, 3), 32769, dtype=np.uint16)
    cc = ColorCorrection()
    assert cc.run(values, config=explicit())[2] is False
    np.testing.assert_array_equal(cc.Image, values / 65535)
    with pytest.raises(TypeError):
        cc.run(values, config=Config(do_ffc=False, do_gc=False, do_wb=False, do_cc=False, check_saturation=False))


def test_saved_input_and_output_encoding_survive_new_pipeline_instance(tmp_path):
    values = np.linspace(0.001, 0.8, 48).reshape(4, 4, 3)
    config = explicit(color_semantics={"version": "explicit-v1", "input_encoding": "linear-srgb", "output_encoding": "linear-srgb"}, save=True, save_path=str(tmp_path))
    cc = ColorCorrection()
    assert cc.run(values, name_="encoding", config=config)[2] is False
    np.testing.assert_allclose(cc.Image, linear_to_srgb(values), atol=1e-15, rtol=0)
    np.testing.assert_allclose(cc.predict_image(values)["CC"], values, atol=1e-15, rtol=0)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "encoding_models")
    np.testing.assert_array_equal(restored.predict_image(values)["CC"], cc.predict_image(values)["CC"])


def test_typed_encoding_is_self_described():
    image = ImageData.from_array(np.full((4, 4, 3), 0.2), color_space="linear-srgb")
    cc = ColorCorrection()
    assert cc.run(image, config=explicit())[2] is False
    np.testing.assert_allclose(cc.Image, linear_to_srgb(image.pixels), atol=1e-15)
    assert cc.models.metadata["input_image"]["color_space"] == "linear-srgb"


def test_white_encoding_is_consistent_for_file_byte_float_and_typed_inputs(tmp_path, monkeypatch):
    class UnitGain:
        def __init__(self, image, **kwargs):
            self.image = image

        def compute_multiplier(self, **kwargs):
            return np.ones(self.image.shape[:2])

    monkeypatch.setattr("ColorCorrectionPipeline.pipeline.FlatFieldCorrection", UnitGain)
    white = np.full((4, 4, 3), 51, dtype=np.uint8)
    path = tmp_path / "linear-white.png"
    assert cv2.imwrite(str(path), white)
    variants = [white, white.astype(float)/255, path, ImageData.from_array(white, color_space="linear-srgb")]
    matrices = []
    config = explicit(do_ffc=True, FFC_kwargs={"get_deltaE": False}, color_semantics={"version": "explicit-v1", "input_encoding": "linear-srgb"})
    for variant in variants:
        cc = ColorCorrection()
        assert not cc.run(np.full((4, 4, 3), 0.2), White_Image=variant, config=config)[2], cc.last_run
        assert cc.white_image_data.color_space == "linear-srgb"
        matrices.append(cc.White_Image)
    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, matrices[0])
    with pytest.raises(ValueError, match="calibrated image dimensions"):
        cc.predict_image(np.full((8, 8, 3), 0.2))


def test_pending_orientation_requires_explicit_normalization():
    image = ImageData.from_array(np.zeros((4, 6, 3), dtype=np.uint8), orientation=6, apply_orientation=False)
    cc = ColorCorrection()
    with pytest.raises(ValueError, match="orientation"):
        cc.run(image, config=explicit())
    assert cc.run(image.with_orientation_applied(), config=explicit())[2] is False
    assert cc.Image.shape == (6, 4, 3)
    assert cc.image_data.source_orientation == 6


def test_custom_reference_truth_is_not_replaced_by_display_rgb():
    base = ReferenceProfile.classic24()
    lab = base.canonical_lab_d50.copy()
    lab[0] = [65, 120, -110]
    custom = ReferenceProfile("out-of-gamut", 1, base.names, lab, base.neutral_ids)
    cc = ColorCorrection()
    assert cc.run(np.full((4, 4, 3), 0.3), config=explicit(reference_profile=custom))[2] is False
    np.testing.assert_array_equal(cc.reference_profile.canonical_lab_d50, lab)
    assert np.any((cc.REFERENCE_RGB_PD.values < 0) | (cc.REFERENCE_RGB_PD.values > 1))
    assert cc.REFERENCE_DISPLAY_RGB_PD.values.min() >= 0 and cc.REFERENCE_DISPLAY_RGB_PD.values.max() <= 1


def test_fitting_uses_float_observations_and_metrics_use_canonical_truth(monkeypatch, tmp_path):
    source, _ = synthetic_chart(monkeypatch)
    source16 = np.round(source * 65535).astype(np.uint16)
    cc = ColorCorrection()
    config = explicit(do_cc=True, save=True, save_path=str(tmp_path),
                      CC_kwargs={"degree": 1, "n_samples": 4, "augment_extremes": False, "use_lut": False})
    _, images, failed = cc.run(ImageData.from_array(source16), name_="precision", config=config)
    assert not failed, cc.last_run
    original = cc.observations.observations[0]
    expected = source16[15::30, 15::30].reshape(24, 3) / 65535
    # An 81-sample float64 mean has accumulation rounding far below one uint16 step.
    np.testing.assert_allclose(original.measured_rgb, expected, atol=2e-14, rtol=0)
    assert np.max(np.abs(expected*255 - np.round(expected*255))) > 0.01
    corrected = cc.stage_observations["CC"].observations[0]
    metric = cc.color_semantics.delta_e00(corrected.measured_rgb, cc.reference_profile.canonical_lab_d50, input_encoding="srgb")
    assert cc.last_run["success"]
    assert np.isfinite(metric).all()
    actual_metrics = cc._explicit_metrics(cc.observations, cc.stage_observations["CC"], 0)
    assert actual_metrics["delta_e00_after_mean"] == pytest.approx(metric.mean(), abs=1e-12)
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "precision_models")
    np.testing.assert_allclose(restored.predict_image(ImageData.from_array(source16))["CC"], images["precision_CC"], atol=1e-12, rtol=0)


def test_declared_neutral_ids_drive_wb_instead_of_hardcoded_last_six(monkeypatch):
    colors = np.random.default_rng(9).uniform(0.1, 0.7, (24, 3))
    colors[0:2] = [[0.3, 0.35, 0.4], [0.5, 0.55, 0.6]]
    source, _ = synthetic_chart(monkeypatch, colors)
    target = colors.copy()
    target[0:2] *= 1.2
    target[-6:] *= 0.7
    profile = profile_from_rgb(target, neutral_ids=("r1c1", "r1c2"))
    cc = ColorCorrection()
    assert not cc.run(ImageData.from_array(source), config=explicit(do_wb=True, reference_profile=profile, WB_kwargs={"get_deltaE": False}))[2], cc.last_run
    expected = np.median(cc.REFERENCE_RGB_PD.values[:2] / colors[:2], axis=0)*0.95
    np.testing.assert_allclose(np.diag(cc.models.model_wb), expected, atol=1e-12)


def test_selected_face_matches_extended_fit_and_observation_identity(monkeypatch):
    source, target = synthetic_chart(monkeypatch, faces=2)
    profile = profile_from_rgb(target)
    cc = ColorCorrection()
    config = explicit(do_cc=True, reference_profile=profile, chart_detection={"device": "cpu", "chart_index": 1},
                      CC_kwargs={"degree": 1, "n_samples": 4, "augment_extremes": False, "use_lut": False})
    metrics, _, failed = cc.run(ImageData.from_array(source), name_="faces", config=config)
    assert not failed, cc.last_run
    assert len(cc.observations.observations) == 2
    assert metrics["faces_CC"]["selected_face_id"] == "classic-01"
    selected = cc.stage_observations["CC"].observations[1]
    np.testing.assert_allclose(selected.measured_rgb, cc.REFERENCE_RGB_PD.values, atol=1e-12, rtol=0)
    assert selected.cell_ids == profile.cell_ids


@pytest.mark.parametrize("change", ["reference", "source", "schema"])
def test_corrupt_contract_metadata_does_not_replace_active_models(tmp_path, change):
    cc = ColorCorrection()
    assert not cc.run(np.full((4, 4, 3), 0.2), config=explicit())[2]
    bad = copy.deepcopy(cc.models)
    if change == "reference":
        bad.metadata["reference_profile"]["revision"] = "forged"
    elif change == "source":
        bad.metadata["input_image"]["digest"] = "f"*64
    else:
        bad.metadata["schema"] = True
    (tmp_path / "bad.pkl").write_bytes(pickle.dumps(bad))
    previous = MyModels()
    previous.model_wb = np.eye(3)*0.5
    old = previous.model_wb
    with pytest.raises(OSError):
        previous.load(str(tmp_path), "bad")
    assert previous.model_wb is old


@pytest.mark.parametrize("kwargs", [
    {"color_semantics": {"version": "explicit-v1", "output_policy": "preserve"}},
    {"color_semantics": {"version": "explicit-v1"}, "REF_ILLUMINANT": np.array([0.3457, 0.3585])},
    {"color_semantics": {"version": "explicit-v1"}, "chart_detection": {"backend": "mcc"}},
    {"color_semantics": {"version": "legacy-v1", "adaptation": "Bradford"}},
    {"reference_profile": ReferenceProfile.classic24()},
])
def test_unimplemented_or_ambiguous_color_requests_fail_before_fitting(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()


def test_explicit_file_reader_preserves_uint16_and_orientation_once(tmp_path):
    pixels = np.arange(72, dtype=np.uint16).reshape(4, 6, 3) + 23456
    png = tmp_path / "precision.png"
    assert cv2.imwrite(str(png), pixels[:, :, ::-1])
    decoded = read_image_data(png)
    assert decoded.source_dtype == "uint16"
    np.testing.assert_array_equal(decoded.pixels, pixels / 65535)
    jpeg = tmp_path / "oriented.jpg"
    image = Image.fromarray((pixels % 255).astype(np.uint8))
    exif = Image.Exif(); exif[274] = 6
    image.save(jpeg, exif=exif)
    oriented = read_image_data(jpeg)
    raw = cv2.imdecode(np.frombuffer(jpeg.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    np.testing.assert_array_equal(oriented.pixels, np.rot90(raw, 3) / 255)
    np.testing.assert_array_equal(oriented.with_orientation_applied().pixels, oriented.pixels)


def test_embedded_icc_requires_a_recorded_decision(tmp_path):
    path = tmp_path / "profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(np.full((4, 4, 3), 80, dtype=np.uint8)).save(path, icc_profile=profile)
    with pytest.raises(ValueError, match="ICC"):
        read_image_data(path)
    decoded = read_image_data(path, icc_policy="assume-declared")
    assert decoded.provenance["icc_present"] is True
    assert decoded.provenance["icc_transform"] == "none"
    assert decoded.provenance["icc_policy"] == "assume-declared"


def test_explicit_gamma_uses_canonical_lightness_and_is_global_scale_independent(monkeypatch):
    import colour
    import pandas as pd
    import ColorCorrectionPipeline.core.correction as correction

    profile = ReferenceProfile.classic24()
    canonical = profile.canonical_lab_d50.copy()
    canonical[0] = [65, 120, -110]
    semantics = ColorSemantics(version="explicit-v1")
    targets = semantics.reference_to_working_rgb(canonical)
    reference_l = semantics.reference_to_working_lab(canonical)[:, 0]
    measured = profile.display_rgb()
    monkeypatch.setattr(correction, "extract_neutral_patches", lambda *a, **k: (pd.DataFrame(measured[-6:]), pd.DataFrame(measured)))
    original_fit = correction.estimate_fit
    observed_targets = []

    def fit(x, y, degree):
        observed_targets.append(y.copy())
        return original_fit(x, y, degree)

    monkeypatch.setattr(correction, "estimate_fit", fit)
    image = np.random.default_rng(44).uniform(0.1, 0.8, (4, 4, 3))
    coeffs, first, _ = correction.estimate_gamma_profile(image, targets, np.array(D65), max_degree=2,
                                                        explicit_color=True, reference_lightness=reference_l)
    np.testing.assert_array_equal(observed_targets[0][1:-1], reference_l)
    with colour.utilities.domain_range_scale("100"):
        second_coeffs, second, _ = correction.estimate_gamma_profile(image, targets, np.array(D65), max_degree=2,
                                                                    explicit_color=True, reference_lightness=reference_l)
        identity = correction.predict_image(image, np.array([1., 0.]), np.array(D65), explicit_color=True)
    np.testing.assert_array_equal(coeffs, second_coeffs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(identity, image, atol=2e-14, rtol=0)
