"""Opt-in linear-light FFC domain: equations, validation, persistence and replay."""

import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import (
    CCOptions,
    ColorCorrection,
    Config,
    FFCOptions,
    GCOptions,
    Steps,
    WBOptions,
)
from ColorCorrectionPipeline.colorimetry import (
    D65,
    delta_e00,
    linear_to_srgb,
    rgb_to_xyz,
    srgb_to_linear,
    xyz_to_lab,
)
from ColorCorrectionPipeline import ui_native

IMAGES = Path(__file__).resolve().parents[1] / "Images"
CHART, WHITE, SCENE = (IMAGES / n for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))


def read_manifest(path):
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def _lab(rgb):
    return xyz_to_lab(rgb_to_xyz(np.clip(rgb, 0, 1), encoding="srgb"), white=D65)


def _l_star(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > ui_native.EPSILON, 116.0 * np.cbrt(y) - 16.0, ui_native.KAPPA * y)


@pytest.mark.parametrize("falloff", [0.2, 0.4])
def test_linear_domain_removes_vignetting_chroma_error_with_ideal_gain(falloff):
    # A Classic24 chart under linear-light vignetting, corrected with the gain
    # each domain would derive from a vignetted neutral white plane (Y=0.9).
    truth = np.clip(ui_native.reference(ui_native.default_reference())[1], 0, 1)
    factor = 1.0 - falloff * np.linspace(0, 1, len(truth))
    observed = linear_to_srgb(srgb_to_linear(truth) * factor[:, None], policy="clip")
    lightness_gain = _l_star(0.9) / _l_star(0.9 * factor)
    errors = {}
    for domain, gain in (("lightness", lightness_gain), ("linear", 1.0 / factor)):
        corrected = ui_native.apply_color(observed, gain, rounded=False, domain=domain)
        errors[domain] = delta_e00(_lab(corrected), _lab(truth))
    print(
        "falloff %.1f: dE00 mean lightness %.3f linear %.3f"
        % (falloff, errors["lightness"].mean(), errors["linear"].mean())
    )
    assert errors["linear"].mean() < 0.05
    assert errors["lightness"].mean() > 10 * errors["linear"].mean()


def test_lightness_is_the_default_and_matches_the_historical_equation():
    rng = np.random.default_rng(3)
    image = rng.random((5, 7, 3))
    gain = 0.8 + 0.4 * rng.random((5, 7))
    historical = ui_native.lab(image)
    historical[..., 0] = np.clip(historical[..., 0] * gain, 0, 100)
    expected = np.clip(
        ui_native._encode(
            ui_native._multiply(
                ui_native.XYZ_TO_RGB, ui_native._lab_to_xyz(historical, ui_native.D65)
            )
        ),
        0,
        1,
    )
    np.testing.assert_array_equal(ui_native.apply_color(image, gain, rounded=False), expected)
    np.testing.assert_array_equal(
        ui_native.apply_color(image, gain, rounded=False, domain="lightness"), expected
    )


@pytest.mark.parametrize(
    "request_",
    [
        lambda: Config(steps=Steps(ffc=FFCOptions(apply={"domain": "rgb"}))),
        lambda: Config(steps=Steps(ffc=FFCOptions(apply={"domain": 1}))),
        lambda: Config(steps=Steps(ffc=FFCOptions(apply={"unknown": "linear"}))),
        lambda: Config(fitting_policy="legacy-v1", FFC_kwargs={"domain": "linear"}),
        lambda: Config.from_dict({"FFC_kwargs": {"domain": "LINEAR"}}, source_version=2),
    ],
)
def test_invalid_domain_is_rejected(request_):
    with pytest.raises((TypeError, ValueError)):
        request_().validate()


def test_domain_round_trips_through_flat_and_grouped_configuration():
    config = Config(steps=Steps(ffc=FFCOptions(apply={"domain": "linear"})))
    assert config.FFC_kwargs["domain"] == "linear"
    grouped = config.to_grouped_dict()
    assert grouped["steps"]["ffc"]["apply"] == {"domain": "linear"}
    assert Config.from_dict(grouped).FFC_kwargs["domain"] == "linear"
    assert Config.from_dict(config.to_dict()).FFC_kwargs["domain"] == "linear"
    default = Config()
    assert "domain" not in default.FFC_kwargs
    assert "apply" not in default.to_grouped_dict().get("steps", {}).get("ffc", {})


def _vignette(path, falloff):
    rgb = cv2.imread(str(path))[:, :, ::-1] / 255.0
    h, w = rgb.shape[:2]
    y, x = np.mgrid[:h, :w]
    radius = np.hypot(x - (w - 1) / 2, y - (h - 1) / 2)
    factor = 1.0 - falloff * (radius / radius.max()) ** 2
    linear = srgb_to_linear(rgb) * factor[..., None]
    return np.rint(linear_to_srgb(linear, policy="clip") * 255).astype(np.uint8)


def _ffc_only(domain, *, precision="auto", **ffc):
    apply = {} if domain is None else {"apply": {"domain": domain}}
    return Config(
        steps=Steps(
            ffc=FFCOptions(enabled=True, quality={"max_gain": 3.0}, **apply, **ffc),
            gc=GCOptions(enabled=False),
            wb=WBOptions(enabled=False),
            cc=CCOptions(enabled=False),
        ),
        chart={"device": "cpu"},
        input={"frame_index": 0},
        numerics={"precision": precision},
    )


def _fit(chart, white, config):
    pipe = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, failed = pipe.run(str(chart), str(white), name_="chart", config=config)
    assert not failed, pipe.last_run["errors"]
    return pipe


@pytest.fixture(scope="module")
def vignetted(tmp_path_factory):
    root = tmp_path_factory.mktemp("vignette")
    paths = []
    for source in (CHART, WHITE):
        path = root / (source.stem + ".png")
        cv2.imwrite(str(path), _vignette(source, 0.4)[:, :, ::-1])
        paths.append(path)
    return paths


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["auto", "high"])
def test_fitted_linear_domain_reduces_residual_vignetting_error(vignetted, precision):
    residual = {}
    for domain in ("lightness", "linear"):
        clean = _fit(CHART, WHITE, _ffc_only(domain, precision=precision))
        shaded = _fit(*vignetted, _ffc_only(domain, precision=precision))
        a = clean.stage_observations["FFC"].observations[0].measured_rgb
        b = shaded.stage_observations["FFC"].observations[0].measured_rgb
        raw = delta_e00(_lab(a), _lab(b))
        # The white-plane peak normalization differs between the two captures;
        # remove that single global exposure factor to isolate spatial error.
        la, lb = srgb_to_linear(a), srgb_to_linear(b)
        scale = float((la * lb).sum() / (lb * lb).sum())
        spatial = delta_e00(_lab(a), _lab(linear_to_srgb(lb * scale, policy="clip")))
        residual[domain] = (raw.mean(), spatial.mean())
        assert shaded.models.metadata.get("ffc_domain") == (
            "linear" if domain == "linear" else None
        )
    print(precision, {k: tuple(round(float(v), 3) for v in item) for k, item in residual.items()})
    assert residual["linear"][0] < residual["lightness"][0]
    assert residual["linear"][1] < 0.6 * residual["lightness"][1]


@pytest.fixture(scope="module")
def linear_fit():
    return _fit(CHART, WHITE, _ffc_only("linear"))


@pytest.mark.slow
def test_linear_domain_is_recorded_and_replayed_by_bundles_and_pickles(linear_fit, tmp_path):
    pipe = linear_fit
    assert pipe.models.metadata["ffc_domain"] == "linear"
    assert "linear-light" in pipe.models.metadata["algorithms"]["FFC"]
    expected = pipe.predict_image(str(SCENE), stages="FFC")["FFC"]
    # The replayed result is the linear-light rule, not the L* gain.
    lightness = ui_native.apply_color(
        ui_native.normalize_codes(cv2.imread(str(SCENE))[:, :, ::-1] / 255.0),
        pipe.models.model_ffc,
    )
    replayed = np.asarray(expected, dtype=np.float64)
    if np.asarray(expected).dtype.kind in "ui":
        replayed = replayed / 255.0
    assert np.abs(replayed - lightness).max() > 1e-2

    path = tmp_path / "linear.ccpb"
    pipe.save_bundle(path)
    assert read_manifest(path)["stages"]["FFC"]["kind"] == "spatial-linear-gain-v1"
    loaded = ColorCorrection().load_bundle(path)
    assert loaded.models.metadata["ffc_domain"] == "linear"
    np.testing.assert_array_equal(loaded.predict_image(str(SCENE), stages="FFC")["FFC"], expected)

    pipe.models.save(str(tmp_path), "linear")
    restored = ColorCorrection()
    restored.models.load(str(tmp_path), "linear")
    np.testing.assert_array_equal(restored.predict_image(str(SCENE), stages="FFC")["FFC"], expected)

    # A recorded domain that contradicts the stored gain kind fails closed.
    tampered = pickle.loads(pickle.dumps(pipe.models))
    tampered.metadata["ffc_domain"] = "chroma"
    with pytest.raises(ValueError, match="FFC domain"):
        tampered._validated_state(tampered)


@pytest.mark.slow
def test_default_bundle_records_no_domain_and_replays_lightness(tmp_path):
    pipe = _fit(CHART, WHITE, _ffc_only(None))
    assert "ffc_domain" not in pipe.models.metadata
    expected = pipe.predict_image(str(SCENE), stages="FFC")["FFC"]
    path = tmp_path / "default.ccpb"
    pipe.save_bundle(path)
    manifest = read_manifest(path)
    assert manifest["stages"]["FFC"]["kind"] == "spatial-Lab-gain-v1"
    assert "ffc_domain" not in manifest["metadata"]
    loaded = ColorCorrection().load_bundle(path)
    assert ui_native.ffc_domain(loaded.models) == "lightness"
    np.testing.assert_array_equal(loaded.predict_image(str(SCENE), stages="FFC")["FFC"], expected)
    # Explicitly requesting the default changes neither the fit nor its record.
    explicit = _fit(CHART, WHITE, _ffc_only("lightness"))
    assert "ffc_domain" not in explicit.models.metadata
    np.testing.assert_array_equal(explicit.models.model_ffc, pipe.models.model_ffc)
    np.testing.assert_array_equal(explicit.predict_image(str(SCENE), stages="FFC")["FFC"], expected)
