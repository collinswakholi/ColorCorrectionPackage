"""Qualify v2 selectors and numerical contracts from an installed distribution.

Run from a neutral directory with the built package installed:
  python -I /path/to/scripts/verify_v2.py --output qualification.json --photos

The UI reference engine is not a runtime dependency of these checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Event
import time
from unittest.mock import patch
import zipfile

import numpy as np

from ColorCorrectionPipeline import CalibrationError, Config, ColorCorrection, ImageData
from ColorCorrectionPipeline import ui_fitting
from ColorCorrectionPipeline.colorimetry import srgb_to_linear
from ColorCorrectionPipeline.examples._common import sample_paths, json_value
from ColorCorrectionPipeline.io import read_image_data, write_image_data


def expect_invalid(**options):
    try:
        Config(**options).validate()
    except (TypeError, ValueError):
        return
    raise AssertionError("Invalid configuration was accepted: " + repr(options))


def numerical_contracts():
    from colour.characterisation import polynomial_expansion_Finlayson2015

    rng = np.random.default_rng(20261106)
    rgb = np.vstack((np.zeros((1, 3)), np.eye(3), np.ones((1, 3)),
                     rng.uniform(.002, .98, (150, 3))))
    encoded = rng.uniform(.02, .97, (48, 3))
    target = srgb_to_linear(encoded) * .88 + .015
    counts = {"polynomial": (3, 9, 19, 34), "root": (3, 6, 13, 22),
              "hybrid": (3, 12, 29, 53)}
    rows = []
    for model in ("linear", "pls", "nn"):
        for basis, dimensions in counts.items():
            for degree in range(1, 5):
                requested = dict(model=model, basis=basis, degree=degree)
                if model == "pls":
                    requested["ncomp"] = 3
                if model == "nn":
                    requested.update(hidden_layers=[8], max_iterations=30, patience=10)
                fitted, fit_report = ui_fitting.fit_ui(encoded, target, requested)
                expected = dimensions[degree - 1] + int(model in ("pls", "nn"))
                assert fit_report["feature_count"] == expected, (requested, fit_report)
                prediction = fitted.predict_rgb(encoded[:8])
                assert prediction.shape == (8, 3) and np.isfinite(prediction).all()
                if basis == "root" and model == "linear":
                    actual = ui_fitting.features(rgb, ui_fitting.options(requested))
                    independent = polynomial_expansion_Finlayson2015(
                        rgb, degree=degree, root_polynomial_expansion=True)
                    distance = np.max(np.abs(actual[:, :, None] - independent[:, None, :]), axis=0)
                    matching = np.argmin(distance, axis=1)
                    assert len(set(matching)) == expected
                    assert np.max(distance[np.arange(expected), matching]) < 1e-13
                    np.testing.assert_allclose(
                        ui_fitting.features(rgb * .37, ui_fitting.options(requested)),
                        actual * .37, rtol=1e-13, atol=1e-14)
                if model == "nn":
                    zero = ui_fitting.UIPredictor(
                        dict(fitted.settings),
                        weights=[np.zeros_like(weight) for weight in fitted.weights],
                        biases=[np.zeros_like(bias) for bias in fitted.biases])
                    np.testing.assert_allclose(zero.predict_rgb(encoded[:8]),
                                               encoded[:8], rtol=0, atol=2e-15)
                rows.append({"model": model, "basis": basis, "degree": degree,
                             "features": expected})
    for degree, preset in ((1, "ccm"), (2, "rpcc2"), (3, "rpcc3")):
        a, _ = ui_fitting.fit_ui(encoded, target, dict(model="linear", basis="root", degree=degree))
        b, _ = ui_fitting.fit_ui(encoded, target, dict(model=preset))
        np.testing.assert_array_equal(a.predict_rgb(encoded), b.predict_rgb(encoded))
    return rows


def configurations():
    cfg = Config()
    cfg.validate()
    assert cfg.fitting_policy["version"] == "ui-compatible-v1"
    assert cfg.fitting_policy["ui_runtime"] == "auto-v2"
    old = Config(fitting_policy="legacy-v1", CC_kwargs={"mtd": "custom", "degree": 2})
    old.validate()
    assert Config.from_dict(old.to_dict()).fitting_policy["version"] == "legacy-v1"
    new = Config(model="pls", basis="root", degree=3, CC_kwargs={"ncomp": 3})
    new.validate()
    replay = Config.from_dict(new.to_dict())
    assert replay.to_dict() == new.to_dict()
    for kwargs in (
        dict(model="missing"), dict(model="nn", degree=True),
        dict(model="linear", degree=0), dict(model="linear", degree=5),
        dict(model="rpcc2", degree=3),
        dict(model="pls", basis="root", degree=3),
        dict(model="pls", basis="hybrid", degree=4, CC_kwargs={"ncomp": 0}),
        dict(model="linear", precision="float32"),
        dict(fitting_policy={"version": "ui-compatible-v1", "ui_runtime": "precision-f64-v1"},
             precision="native-byte"),
    ):
        expect_invalid(**kwargs)
    mutated = Config(precision="native-byte")
    mutated.precision = "high"
    try:
        mutated.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Mutated precision contradicts the stored byte runtime")
    return {"default": cfg.to_dict(), "historical_custom_roundtrip": True,
            "generalized_roundtrip": True, "invalid_controls": 10}


def historical_trainers():
    from ColorCorrectionPipeline.core.correction import fit_model, predict_

    rng = np.random.default_rng(1729)
    x = rng.uniform(.03, .96, (96, 3))
    rows = []
    for method in ("linear", "pls", "nn", "custom"):
        model = fit_model(x, x * .85 + .05,
                          dict(mtd=method, degree=2, hidden_layers=[8], max_iterations=20,
                               ncomp=2, batch_size=12, patience=4, random_state=17, use_lut=False))
        prediction = predict_(x[:6].reshape(2, 3, 3), model)
        assert np.isfinite(prediction).all() and prediction.shape == (2, 3, 3)
        rows.append({"method": method, "estimator": type(model.model).__name__})
    return rows


def legacy_import_contracts():
    """Historical model imports remain usable without optional chart backends."""
    import pickle
    import sys
    from types import SimpleNamespace
    from ColorCorrectionPipeline import key_functions as legacy

    model = legacy.Regressor_Model()
    model.degree = 2
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is type(model) and restored.degree == 2
    assert type(restored).__module__ == "ColorCorrectionPipeline.key_functions"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch.dict(sys.modules, {"colour_checker_detection": None}):
        try:
            legacy.detect_colour_checkers_segmentation(image)
        except ImportError as exc:
            assert "ColorCorrectionPipeline[segmentation]" in str(exc)
        else:
            raise AssertionError("Missing segmentation backend was accepted")
    with patch.object(legacy, "cv2", SimpleNamespace()):
        requests = (
            lambda: legacy.extract_color_chart(image),
            lambda: legacy.extract_color_charts(image),
            lambda: legacy.extract_color_chart_ex(image, np.zeros((24, 3))),
        )
        for request in requests:
            try:
                request()
            except ImportError as exc:
                assert "opencv-contrib-python" in str(exc)
            else:
                raise AssertionError("Missing MCC backend was accepted")
    return {"base_import": True, "pickle_class_path": type(restored).__module__,
            "optional_backend_errors": 4}


def precision_estimation():
    from types import SimpleNamespace
    from ColorCorrectionPipeline import FittingPolicy
    from ColorCorrectionPipeline.ui_stages import fit_ffc

    white = np.full((64, 64, 3), .6)
    changed = white + np.linspace(0, 2.0 ** -26, 64)[None, :, None]
    np.testing.assert_array_equal(np.rint(white * 255), np.rint(changed * 255))
    gains = []
    for pixels in (white, changed):
        owner = SimpleNamespace(
            fitting_policy=FittingPolicy(version="ui-compatible-v1", ui_runtime="precision-f64-v1"),
            white_image_data=ImageData.from_array(pixels),
            White_Image=np.rint(pixels[:, :, ::-1] * 255).astype(np.uint8),
            models=SimpleNamespace(model_ffc=None), numerical_report={})
        fit_ffc(owner, white, {"fit_method": "linear", "crop_rect": [0, 0, 64, 64],
                              "bins": 8, "smooth_window": 3})
        gains.append(owner.models.model_ffc)
    difference = float(np.max(np.abs(gains[0] - gains[1])))
    assert difference > 1e-9, difference
    return {"byte_previews_equal": True, "white_gain_difference": difference}


def altered_bundle(bundle, folder, alteration):
    with zipfile.ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    assert manifest["version"] == 2
    if alteration == "downgraded":
        manifest["version"] = 1
    elif alteration == "chart-selection":
        manifest["metadata"]["chart_selection"] = []
    elif alteration == "high-lut":
        def enable_lut(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "use_lut":
                        value[key] = True
                    else:
                        enable_lut(item)
            elif isinstance(value, list):
                for item in value:
                    enable_lut(item)
        enable_lut(manifest)
    else:
        raise ValueError(alteration)
    manifest.pop("content_sha256")
    def encode(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")).encode("utf8")
    manifest["content_sha256"] = hashlib.sha256(encode(manifest)).hexdigest()
    members["manifest.json"] = encode(manifest)
    altered = folder / (alteration + ".ccpb")
    with zipfile.ZipFile(altered, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return altered


def reject_altered_bundle(bundle, folder, alteration):
    expected_error = {"downgraded": "require bundle version 2",
                      "high-lut": "high precision requires direct CC"}[alteration]
    altered = altered_bundle(bundle, folder, alteration)
    try:
        ColorCorrection().load_bundle(altered)
    except ValueError as exc:
        assert expected_error in str(exc), str(exc)
    else:
        raise AssertionError("An incompatible altered calibration was accepted: " + alteration)


def verify_native_batch_tiles(calibration, scene):
    """Public batch prediction must align spatial gains and respect the tile limit."""
    from ColorCorrectionPipeline import ui_execution

    if not ui_execution.accel.HAS_NUMBA:
        return
    stages = ("FFC", "CC")
    kernel = ui_execution._run_into
    final = calibration.predict_image(scene, stages="final")["CC"]
    signatures = tuple(kernel.signatures)
    expected = calibration.predict_image(scene, stages=stages)
    # Retaining FFC changes the output count, but needs no second JIT kernel.
    assert signatures and tuple(kernel.signatures) == signatures
    np.testing.assert_array_equal(expected["CC"], final)
    tile_sizes = []

    def record_tile(output, pixels, gain, *parameters):
        tile_sizes.append(len(pixels))
        assert len(gain) == len(pixels)
        return kernel(output, pixels, gain, *parameters)

    # Deliberately split scanlines and leave a partial last tile.
    limit = 65521
    with patch.object(ui_execution, "_run_into", record_tile):
        item, = calibration.predict_batch([scene], stages=stages, chunk_pixels=limit,
                                          max_workers=1).items
    assert item.status == "succeeded", item.error
    assert tuple(item.images) == stages
    assert sum(tile_sizes) == expected["CC"].size // 3 and max(tile_sizes) <= limit
    for stage in stages:
        np.testing.assert_array_equal(item.images[stage], expected[stage])


@contextmanager
def prediction_in_flight(pipe, scene, expected):
    """Hold one reader after its snapshot, then check it against the old calibration."""
    captured, resume = Event(), Event()
    predict = ColorCorrection._predict_image

    def pause(snapshot, *args, **kwargs):
        captured.set()
        if not resume.wait(120):
            raise TimeoutError("prediction was never released")
        return predict(snapshot, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with patch.object(ColorCorrection, "_predict_image", pause):
                reader = pool.submit(pipe.predict_image, scene, stages="final")
                assert captured.wait(30), "prediction did not take its snapshot"
            yield
        finally:
            resume.set()
        np.testing.assert_array_equal(reader.result()["CC"], expected)


def verify_failed_publication(pipe, chart, white, config, artifacts):
    """Fail the final replacement after a real metrics write to existing destinations."""
    bundle = Path(pipe.Models_path)
    replace = os.replace
    replaced = []

    def reject_bundle(source, destination):
        destination = Path(destination)
        if destination == bundle:
            raise PermissionError("injected bundle replacement failure")
        replace(source, destination)
        if destination.parent == bundle.parent:
            replaced.append(destination.name)

    try:
        with patch.object(os, "replace", reject_bundle):
            pipe.run(chart, white, "chart", config)
    except PermissionError as error:
        assert "injected bundle replacement failure" in str(error)
    else:
        raise AssertionError("failed publication was accepted")
    assert replaced.count("chart_metrics.csv") == 2  # Publish and restore.
    assert not pipe.last_run["success"] and not pipe.last_run["artifacts"]
    assert pipe.last_run["errors"][-1]["type"] == "PermissionError"
    assert not list(bundle.parent.glob(".ccp-run-*"))
    for path, digest in artifacts.items():
        assert hashlib.sha256(path.read_bytes()).digest() == digest


def verify_refits(samples, folder, device):
    chart, white, scene = (samples[key] for key in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))
    pipe = ColorCorrection()

    def config(model):
        return Config(model=model, input={"frame_index": 0}, chart={"device": device},
                      persistence={"enabled": True, "directory": str(folder), "format": "bundle"})

    first = pipe.fit_or_raise(chart, white, "chart", config("ccm"))
    bundle = folder / "chart_models.ccpb"
    assert str(bundle) in first.artifacts and first.name == "chart" and first.images
    artifacts = {Path(path): hashlib.sha256(Path(path).read_bytes()).digest() for path in first.artifacts}
    baseline = pipe.predict_image(scene, stages="final")["CC"]
    previous = ColorCorrection().load_bundle(bundle)
    verify_native_batch_tiles(previous, scene)
    np.testing.assert_array_equal(previous.predict_image(scene, stages="final")["CC"], baseline)

    try:
        pipe.fit_or_raise(chart, None, "chart", config("rpcc3"))
    except CalibrationError as error:
        assert error.errors[0]["stage"] == "FFC" and "White_Image" in str(error)
    else:
        raise AssertionError("Missing white field accepted")
    _, _, failed = pipe.run(chart, None, "chart", config("rpcc3"))
    assert failed and not pipe.last_run["artifacts"] and pipe.models is first.models
    assert not pipe.last_run["success"] and pipe.last_run["errors"][0]["stage"] == "FFC"
    for path, digest in artifacts.items():
        assert hashlib.sha256(path.read_bytes()).digest() == digest
    np.testing.assert_array_equal(pipe.predict_image(scene, stages="final")["CC"], baseline)

    verify_failed_publication(pipe, chart, white, config("rpcc3"), artifacts)
    assert pipe.models is first.models and Path(pipe.Models_path) == bundle
    np.testing.assert_array_equal(pipe.predict_image(scene, stages="final")["CC"], baseline)

    invalid = folder / "corrupt.ccpb"
    invalid.write_bytes(b"invalid bundle")
    last_attempt = pipe.last_run
    for bad_bundle in (invalid, altered_bundle(bundle, folder, "chart-selection")):
        try:
            pipe.load_bundle(bad_bundle)
        except (ValueError, AttributeError):
            pass
        else:
            raise AssertionError(f"corrupt load was accepted: {bad_bundle}")
        assert pipe.last_run is last_attempt and pipe.models is first.models
        assert Path(pipe.Models_path) == bundle and pipe.fitted_config is not None
        for path, digest in artifacts.items():
            assert hashlib.sha256(path.read_bytes()).digest() == digest
        np.testing.assert_array_equal(pipe.predict_image(scene, stages="final")["CC"], baseline)

    with prediction_in_flight(pipe, scene, baseline):
        second = pipe.fit_or_raise(chart, white, "chart", config("rpcc3"))
        corrected = pipe.predict_image(scene, stages="final")["CC"]
    assert second.models is pipe.models and second.models is not first.models
    changed_pixels = int(np.count_nonzero(np.any(corrected != baseline, axis=-1)))
    assert changed_pixels > 0
    np.testing.assert_array_equal(ColorCorrection().load_bundle(bundle).predict_image(scene, stages="final")["CC"], corrected)
    first.models.save_bundle(folder / "first.ccpb")
    with prediction_in_flight(pipe, scene, corrected):
        assert pipe.load_bundle(folder / "first.ccpb") is pipe
        assert Path(pipe.Models_path) == folder / "first.ccpb"
        assert pipe.last_run["operation"] == "load_bundle" and pipe.last_run["success"]
        assert pipe.Image is None and pipe.White_Image is None and pipe.fitted_config is None
        assert pipe.image_data is None and pipe.white_image_data is None
        assert pipe.REFERENCE_RGB_PD is None and pipe.REFERENCE_DISPLAY_RGB_PD is None
        assert not pipe._ffc_multiplier_cache and not pipe._ffc_quality_cache
        assert pipe.observations.to_dict() == previous.observations.to_dict()
        assert pipe.last_run["contracts"] == previous.last_run["contracts"]
        np.testing.assert_array_equal(pipe.predict_image(scene, stages="final")["CC"], baseline)
    return {"successful_refit": True, "failed_refit_preserved": True, "named_bundle_replay_exact": True,
            "failed_save_rollback_files": len(artifacts), "corrupt_load_preserved": True, "malformed_metadata_preserved": True,
            "in_flight_refit_exact": True, "in_flight_load_exact": True, "ccm_rpcc3_distinct_scene_pixels": changed_pixels,
            "ccm_chart_de00": first.metrics["chart_CC"]["delta_e00_after_mean"],
            "rpcc3_chart_de00": second.metrics["chart_CC"]["delta_e00_after_mean"]}


def verify_legacy_conv_notice(samples, folder, device):
    chart, scene = samples["Image_1.JPG"], samples["Sample_1.JPG"]
    options = dict(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
                   chart_detection={"device": device}, CC_kwargs={"cc_method": "CoNv"})
    logger = logging.getLogger("ColorCorrectionPipeline.pipeline")
    with patch.object(logger, "warning", wraps=logger.warning) as notice:
        implicit = Config.from_dict(options)
        explicit = Config(fitting_policy="legacy-v1", **options)
        assert implicit.to_dict() == explicit.to_dict()
        implicit.validate()
        pipe = ColorCorrection()
        notice.assert_not_called()

        _, _, failed = pipe.run(chart, config=implicit)
        assert not failed, pipe.last_run
        notice.assert_called_once()
        message = notice.call_args[0][0]
        assert "encoded sRGB" in message and 'Config(model="3x3")' in message
        assert 'fitting_policy="legacy-v1"' in message
        notice.reset_mock()

        expected = pipe.predict_image(scene, stages="final")["CC"]
        bundle = folder / "legacy-conv.ccpb"
        pipe.save_bundle(bundle)
        restored = ColorCorrection().load_bundle(bundle)
        np.testing.assert_array_equal(restored.predict_image(scene, stages="final")["CC"], expected)
        for cfg in (explicit, Config.from_dict(implicit.to_dict())):
            restored.fit_or_raise(chart, config=cfg)
            np.testing.assert_array_equal(restored.models.model_cc[0], pipe.models.model_cc[0])
            np.testing.assert_array_equal(restored.predict_image(scene, stages="final")["CC"], expected)
        implicit.do_cc = False
        restored.fit_or_raise(chart, config=implicit)
        notice.assert_not_called()
    return {"implicit_conv_notices": 1, "explicit_and_replay_quiet": True,
            "legacy_matrix_and_prediction_exact": True}


def photos(device):
    results = []
    with sample_paths() as samples, tempfile.TemporaryDirectory(prefix="ccp-v2-images-") as name:
        folder = Path(name)
        results.append(verify_refits(samples, folder, device))
        results.append(verify_legacy_conv_notice(samples, folder, device))
        for model, basis, degree in (("rpcc2", None, None), ("pls", "root", 3), ("nn", "hybrid", 2)):
            settings = {"model": model}
            cc = {"use_lut": False}
            if basis:
                settings.update(basis=basis, degree=degree)
            if model == "pls":
                cc["ncomp"] = 3
            if model == "nn":
                cc.update(hidden_layers=[8], max_iterations=100, patience=50)
            cfg = Config(**settings, image_options={"frame_index": 0},
                         chart_detection={"device": device},
                         validation_options={"enabled": False}, CC_kwargs=cc)
            pipe = ColorCorrection()
            started = time.perf_counter()
            metrics, _, failed = pipe.run(samples["Image_1.JPG"], samples["white.JPG"],
                                         name_="fit", config=cfg)
            assert not failed, pipe.last_run
            assert pipe.models.metadata["fitting_policy"]["ui_runtime"] == "native-v1"
            elapsed = time.perf_counter() - started
            before = pipe.predict_image(samples["Sample_1.JPG"], stages="final")["CC"]
            bundle = folder / (model + ".ccpb")
            pipe.save_bundle(bundle)
            restored = ColorCorrection().load_bundle(bundle)
            after = restored.predict_image(samples["Sample_1.JPG"], stages="final")["CC"]
            np.testing.assert_array_equal(before, after)
            if basis is not None:
                reject_altered_bundle(bundle, folder, "downgraded")
            record = write_image_data(folder / (model + ".png"), after, dtype="uint8",
                                      allow_precision_loss=True, png_profile="fast-lossless")
            assert record.maximum_codec_error == 0
            results.append({"model": model, "basis": basis, "degree": degree,
                            "calibration_seconds": elapsed,
                            "training_chart_de00": metrics["fit_CC"]["delta_e00_after_mean"],
                            "runtime": "native-v1", "use_lut": False,
                            "reload_exact": True, "codec_error": 0})
        # Check the high runtime with genuinely adjacent source values, using CC only.
        chart = read_image_data(samples["Image_1.JPG"], color_space="srgb", frame_index=0)
        white = read_image_data(samples["white.JPG"], color_space="srgb", frame_index=0)
        white16 = ImageData.from_array(np.rint(white.to_srgb() * 65535).astype(np.uint16))
        mixed = ColorCorrection()
        _, _, failed = mixed.run(
            chart, white16, name_="mixed",
            config=Config(chart_detection={"device": device}, validation_options={"enabled": False}))
        assert not failed, mixed.last_run
        assert mixed.models.metadata["fitting_policy"]["ui_runtime"] == "precision-f64-v1"
        scene = read_image_data(samples["Sample_1.JPG"], color_space="srgb", frame_index=0)
        scene16 = ImageData.from_array(np.rint(scene.to_srgb() * 65535).astype(np.uint16))
        mixed_before = mixed.predict_image(scene16, stages="final")["CC"]
        mixed.save_bundle(folder / "mixed.ccpb")
        reject_altered_bundle(folder / "mixed.ccpb", folder, "high-lut")
        mixed_after = ColorCorrection().load_bundle(folder / "mixed.ccpb").predict_image(
            scene16, stages="final")["CC"]
        np.testing.assert_array_equal(mixed_before, mixed_after)
        linear_scene = ImageData.from_array(srgb_to_linear(scene16.to_srgb()), color_space="linear-srgb")
        np.testing.assert_allclose(mixed.predict_image(linear_scene, stages="final")["CC"],
                                   mixed_before, rtol=0, atol=1e-12)
        twelve_bit = ImageData.from_array(np.rint(scene.to_srgb() * 4095).astype(np.uint16),
                                         value_range=(0, 4095))
        assert np.isfinite(mixed.predict_image(twelve_bit, stages="final")["CC"]).all()
        results.append({"mixed_chart_uint8_white_uint16": True, "reload_exact": True,
                        "linear_srgb_scene_equivalent": True, "declared_uint12_scene": True})
        cfg = Config(model="rpcc2", precision="high", do_ffc=False, do_gc=False, do_wb=False,
                     chart_detection={"device": device},
                     validation_options={"enabled": False}, CC_kwargs={"use_lut": False})
        pipe = ColorCorrection()
        _, _, failed = pipe.run(chart, name_="high", config=cfg)
        assert not failed, pipe.last_run
        assert pipe.models.metadata["fitting_policy"]["ui_runtime"] == "precision-f64-v1"
        probes = [
            np.repeat(np.arange(32768, 32832, dtype=np.uint16).reshape(8, 8, 1), 3, axis=2),
            np.repeat((.5 + np.arange(4) * 2.0 ** -32).reshape(2, 2, 1), 3, axis=2),
        ]
        for index, values in enumerate(probes):
            typed = ImageData.from_array(values, color_space="srgb")
            corrected = pipe.predict_image(typed, stages="final")["CC"]
            levels = len(np.unique(corrected.reshape(-1, 3), axis=0))
            assert levels == values.shape[0] * values.shape[1], levels
            record = write_image_data(folder / ("high-" + str(index) + ".tiff"),
                                      corrected, dtype="float64")
            assert record.maximum_codec_error == 0
        results.append({"precision_profile": pipe.models.metadata["fitting_policy"],
                        "adjacent_uint16_levels": 64, "adjacent_float64_levels": 4})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photos", action="store_true")
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--forbid-source", type=Path)
    args = parser.parse_args()
    import ColorCorrectionPipeline
    package = Path(ColorCorrectionPipeline.__file__).resolve()
    if args.forbid_source:
        assert args.forbid_source.resolve() not in package.parents, package
    result = {"version": version("ColorCorrectionPipeline"), "package": str(package),
              "configurations": configurations(), "models": numerical_contracts(),
              "historical_trainers": historical_trainers(),
              "legacy_import_contracts": legacy_import_contracts(),
              "precision_estimation": precision_estimation()}
    if args.photos:
        result["photos"] = photos(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=json_value) + "\n", encoding="utf8")
    print("Passed v2 contracts, 36 model/basis/degree fits, historical trainers"
          + (" and real-image roundtrip/precision checks." if args.photos else "."))


if __name__ == "__main__":
    main()
