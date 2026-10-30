"""Qualify an installed wheel from a neutral cwd, using no source-tree imports.

Run with python -I. This script intentionally supports the advertised Python 3.8
minimum and writes machine-readable evidence instead of publishing anything.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import site
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photo", type=Path)
    parser.add_argument("--photo-device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--require-photo-gpu", action="store_true")
    parser.add_argument("--legacy-baseline", type=Path)
    parser.add_argument("--forbid-source", type=Path)
    parser.add_argument("--expect-numba", choices=("yes", "no", "either"), default="either")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / "matplotlib-cache"))
    import importlib.metadata as metadata
    import importlib.util
    import numpy as np
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection, Config, ImageData, MyModels
    from ColorCorrectionPipeline.core.correction import predict_
    from ColorCorrectionPipeline.fitting import fit_observations
    from ColorCorrectionPipeline.stage_policies import MonotonicToneModel
    from ColorCorrectionPipeline.io import read_image_data, write_image_data
    from ColorCorrectionPipeline.core.accel import HAS_NUMBA, backend_info

    prefix = Path(sys.prefix).resolve()
    package_file = Path(package.__file__).resolve()
    assert sys.prefix != sys.base_prefix, "qualification requires an isolated virtual environment"
    assert prefix in package_file.parents, (prefix, package_file)
    config = (prefix / "pyvenv.cfg").read_text().lower()
    assert "include-system-site-packages = false" in config
    assert site.ENABLE_USER_SITE is False, "run with python -I"
    assert not any(Path(path).name == "site-packages" and prefix not in Path(path).resolve().parents for path in sys.path if path)
    if args.forbid_source:
        assert args.forbid_source.resolve() not in package_file.parents
    numba_present = importlib.util.find_spec("numba") is not None
    assert HAS_NUMBA == numba_present
    if args.expect_numba != "either":
        assert numba_present == (args.expect_numba == "yes")
    work = args.output.parent / (args.output.stem + "-artifacts")
    work.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(901).uniform(.04, .94, (24, 3))
    target = x * .9 + .02
    groups = ["r%dc%d" % (i // 6 + 1, i % 6 + 1) for i in range(24)]
    image = np.random.default_rng(902).uniform(.03, .9, (96, 128, 3))
    families = {}
    for method in ("linear", "pls", "nn", "custom"):
        params = {"mtd": method, "degree": 1, "use_lut": False}
        if method != "linear":
            params["max_iterations"] = 4
        if method in ("nn", "custom"):
            params.update(hidden_layers=[4], patience=2, batch_size=8)
        fitted = fit_observations(x, target, groups, params, "reproducible-v1")
        models = MyModels()
        models.model_cc = (fitted, {}, "ours")
        models.model_gc = MonotonicToneModel(np.array([0., 20., 50., 80., 100.]), np.array([0., 20., 50., 80., 100.]))
        models.model_wb = np.eye(3) * .95
        models.model_ffc = np.ones(image.shape[:2])
        before = predict_(x, fitted)
        path = work / (method + ".ccpb")
        models.save_bundle(path)
        loaded = MyModels().load_bundle(path)
        np.testing.assert_allclose(predict_(x, loaded.model_cc[0]), before, atol=2e-12, rtol=2e-12)
        pipeline = ColorCorrection().load_bundle(path)
        predicted = pipeline.predict_image(image, stages="final")["CC"]
        batch = pipeline.predict_batch([image, work / "missing.png", image], ordered=True, max_workers=1, max_in_flight=2)
        assert [item.status for item in batch.items] == ["succeeded", "failed", "succeeded"]
        tolerance = 2e-6 if method == "custom" else 2e-12
        np.testing.assert_allclose(batch.items[0].images["CC"], predicted, atol=tolerance, rtol=tolerance)
        families[method] = {"fit_bundle_apply_batch": True, "max_direct_reload_difference": float(np.max(np.abs(predict_(x, loaded.model_cc[0]) - before)))}

    # Qualify prepared execution against an installed portable affine model. A small
    # affine LUT makes a tight numerical oracle without a costly benchmark.
    prepared_pipeline = ColorCorrection().load_bundle(work / "linear.ccpb")
    prepared_pipeline.models.model_gc = None
    expected = prepared_pipeline.predict_image(image, stages="final")["CC"]
    execution = {}
    for backend in ("numpy", "auto", "numba"):
        options = {"mode": "combined-lut", "backend": backend, "grid_sizes": [3],
                   "validation_samples": 64, "max_rgb_error": 1e-10,
                   "max_delta_e00": 1e-8, "allow_fallback": True}
        with prepared_pipeline.prepare_execution(options) as prepared:
            result = prepared.predict_image(image)
            np.testing.assert_allclose(result.images["CC"], expected, atol=2e-12, rtol=2e-12)
            diagnostic = dict(result.diagnostics["backend"])
            actual = "cpu-numpy" if backend == "numpy" or not numba_present else "cpu-numba"
            assert diagnostic["actual_backend"] == actual, diagnostic
            if backend == "numba" and not numba_present:
                assert diagnostic["fallback_reason"] == "Numba unavailable"
            execution[backend] = {"actual_backend": diagnostic["actual_backend"],
                "fallback_reason": diagnostic["fallback_reason"],
                "max_apply_difference": float(np.max(np.abs(result.images["CC"] - expected)))}
            if backend == "numpy":
                prepared.export_cube(work / "combined.cube", exclude_ffc=True)
    if not numba_present:
        try:
            prepared_pipeline.prepare_execution({"mode": "direct", "backend": "numba",
                "validation_samples": 64, "allow_fallback": False})
        except RuntimeError as exc:
            assert "Numba" in str(exc) and "unavailable" in str(exc), str(exc)
            execution["strict_numba_unavailable_rejected"] = True
        else:
            raise AssertionError("strict unavailable Numba request must fail")

    legacy_report = None
    if args.legacy_baseline:
        from ColorCorrectionPipeline.core import correction
        baseline = json.loads(args.legacy_baseline.read_text(encoding="utf8"))
        gamma_image = np.asarray(baseline["gamma_input"], dtype=np.float64)
        coefficients = np.asarray(baseline["gamma_coefficients"], dtype=np.float64)
        gamma = correction.predict_image(gamma_image, coefficients)
        np.testing.assert_allclose(gamma, baseline["gamma_output"], rtol=0, atol=2e-6)
        legacy = ColorCorrection()
        legacy.REF_ILLUMINANT = np.asarray(baseline["reference_illuminant"], dtype=np.float64)
        legacy.models.model_gc = coefficients
        legacy.models.model_wb = np.eye(3) * .95
        before = legacy.predict_image(gamma_image, stages="final")["CC"]
        legacy.models.save_bundle(work / "legacy-gc-wb.ccpb")
        restored = ColorCorrection().load_bundle(work / "legacy-gc-wb.ccpb")
        after = restored.predict_image(gamma_image, stages="final")["CC"]
        np.testing.assert_array_equal(after, before)
        legacy_report = {"source_revision": baseline["source_revision"],
            "fixture_sha256": hashlib.sha256(args.legacy_baseline.read_bytes()).hexdigest(),
            "max_gamma_difference_from_baseline": float(np.max(np.abs(gamma - np.asarray(baseline["gamma_output"])))),
            "baseline_atol": 2e-6, "max_bundle_reload_difference": float(np.max(np.abs(after - before))),
            "legacy_gc_torch_cuda": bool(correction.is_cuda)}

    from ColorCorrectionPipeline.flat_field import FlatFieldCorrection
    yy, xx = np.mgrid[:96, :128]
    white = np.repeat(np.rint(160 + 20*xx/127 + 10*yy/95).astype(np.uint8)[..., None], 3, axis=2)
    flat = FlatFieldCorrection(white, crop_rect=[0, 0, 128, 96], bins=8, smooth_window=3, safe_mode=True)
    gain = flat.compute_multiplier(degree=2, fit_method="linear")
    assert np.isfinite(gain).all() and np.min(gain) > 0
    integer = np.arange(180, dtype=np.uint16).reshape(6, 10, 3) * 307
    record = write_image_data(work / "uint16.png", ImageData.from_array(integer), overwrite=True)
    np.testing.assert_array_equal(read_image_data(record.path).pixels, integer / 65535)
    record = write_image_data(work / "float.tiff", image, overwrite=True)
    np.testing.assert_array_equal(read_image_data(record.path).pixels, image)
    # Use a new file-mode namespace per call, preserving previous evidence.
    outputs = pipeline.predict_batch([image, image], output_dir=work / "exports", max_workers=1)
    assert outputs.summary["succeeded"] == 2 and all(not item.images for item in outputs.items)

    photo = None
    if args.photo:
        import cv2
        bgr = cv2.imread(str(args.photo))
        assert bgr is not None
        scale = min(1., 900 / max(bgr.shape[:2]))
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        calibration = ColorCorrection()
        config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
            color_semantics={"version": "explicit-v1"}, stage_policy="modern-v1", fitting_policy="reproducible-v1",
            CC_kwargs={"degree": 1}, chart_detection={"device": args.photo_device}, save=True, model_format="bundle", save_path=str(work))
        _, stages, failed = calibration.run(ImageData.from_array(bgr, channel_order="BGR"), name_="photo", config=config)
        assert not failed, calibration.last_run
        restored = ColorCorrection().load_bundle(work / "photo_models.ccpb")
        restored_pixels = restored.predict_image(ImageData.from_array(bgr, channel_order="BGR"))["CC"]
        np.testing.assert_allclose(restored_pixels, stages["photo_CC"], atol=2e-12, rtol=2e-12)
        detection = calibration.last_run["detection"]
        successful = [row for row in detection if row.get("backend") == "model" and row.get("status") == "detected"]
        assert successful and all(row.get("execution_verified") for row in successful), detection
        for row in successful:
            if row.get("selected_device") != "cpu":
                assert row.get("node_provider_counts", {}).get(row["provider"], 0) > 0, row
        photo = {"name": args.photo.name, "sha256": hashlib.sha256(args.photo.read_bytes()).hexdigest(),
            "requested_device": args.photo_device, "detection": detection,
            "gpu_requirement_met": all(row.get("selected_device") != "cpu" for row in successful),
            "fit_bundle_apply": True,
            "max_saved_reload_difference": float(np.max(np.abs(restored_pixels - stages["photo_CC"])))}
    # Exercise the additive UI numerical contract without any native UI runtime.
    from ColorCorrectionPipeline.ui_fitting import fit_ui
    from ColorCorrectionPipeline.colorimetry import srgb_to_linear
    ui_families = {}
    for name in ("ccm", "rpcc2", "rpcc3", "linear", "pls", "nn"):
        params = {"model": name, "degree": 2}
        if name == "nn":
            params.update(hidden_layers=[4], max_iterations=4, patience=2)
        if name == "pls":
            params["ncomp"] = 3
        model, fitted_report = fit_ui(x, srgb_to_linear(target), params)
        prediction = model.predict_rgb(x)
        assert prediction.shape == x.shape and np.isfinite(prediction).all()
        assert fitted_report["version"] == "ui-compatible-v1"
        assert not model.weights or all(not w.flags.writeable for w in model.weights)
        ui_families[name] = {"fit_predict": True, "fitting_domain": fitted_report["fitting_domain"]}
        if name == "nn":
            from ColorCorrectionPipeline import ui_nn
            actual_backend = fitted_report["details"]["backend"]
            assert actual_backend == ("cpu-numba-ordered-f64" if numba_present else "cpu-numpy-ordered-f64")
            assert fitted_report["details"]["arithmetic"] == "native-ordered-f64-v1"
            # Exercise the fallback too; allow platform libm roundoff rather
            # than promising cross-platform bit identity from a short smoke fit.
            enabled = ui_nn._enabled
            try:
                ui_nn._enabled = False
                reference_nn, _ = fit_ui(x, srgb_to_linear(target), params)
            finally:
                ui_nn._enabled = enabled
            maximum = 0.0
            for left, right in zip(model.weights + model.biases, reference_nn.weights + reference_nn.biases):
                np.testing.assert_allclose(left, right, rtol=0, atol=2e-12)
                maximum = max(maximum, float(np.max(np.abs(left - right))))
            ui_families[name].update(arithmetic_backend=actual_backend,
                max_numpy_parameter_difference=maximum)
    native_ui = None
    if args.photo:
        config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
            fitting_policy="ui-compatible-v1", CC_kwargs={"model":"rpcc2"},
            chart_detection={"device":args.photo_device}, validation_options={"enabled":False},
            save=True, model_format="bundle", save_path=str(work))
        _, stages, failed = calibration.run(ImageData.from_array(bgr, channel_order="BGR"), name_="ui-photo", config=config)
        assert not failed, calibration.last_run
        restored = ColorCorrection().load_bundle(work / "ui-photo_models.ccpb")
        pixels = restored.predict_image(ImageData.from_array(bgr, channel_order="BGR"), stages="final")["CC"]
        np.testing.assert_array_equal(pixels, stages["ui-photo_CC"])
        ui_families["photo_bundle_reload_exact"] = True

        # Qualify the new runtime from bytes, without depending on the UI app.
        native_input = ImageData.from_array(bgr, channel_order="BGR")
        config = Config(do_ffc=False, do_gc=False, do_wb=False, check_saturation=False,
            fitting_policy="ui-native-v1", CC_kwargs={"model": "rpcc2"},
            chart_detection={"device": args.photo_device}, validation_options={"enabled": False},
            save=True, model_format="bundle", save_path=str(work))
        _, _, failed = calibration.run(native_input, name_="native-photo", config=config)
        assert not failed, calibration.last_run
        before = calibration.predict_image(native_input, stages="final")["CC"]
        restored = ColorCorrection().load_bundle(work / "native-photo_models.ccpb")
        after = restored.predict_image(native_input, stages="final")["CC"]
        np.testing.assert_array_equal(before, after)
        assert restored.models.metadata["fitting_policy"]["ui_runtime"] == "native-v1"
        from ColorCorrectionPipeline import ui_execution
        candidate = ui_execution.try_execute(native_input.pixels, restored.models, ("CC",), native_codes=True)
        if numba_present:
            assert candidate is not None and ui_execution._enabled
            assert ui_execution._run_into.signatures, "native image kernel did not execute"
            np.testing.assert_array_equal(candidate["CC"], after)
        else:
            assert candidate is None
        for operation, message in (
            (lambda: restored.predict_image(np.zeros(native_input.shape, dtype=np.float64), stages="final"), "uint8 sRGB"),
            (restored.prepare_execution, "recorded fusion"),
        ):
            try:
                operation()
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError("native byte/fusion contract was not enforced")
        exported = write_image_data(work / "native-photo.png", after, dtype="uint8",
            allow_precision_loss=True, overwrite=True)
        assert exported.maximum_codec_error == 0
        native_ui = {"runtime": "native-v1", "photo_fit_bundle_reload_exact": True,
            "max_saved_reload_difference": float(np.max(np.abs(before - after))),
            "image_kernel_executed": numba_present, "byte_and_prepared_contracts_enforced": True,
            "strict_png_export_passed": True}
    report = {"ui_compatible_families": ui_families, "native_ui": native_ui,
        "python": platform.python_version(), "executable": sys.executable,
        "prefix": str(prefix), "base_prefix": sys.base_prefix, "package_file": str(package_file),
        "system_site_packages": False, "user_site_enabled": site.ENABLE_USER_SITE, "sys_path": list(sys.path),
        "numba_present": numba_present, "backend_info": backend_info(), "prepared_execution": execution,
        "legacy_gc": legacy_report,
        "families": families, "guarded_ffc_passed": True,
        "precision_io_passed": True, "file_mode_batch_passed": True, "photo": photo,
        "versions": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    if args.require_photo_gpu:
        assert photo and photo["gpu_requirement_met"], "required GPU default was not met; see recorded provider/fallback evidence"


if __name__ == "__main__":
    main()
