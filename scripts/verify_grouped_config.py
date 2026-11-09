"""Qualify additive grouped configuration without changing numerical recipes.

Run against an installed wheel from a neutral directory:
  python -I verify_grouped_config.py --output grouped.json --photos

Add --oracle-python /path/to/original-2.0/python to refit the same independently
specified flat recipes with an immutable pre-change wheel. Use matching Python,
dependencies and optional acceleration for strict floating-array comparison.
The normal default NN cap is a short contract smoke test; --nn-epochs 10000
qualifies the full NN100 configuration on the sample photographs.
"""
import argparse
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from ColorCorrectionPipeline import Config, ColorCorrection, ImageData
from ColorCorrectionPipeline.examples._common import sample_paths, json_value
from ColorCorrectionPipeline.io import read_image_data, write_image_data


def encoded(value):
    return json.loads(json.dumps(value, default=json_value, sort_keys=True))


def reject(operation, field=None):
    try:
        operation()
    except (TypeError, ValueError, AttributeError) as error:
        if field is not None:
            assert field.lower() in str(error).lower(), (field, str(error))
        return type(error).__name__
    raise AssertionError("An invalid or contradictory configuration was accepted")


def check_config(options):
    cfg = Config(**options)
    cfg.validate()
    return cfg


def fixture(method="ccm", nn_epochs=30):
    """Author flat and grouped recipes independently of the adapter under test."""
    solver = {"use_lut": True, "lazy_lut": True, "lut_grid_size": 33, "lut_min_pixels": 1}
    fit = {"model": method}
    selection = {"model": method}
    if method == "ccm":
        selection["degree"] = fit["degree"] = 1
    elif method == "rpcc3":
        selection["degree"] = fit["degree"] = 3
    elif method == "nn":
        selection.update(basis="polynomial", degree=2)
        fit.update(basis="polynomial", degree=2)
        parameters = dict(hidden_layers=[100], max_iterations=nn_epochs,
                          patience=1500, learning_rate=.001, tol=1e-6,
                          optim_type="Adam", dropout_rate=0., use_batch_norm=False)
        solver.update(parameters)
        fit["nn"] = deepcopy(parameters)
    elif method == "root-pls":
        selection = dict(model="pls", basis="root", degree=3)
        fit = dict(model="pls", basis="root", degree=3, pls={"components": 3, "tol": 1e-6})
        solver.update(ncomp=3, tol=1e-6)
    elif method == "hybrid-nn":
        selection = dict(model="nn", basis="hybrid", degree=2)
        fit = dict(model="nn", basis="hybrid", degree=2,
                   nn={"hidden_layers": [8], "max_iterations": 30, "patience": 15})
        solver.update(hidden_layers=[8], max_iterations=30, patience=15)
    else:
        raise ValueError(method)
    flat = dict(selection, precision="native-byte", check_saturation=False,
                do_ffc=True, do_gc=True, do_wb=True, do_cc=True,
                image_options={"frame_index": 0},
                chart_detection={"device": "cpu", "cpu_threads": 8,
                                 "allow_cpu_fallback": False, "sampling_inset": .75},
                FFC_kwargs={"fit_method": "linear", "degree": 1, "bins": 30,
                            "smooth_window": 7, "crop_rect": [43, 42, 1107, 723],
                            "min_signal": .02, "max_gain": 8.},
                stage_policy={"version": "modern-v1", "saturation": "report",
                              "max_clipped_fraction": .02, "gc_min_neutrals": 4,
                              "gc_min_span": 25., "gc_duplicate_tolerance": 2e-6,
                              "gc_target_tolerance": 1.2, "wb_headroom": .92,
                              "wb_min_signal": .02, "wb_max_gain": 7.},
                CC_kwargs=solver, validation_options={"enabled": False}, model_format="bundle")
    grouped = dict(
        input={"frame_index": 0},
        chart={"device": "cpu", "cpu_threads": 8, "allow_cpu_fallback": False, "sampling_inset": .75},
        numerics={"precision": "native-byte"},
        observations={"saturation": "report", "max_clipped_fraction": .02},
        steps={
            "ffc": {"enabled": True, "fit": {"method": "linear", "degree": 1, "bins": 30,
                                                 "smooth_window": 7},
                    "white_plane": {"crop_rect": [43, 42, 1107, 723]},
                    "quality": {"min_signal": .02, "max_gain": 8.}},
            "gc": {"enabled": True, "quality": {"min_neutrals": 4, "min_span": 25.,
                                                      "duplicate_tolerance": 2e-6, "target_tolerance": 1.2}},
            "wb": {"enabled": True, "fit": {"headroom": .92},
                   "quality": {"min_signal": .02, "max_gain": 7.}},
            "cc": {"enabled": True, "fit": fit,
                   "apply": {"use_lut": True, "lazy_lut": True, "grid_size": 33, "min_pixels": 1},
                   "validation": {"enabled": False}},
        }, persistence={"format": "bundle"})
    return flat, grouped


def configuration_contracts():
    from ColorCorrectionPipeline import Steps, FFCOptions, GCOptions, WBOptions, CCOptions, ReferenceProfile, StagePolicy
    flat, grouped = fixture("nn")
    ordinary, nested = check_config(flat), check_config(grouped)
    assert encoded(ordinary.to_dict()) == encoded(nested.to_dict())
    typed = deepcopy(grouped)
    parts = typed.pop("steps")
    typed["steps"] = Steps(ffc=FFCOptions(**parts["ffc"]), gc=GCOptions(**parts["gc"]),
                           wb=WBOptions(**parts["wb"]), cc=CCOptions(**parts["cc"]))
    assert encoded(check_config(typed).to_dict()) == encoded(ordinary.to_dict())
    for copier in (deepcopy, lambda item: pickle.loads(pickle.dumps(item))):
        assert copier(Steps()).to_dict() == {}
        assert encoded(check_config(dict(typed, steps=copier(typed["steps"]))).to_dict()) == encoded(ordinary.to_dict())
    defaults = Config()
    profile = ReferenceProfile.classic24()
    # Sparse shared-group requests must stand alone without a steps request.
    single_groups = (
        ({"chart": {"device": "cpu"}}, {"chart_detection": {"device": "cpu"}}),
        ({"input": {"frame_index": 0}}, {"image_options": {"frame_index": 0}}),
        ({"color": {"output_encoding": "linear-srgb"}},
         {"color_semantics": dict(defaults.color_semantics, output_encoding="linear-srgb")}),
        ({"numerics": {"precision": "high"}}, {"precision": "high"}),
        ({"observations": {"saturation": "report", "max_clipped_fraction": .1}},
         {"stage_policy": dict(defaults.stage_policy, saturation="report", max_clipped_fraction=.1)}),
        ({"persistence": {"enabled": False, "format": "bundle"}}, {"save": False, "model_format": "bundle"}),
        ({"reference": profile}, {"reference_profile": profile}),
        ({"steps": Steps()}, {}),
    )
    for grouped_only, flat_only in single_groups:
        assert encoded(check_config(grouped_only).to_dict()) == encoded(check_config(flat_only).to_dict())
    string_policy = check_config(dict(stage_policy="modern-v1", observations={"saturation": "report"}))
    object_policy = StagePolicy(version="modern-v1", saturation="report")
    assert check_config(dict(stage_policy=object_policy,
        observations={"saturation": "report"})).stage_policy == string_policy.stage_policy
    path = Path("not-created.ffc")
    surface = FFCOptions(white_plane={"model_path": path})
    assert surface.white_plane["model_path"] == str(path)
    assert check_config(dict(steps=Steps(ffc=surface))).FFC_kwargs["model_path"] == str(path)

    class MutablePath:
        def __init__(self, value):
            self.value = value

        def __fspath__(self):
            return self.value

    supplied_path = MutablePath("captured.ffc")
    captured = FFCOptions(white_plane={"model_path": supplied_path})
    supplied_path.value = "changed.ffc"
    assert captured.white_plane["model_path"] == "captured.ffc"
    inputs = dict(frame_index=0, icc_policy="assume-declared", alpha_policy="composite",
                  alpha_background=[.1, .2, .3], grayscale_policy="replicate", max_pixels=2000000)
    shared_flat = check_config(dict(model="ccm", precision="high", image_options=inputs,
        chart_detection={"device": "cpu", "cpu_threads": 8}, reference_profile=profile,
        color_semantics={"version": "explicit-v1", "input_encoding": "srgb",
                         "output_encoding": "linear-srgb", "fitting_domain": "linear-rgb"},
        save=True, save_path="not-created", model_format="bundle"))
    shared_grouped = check_config(dict(input=inputs, chart={"device": "cpu", "cpu_threads": 8},
        reference=profile, color={"output_encoding": "linear-srgb"}, numerics={"precision": "high"},
        steps={"cc": {"fit": {"model": "ccm"}}},
        persistence={"enabled": True, "directory": "not-created", "format": "bundle"}))
    assert encoded(shared_flat.to_dict()) == encoded(shared_grouped.to_dict())
    assert encoded(Config.from_dict(encoded(shared_grouped.to_grouped_dict())).to_dict()) == encoded(shared_flat.to_dict())
    assert not {"steps", "layout", "input", "numerics"}.intersection(nested.to_dict())
    for copier in (deepcopy, lambda item: pickle.loads(pickle.dumps(item))):
        assert encoded(copier(nested).to_dict()) == encoded(nested.to_dict())
    document = nested.to_grouped_dict()
    assert document["layout"] == "grouped-v1" and document["api_version"] == 2
    restored = Config.from_dict(encoded(document))
    assert encoded(restored.to_dict()) == encoded(nested.to_dict())
    assert encoded(Config.from_dict(nested.to_dict()).to_dict()) == encoded(nested.to_dict())
    defaults = Config()
    assert encoded(Config.from_dict(defaults.to_grouped_dict()).to_dict()) == encoded(defaults.to_dict())
    assert "use_lut" not in Config.from_dict(defaults.to_grouped_dict()).CC_kwargs
    assert Config.from_dict({}).api_version == 1
    assert Config.from_dict({}, source_version=2).api_version == 2
    raw_grouped = deepcopy(grouped)
    assert Config.from_dict(raw_grouped, source_version=2).api_version == 2
    reject(lambda: Config.from_dict(raw_grouped))
    for bad in (dict(document, api_version=1), dict(document, layout="grouped-v99")):
        reject(lambda bad=bad: Config.from_dict(bad))

    historical = Config(fitting_policy="legacy-v1", CC_kwargs={"mtd": "nn", "degree": 3,
        "hidden_layers": [16], "max_iterations": 17, "random_state": 13})
    historical.validate()
    assert Config.from_dict(historical.to_dict()).to_dict() == historical.to_dict()
    reject(historical.to_grouped_dict)
    reject(lambda: Config(fitting_policy="legacy-v1", steps={"wb": {"enabled": False}}))
    assert Config(CC_kwargs={"mtd": "nn"}).api_version == 1

    # Frozen snapshots must not become a second mutable source of truth.
    snapshot = nested.steps
    try:
        snapshot.wb.enabled = False
    except (FrozenInstanceError, AttributeError, TypeError):
        pass
    else:
        raise AssertionError("A grouped step snapshot was mutable")
    for values in (nested.input, nested.chart, nested.color, nested.numerics,
                   nested.observations, nested.persistence, snapshot.wb.fit):
        try:
            values["unexpected"] = True
        except TypeError:
            pass
        else:
            raise AssertionError("A grouped mapping snapshot was mutable")
    old_headroom = snapshot.wb.fit["headroom"]
    nested.stage_policy["wb_headroom"] = .89
    nested.do_wb = False
    nested.CC_kwargs["hidden_layers"][0] = 32
    assert nested.steps.wb.fit["headroom"] == .89 and nested.steps.wb.enabled is False
    assert snapshot.wb.fit["headroom"] == old_headroom and snapshot.wb.enabled is True
    assert nested.steps.cc.fit["nn"]["hidden_layers"] == (32,)
    assert Config.from_dict(nested.to_grouped_dict()).to_dict() == nested.to_dict()
    owned = check_config(grouped)
    grouped["steps"]["wb"]["fit"]["headroom"] = .70
    grouped["steps"]["cc"]["fit"]["nn"]["hidden_layers"][0] = 4
    assert owned.stage_policy["wb_headroom"] == .92 and owned.CC_kwargs["hidden_layers"] == [100]

    equal = Config(do_ffc=False, steps={"ffc": {"enabled": False}}, model="rpcc3",
                   CC_kwargs={"model": "rpcc3"}, persistence={"format": "bundle"}, model_format="bundle")
    equal.validate()
    check_config(dict(do_ffc=True, steps={"ffc": {"enabled": True}}, save=True,
                      persistence={"enabled": True, "directory": "not-created"}))
    errors = (
        dict(stage_policy="legacy-v1", observations={"saturation": "report"}),
        dict(stage_policy=object_policy, observations={"saturation": "exclude"}),
        # Equal-looking values of different types must not hide an invalid
        # explicit flat alias or let a grouped alias bypass its own validation.
        dict(do_ffc=1, steps={"ffc": {"enabled": True}}),
        dict(do_ffc=None, steps={"ffc": {"enabled": True}}),
        dict(CC_kwargs={"degree": 2.0}, steps={"cc": {"fit": {"degree": 2}}}),
        dict(save=True, save_path="not-created", persistence={"enabled": 1}),
        dict(do_ffc=True, steps={"ffc": {"enabled": False}}),
        dict(do_gc=False, steps={"gc": {"enabled": True}}),
        dict(do_wb=True, steps={"wb": {"enabled": False}}),
        dict(do_cc=True, steps={"cc": {"enabled": False}}),
        dict(save=False, persistence={"enabled": True, "directory": "not-created"}),
        dict(model_format="pickle", persistence={"format": "bundle"}),
        dict(save_path="one", persistence={"directory": "two"}),
        dict(model="ccm", steps={"cc": {"fit": {"model": "rpcc3"}}}),
        dict(CC_kwargs={"lut_grid_size": 17}, steps={"cc": {"apply": {"grid_size": 33}}}),
        dict(stage_policy={"version": "modern-v1", "wb_headroom": .91}, steps={"wb": {"fit": {"headroom": .92}}}),
        dict(image_options={"frame_index": 0}, input={"frame_index": 1}),
        dict(chart_detection={"cpu_threads": 4}, chart={"cpu_threads": 8}),
        dict(precision="high", numerics={"precision": "native-byte"}),
        dict(steps={"missing": {}}),
        dict(steps={"ffc": {"fit": {"missing": 1}}}),
        dict(steps={"ffc": {"fit": {"method": "nn"}}}),
        dict(steps={"gc": {"quality": {"gamma_degree": 3}}}),
        dict(steps={"wb": {"fit": {"temperature": 6500}}}),
        dict(steps={"cc": {"fit": {"model": "ccm", "nn": {"hidden_layers": [8]}}}}),
        dict(steps={"cc": {"apply": {"grid_size": 0}}}),
        dict(steps={"cc": {"fit": {"model": "pls", "basis": "root", "degree": 3,
                                       "pls": {"components": 0}}}}),
        dict(steps={"cc": {"validation": {"n_splits": 1}}}),
        dict(steps={"ffc": {"enabled": 1}}),
        dict(input={"frame_index": True}),
        dict(chart={"cpu_threads": 0}),
        dict(color={"input_encoding": "unknown"}),
        dict(numerics={"precision": "float32"}),
        dict(observations={"saturation": "recover"}),
        dict(persistence={"format": "json"}),
        dict(export={"dtype": "uint8"}),
    )
    for options in errors:
        reject(lambda options=options: check_config(options))
    return {"typed_and_mapping_equivalent": True, "flat_serialization_preserved": True,
            "grouped_json_roundtrip": True, "legacy_mapping_and_training_preserved": True,
            "single_group_defaults_equivalent": len(single_groups),
            "shared_input_chart_reference_color_numerics_persistence_equivalent": True,
            "fresh_immutable_views": True, "invalid_or_conflicting_requests": len(errors)}


def fitted(config, chart, white, name="fit"):
    pipeline = ColorCorrection()
    metrics, images, failed = pipeline.run(chart, white, name_=name, config=config)
    assert not failed, pipeline.last_run
    return pipeline, metrics, images


def oracle_worker(specification):
    """This branch intentionally uses only the original flat API."""
    spec = json.loads(Path(specification).read_text())
    config = Config.from_dict(spec["flat"], source_version=2)
    pipeline, _, images = fitted(config, spec["chart"], spec["white"])
    folder = Path(spec["output"])
    folder.mkdir(exist_ok=False)
    for name in ("fit_FFC", "fit_GC", "fit_WB", "fit_CC"):
        np.save(folder / (name + ".npy"), images[name], allow_pickle=False)
    scene = pipeline.predict_image(spec["scene"], stages="final")["CC"]
    np.save(folder / "scene.npy", scene, allow_pickle=False)
    import ColorCorrectionPipeline
    import inspect
    source = Path(inspect.getfile(Config))
    (folder / "identity.json").write_text(json.dumps({"version": version("ColorCorrectionPipeline"),
        "origin": str(Path(ColorCorrectionPipeline.__file__).resolve()),
        "numpy": np.__version__, "grouped_api_present": hasattr(Config, "to_grouped_dict"),
        "default_config": Config().to_dict(),
        "config_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}), encoding="utf8")


def photo_contracts(folder, nn_epochs, oracle_python=None, forbid_source=None):
    rows = []
    with sample_paths() as samples:
        chart, white, scene = [samples[name] for name in ("Image_1.JPG", "white.JPG", "Sample_1.JPG")]
        for method in ("ccm", "rpcc3", "nn", "root-pls", "hybrid-nn"):
            flat_options, grouped_options = fixture(method, nn_epochs)
            flat, grouped = check_config(flat_options), check_config(grouped_options)
            assert encoded(flat.to_dict()) == encoded(grouped.to_dict())
            before = encoded(flat.to_dict())
            ordinary, ordinary_metrics, ordinary_images = fitted(flat, chart, white)
            # Saving is an additional behavior, independent from the mathematical recipe.
            destination = folder / method
            grouped.save = True
            grouped.save_path = str(destination)
            current, metrics, images = fitted(grouped, chart, white)
            assert encoded(ordinary_metrics) == encoded(metrics)
            for stage in ("fit_FFC", "fit_GC", "fit_WB", "fit_CC"):
                np.testing.assert_array_equal(images[stage], ordinary_images[stage])
            expected = ordinary.predict_image(scene, stages="final")["CC"]
            actual = current.predict_image(scene, stages="final")["CC"]
            np.testing.assert_array_equal(actual, expected)
            saved = destination / "fit_models.ccpb"
            assert saved.is_file()
            replay = ColorCorrection().load_bundle(saved).predict_image(scene, stages="final")["CC"]
            np.testing.assert_array_equal(replay, actual)
            exported = write_image_data(destination / "corrected.png", actual, dtype="uint8",
                                        allow_precision_loss=True, png_profile="fast-lossless")
            assert exported.maximum_codec_error == 0
            resolved = current.fitted_config
            assert resolved.steps.cc.fit["model"] == ("pls" if method == "root-pls" else "nn" if method == "hybrid-nn" else method)
            assert resolved.numerics["precision"] == "native-byte"
            if oracle_python and method in ("ccm", "rpcc3", "nn"):
                spec = folder / (method + "-oracle.json")
                spec.write_text(json.dumps({"flat": before, "chart": str(chart), "white": str(white),
                    "scene": str(scene), "output": str(folder / (method + "-oracle"))}), encoding="utf8")
                log = folder / (method + "-oracle.log")
                with log.open("w", encoding="utf8") as stream:
                    result = subprocess.run([str(oracle_python), "-B", "-I", str(Path(__file__).resolve()),
                        "--oracle-worker", str(spec)], stdout=stream, stderr=subprocess.STDOUT)
                assert result.returncode == 0, log.read_text()
                identity = json.loads((folder / (method + "-oracle/identity.json")).read_text())
                assert identity["version"] == "2.0.0" and identity["grouped_api_present"] is False, identity
                assert encoded(Config().to_dict()) == identity["default_config"]
                if forbid_source:
                    assert forbid_source.resolve() not in Path(identity["origin"]).resolve().parents
                for stage in ("fit_FFC", "fit_GC", "fit_WB", "fit_CC"):
                    np.testing.assert_array_equal(images[stage], np.load(folder / (method + "-oracle") / (stage + ".npy"), allow_pickle=False))
                np.testing.assert_array_equal(actual, np.load(folder / (method + "-oracle/scene.npy"), allow_pickle=False))
            rows.append({"method": method, "stages_exact": True, "scene_exact": True,
                         "safe_bundle_replay_exact": True, "codec_error": 0,
                         "oracle_compared": bool(oracle_python and method in ("ccm", "rpcc3", "nn")),
                         "nn_epoch_cap": nn_epochs if method == "nn" else None,
                         "epochs_trained": current.models.model_cc[0].fit_report.get("details", {}).get("epochs"),
                         "training_chart_deltaE00": metrics["fit_CC"]["delta_e00_after_mean"]})
            print("Passed grouped/flat real images:", method, flush=True)

        # A changed WB knob must affect its own stage, even if a later CC refit
        # compensates that scale. A guard must reject insufficient measurements.
        options, grouped = fixture("ccm")
        original = check_config(grouped)
        changed = deepcopy(grouped)
        changed["steps"]["wb"]["fit"]["headroom"] = .85
        first, _, images = fitted(original, chart, white)
        other, _, altered = fitted(check_config(changed), chart, white)
        assert not np.array_equal(first.models.model_wb, other.models.model_wb)
        assert not np.array_equal(images["fit_WB"], altered["fit_WB"])
        scene_before = first.predict_image(scene, stages="final")["CC"]
        changed["steps"]["gc"]["quality"]["min_neutrals"] = 7
        _, _, failed = first.run(chart, white, name_="guarded", config=check_config(changed))
        assert failed and first.last_run["errors"]
        np.testing.assert_array_equal(first.predict_image(scene, stages="final")["CC"], scene_before)
        rows.append({"wb_headroom_changes_wb_stage": True, "gc_guard_rejected": True,
                     "failed_grouped_fit_preserves_prior_calibration": True})

        source_chart, source_white, source_scene = [read_image_data(path, color_space="srgb", frame_index=0)
                                                   for path in (chart, white, scene)]
        high_inputs = [ImageData.from_array(np.rint(item.to_srgb() * 65535).astype(np.uint16),
                                           color_space="srgb") for item in (source_chart, source_white, source_scene)]
        options, grouped = fixture("rpcc3")
        options["precision"] = "auto"
        options["CC_kwargs"].pop("use_lut")
        grouped["numerics"]["precision"] = "auto"
        grouped["steps"]["cc"]["apply"].pop("use_lut")
        flat = check_config(options)
        nested = Config.from_dict(check_config(grouped).to_grouped_dict())
        assert "use_lut" not in nested.CC_kwargs
        a, _, _ = fitted(flat, *high_inputs[:2])
        b, _, _ = fitted(nested, *high_inputs[:2])
        before = a.predict_image(high_inputs[2], stages="final")["CC"]
        after = b.predict_image(high_inputs[2], stages="final")["CC"]
        np.testing.assert_array_equal(before, after)
        assert b.models.metadata["fitting_policy"]["ui_runtime"] == "precision-f64-v1"
        bundle = folder / "high.ccpb"
        b.save_bundle(bundle)
        np.testing.assert_array_equal(after, ColorCorrection().load_bundle(bundle).predict_image(high_inputs[2], stages="final")["CC"])
        record = write_image_data(folder / "high.tiff", after, dtype="float64")
        assert record.maximum_codec_error == 0
        rows.append({"grouped_high_precision_exact": True, "high_replay_exact": True,
                     "auto_high_after_grouped_json_roundtrip": True,
                     "high_codec_error": 0, "note": "Byte JPEG codes expanded into uint16; no capture information added."})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--photos", action="store_true")
    parser.add_argument("--nn-epochs", type=int, default=30)
    parser.add_argument("--oracle-python", type=Path)
    parser.add_argument("--oracle-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--forbid-source", type=Path)
    args = parser.parse_args()
    if args.oracle_worker:
        oracle_worker(args.oracle_worker)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.nn_epochs < 1:
        parser.error("--nn-epochs must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import ColorCorrectionPipeline
    origin = Path(ColorCorrectionPipeline.__file__).resolve()
    if args.forbid_source:
        assert args.forbid_source.resolve() not in origin.parents, origin
    result = {"version": version("ColorCorrectionPipeline"), "origin": str(origin),
              "configurations": configuration_contracts()}
    if args.photos:
        artifacts = args.output.with_suffix("")
        artifacts.mkdir(parents=True, exist_ok=False)
        result["photos"] = photo_contracts(artifacts, args.nn_epochs, args.oracle_python, args.forbid_source)
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=json_value) + "\n", encoding="utf8")
    print("Passed grouped configuration contracts" + (" and real-image equivalence." if args.photos else "."))


if __name__ == "__main__":
    main()
