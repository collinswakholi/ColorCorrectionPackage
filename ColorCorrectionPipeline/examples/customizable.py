"""An editable, complete ColorCorrectionPipeline 2.x calibration workflow.

Install the package with pip install ColorCorrectionPipeline. This example
requires the 2.x API. Run the sample captures with:

    python -m ColorCorrectionPipeline.examples.customizable --output my-run

Edit SETTINGS below, or create a JSON file and edit it:

    python -m ColorCorrectionPipeline.examples.customizable --write-settings settings.json
    python -m ColorCorrectionPipeline.examples.customizable --settings settings.json --output my-run

JSON top-level keys override SETTINGS. A supplied calibration dictionary or
exports list replaces that entire section. The starter uses the explicit
grouped-v1 layout; earlier flat calibration dictionaries remain accepted.
Chart, white, scene, output_directory and calibration persistence.directory
(or the earlier save_path alias) are relative to the JSON file. Nested detector/
model paths follow Config's working-directory rules. The --output argument is
relative to the current working directory.

Set use_sample_images=False and provide chart, white and scenes for your own
Classic24 captures. Set calibration.steps.ffc.enabled=False when no matching
white exists. The earlier calibration.do_ffc alias is still accepted.
Enabled FFC requires scenes with the calibrated spatial dimensions.

Replace calibration.steps.cc.fit with one of these dictionaries:

    {"model": "3x3"}
    {"model": "linear", "basis": "polynomial", "degree": 4}
    {"model": "pls", "basis": "root", "degree": 3,
     "pls": {"components": 6}}
    {"model": "nn", "basis": "hybrid", "degree": 4,
     "nn": {"hidden_layers": [100, 32], "max_iterations": 10000,
            "patience": 1500, "learning_rate": 0.001,
            "optim_type": "Adam", "dropout_rate": 0.0,
            "use_batch_norm": False}}

Use calibration.numerics.precision="high" when an 8-bit chart will serve
uint16/float scenes. Auto precision resolves from the chart and enabled white.
Keep steps.cc.apply.use_lut omitted for automatic precision selection; explicit
high precision requires direct evaluation. The saved bundle governs application.

Optional reference settings are keyword arguments for
ReferenceProfile.from_values. Supply values, color_space, white, scale,
profile_id, revision, names and neutral_ids in the declared chart order.
Retained physical reference coordinates are enabled by default here.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config, ReferenceProfile
from ColorCorrectionPipeline.io import write_image_data
from ._common import json_value, new_output, sample_paths


# This is the only source of example defaults. Config validates calibration options.
SETTINGS = {
    "use_sample_images": True,
    "chart": None,
    "white": None,
    "scenes": [],
    "output_directory": "customized-results",
    "calibration": {
        "layout": "grouped-v1",
        "api_version": 2,
        "steps": {
            "ffc": {
                "enabled": True,
                "fit": {"method": "linear", "degree": 1, "bins": 30, "smooth_window": 7},
                # Optional manual crop belongs in white_plane.crop_rect (x1,y1,x2,y2).
            },
            "gc": {
                "enabled": True,
                # GC fits its neutral curve automatically; these are measured-data guards.
                "quality": {"min_neutrals": 3, "min_span": 20.0,
                            "duplicate_tolerance": 1e-6, "target_tolerance": 1.0},
            },
            "wb": {
                "enabled": True,
                "fit": {"headroom": 0.95},
                "quality": {"min_signal": 0.01, "max_gain": 8.0},
            },
            "cc": {
                "enabled": True,
                "fit": {"model": "rpcc2"},
                # LUT rendering is separate from training. Omit use_lut for auto precision.
                "apply": {"grid_size": 33, "min_pixels": 1, "lazy_lut": True},
                "validation": {"enabled": True, "n_splits": 4, "random_state": 42},
            },
        },
        "input": {"frame_index": 0, "icc_policy": "reject"},
        # GPU-first chart detection can recover to CPU. Fitting uses CPU float64.
        "chart": {"device": "auto", "allow_cpu_fallback": True},
        "color": {
            "version": "explicit-v1",
            "input_encoding": "srgb",
            "output_encoding": "srgb",
            "fitting_domain": "linear-rgb",
        },
        "numerics": {"precision": "auto"},
        "observations": {"saturation": "report", "max_clipped_fraction": 0.0},
        # Automatic run-time saving is off; the workflow explicitly saves its safe bundle.
        "persistence": {"enabled": False, "format": "bundle"},
    },
    # Optional physical reference data passed to ReferenceProfile.from_values.
    # A canonical profile may instead be supplied in calibration.reference.
    "reference": None,
    "exports": [
        {
            "filename": "preview.png",
            "dtype": "uint8",
            "allow_precision_loss": True,
            "png_profile": "fast-lossless",
        },
        {
            "filename": "corrected.tiff",
            "dtype": "float64",
            "allow_precision_loss": False,
        },
    ],
}


def settings_from_file(path):
    settings = deepcopy(SETTINGS)
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(supplied, dict) or set(supplied) - set(settings):
            raise ValueError("Settings must be an object containing the documented top-level keys.")
        settings.update(supplied)
    if not isinstance(settings["calibration"], dict):
        raise ValueError("calibration must be an object of Config keyword arguments.")
    return settings


def input_paths(settings, base, stack, config):
    """Keep extracted sample files alive until every scene has been processed."""
    if type(settings["use_sample_images"]) is not bool:
        raise ValueError("use_sample_images must be boolean")
    if settings["use_sample_images"]:
        if settings["chart"] or settings["white"] or settings["scenes"]:
            raise ValueError("Set use_sample_images=false when providing capture paths.")
        samples = stack.enter_context(sample_paths())
        white = samples["white.JPG"] if config.do_ffc else None
        return samples["Image_1.JPG"], white, [samples["Sample_1.JPG"]]

    def existing(value):
        path = base / Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    if not settings["chart"] or not isinstance(settings["scenes"], list) or not settings["scenes"]:
        raise ValueError("Provide a chart path and a nonempty scenes list.")
    chart = existing(settings["chart"])
    needs_white = config.do_ffc
    white = existing(settings["white"]) if needs_white and settings["white"] else None
    scenes = [existing(scene) for scene in settings["scenes"]]
    if needs_white and white is None:
        raise ValueError("Provide white or disable calibration.steps.ffc.enabled (earlier alias: do_ffc).")
    return chart, white, scenes


def export_options(entries, encoding):
    """Restrict filenames to the new scene directory and declare actual pixel encoding."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("exports must be a nonempty list.")
    exports = []
    filenames = set()
    for entry in entries:
        if not isinstance(entry, dict) or "filename" not in entry:
            raise ValueError("Each export must be an object with a filename.")
        options = dict(entry)
        name = options.pop("filename")
        if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in "/\\:"):
            raise ValueError("Each export filename must be a basename without path separators.")
        if name.casefold() in filenames:
            raise ValueError("Export filenames must be distinct.")
        filenames.add(name.casefold())
        if options.get("input_encoding", encoding) != encoding:
            raise ValueError("Export input_encoding contradicts the calibrated output encoding.")
        options["input_encoding"] = encoding
        exports.append((name, options))
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--settings", type=Path, help="JSON overrides for SETTINGS.")
    source.add_argument("--write-settings", type=Path, help="Write a new starter JSON file and exit.")
    parser.add_argument("--output", type=Path, help="New output directory.")
    args = parser.parse_args()

    if args.write_settings:
        with args.write_settings.open("x", encoding="utf8") as stream:
            json.dump(SETTINGS, stream, indent=2)
            stream.write("\n")
        print("Edit settings in", args.write_settings.resolve())
        return

    settings = settings_from_file(args.settings)
    base = args.settings.resolve().parent if args.settings else Path.cwd()
    parameters = deepcopy(settings["calibration"])
    if settings["reference"] is not None:
        if parameters.get("reference_profile") is not None or parameters.get("reference") is not None:
            raise ValueError("Choose reference or calibration.reference/profile.")
        reference = dict(settings["reference"])
        reference.setdefault("retain_source", True)
        parameters["reference_profile"] = ReferenceProfile.from_values(**reference)
    # Explicit versioning avoids historical interpretation of unversioned JSON.
    # from_dict also restores JSON-compatible array options such as REF_ILLUMINANT.
    config = Config.from_dict(parameters, source_version=2)
    if config.save_path is not None:
        config.save_path = str(base / Path(config.save_path))
    config.validate()
    exports = export_options(settings["exports"], config.color_semantics["output_encoding"])
    output = args.output if args.output is not None else base / Path(settings["output_directory"])

    with ExitStack() as stack:
        chart, white, scenes = input_paths(settings, base, stack, config)
        new_output(output)
        pipeline = ColorCorrection()
        started = time.perf_counter()
        fit = pipeline.fit_or_raise(chart, white, name_="chart", config=config)
        calibration_seconds = time.perf_counter() - started

        # A safe bundle owns the fitted recipe, including the resolved precision.
        bundle = output / "calibration.ccpb"
        pipeline.save_bundle(bundle)
        restored = ColorCorrection().load_bundle(bundle)
        results = []
        for index, scene in enumerate(scenes, start=1):
            folder = new_output(output / ("scene-%03d" % index))
            started = time.perf_counter()
            corrected = restored.predict_image(scene, stages="final")["CC"]
            apply_seconds = time.perf_counter() - started
            direct = pipeline.predict_image(scene, stages="final")["CC"]
            np.testing.assert_array_equal(corrected, direct)
            records = []
            for filename, options in exports:
                started = time.perf_counter()
                record = write_image_data(folder / filename, corrected, **options)
                records.append({"export": record.to_dict(),
                                "verified_export_seconds": time.perf_counter() - started})
            results.append({"input": str(scene), "apply_seconds": apply_seconds,
                            "bundle_replay_exact": True, "outputs": records})

        result = {
            "bundle": str(bundle),
            "calibration_seconds": calibration_seconds,
            "requested_config": config.to_dict(),
            "effective_fitting_policy": pipeline.models.metadata["fitting_policy"],
            "numerical_report": pipeline.numerical_report,
            "training_chart_metrics": fit.metrics,
            "metric_scope": "Training-chart scores; when enabled, CC-only held-out colors "
                            "conditional on fitted upstream stages. Scene pixels have no "
                            "measured reference.",
            "scenes": results,
        }
        (output / "results.json").write_text(
            json.dumps(result, indent=2, default=json_value, allow_nan=False) + "\n", encoding="utf8")
        print("Calibration:", bundle.resolve())
        print("Effective fitting policy:", pipeline.models.metadata["fitting_policy"])
        print("Calibration seconds:", round(calibration_seconds, 3))
        print("Results:", (output / "results.json").resolve())
        for item in results:
            print("Scene:", item["input"], "apply seconds:", round(item["apply_seconds"], 3))


if __name__ == "__main__":
    main()
