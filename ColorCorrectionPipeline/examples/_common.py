"""Shared file handling for the executable tutorials."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
from importlib import resources
import json
from pathlib import Path
import time

import numpy as np

from ColorCorrectionPipeline import ColorCorrection
from ColorCorrectionPipeline.io import write_image_data

SAMPLES = {
    "Image_1.JPG": "3d4450764d415891fd4b2302d51df053b9b20dbf82b39170e7bcbb3dd875a51c",
    "white.JPG": "2bb756a51a54619b5976293dca20e4958e0d93d5e621fae2d3f67f1c1ae819be",
    "Sample_1.JPG": "9fe40926962bb450d8f4c41f4c3e276d1c5fe5a07f50fc82125096378ea36d99",
}


@contextmanager
def sample_paths():
    """Keep extracted wheel resources alive for the whole example run."""
    with ExitStack() as stack:
        paths = {}
        for name, digest in SAMPLES.items():
            if resources.is_resource("ColorCorrectionPipeline.examples.data", name):
                path = stack.enter_context(resources.path("ColorCorrectionPipeline.examples.data", name))
            else:
                path = Path(__file__).resolve().parents[2] / "Images" / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Sample resource is missing or changed: " + name)
            paths[name] = path
        yield paths


def input_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--chart", type=Path, help="Classic24 chart capture. Omit to use the sample.")
    parser.add_argument("--white", type=Path, help="Matching flat-field capture.")
    parser.add_argument("--scene", type=Path, help="Image captured under the calibrated conditions.")
    parser.add_argument("--no-white", action="store_true", help="Disable flat-field correction.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"),
                        help="Chart detection backend. Color fitting uses its recorded CPU profile.")
    return parser


def inputs(args, samples):
    if args.no_white and args.white:
        raise ValueError("Choose --white or --no-white.")
    chart = args.chart or samples["Image_1.JPG"]
    scene = args.scene or samples["Sample_1.JPG"]
    white = None if args.no_white else args.white
    if white is None and args.chart is None and not args.no_white:
        white = samples["white.JPG"]
    return chart, white, scene


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def fit(chart, white, config):
    pipeline = ColorCorrection()
    started = time.perf_counter()
    metrics, _, failed = pipeline.run(chart, white, name_="chart", config=config)
    seconds = time.perf_counter() - started
    if failed:
        raise RuntimeError(pipeline.last_run["errors"])
    return pipeline, metrics, seconds


def export_and_replay(pipeline, scene, output, dtype="uint8"):
    """Check the saved evaluator before exporting the replayed image."""
    bundle = output / "calibration.ccpb"
    pipeline.save_bundle(bundle)
    started = time.perf_counter()
    direct = pipeline.predict_image(scene, stages="final")["CC"]
    apply_seconds = time.perf_counter() - started
    restored = ColorCorrection().load_bundle(bundle)
    replay = restored.predict_image(scene, stages="final")["CC"]
    np.testing.assert_array_equal(replay, direct)
    suffix = ".png" if dtype in ("uint8", "uint16") else ".tiff"
    started = time.perf_counter()
    record = write_image_data(output / ("corrected" + suffix), replay, dtype=dtype,
                              allow_precision_loss=True,
                              png_profile="fast-lossless" if suffix == ".png" else "compatible")
    export_seconds = time.perf_counter() - started
    if record.maximum_codec_error != 0:
        raise RuntimeError("Strict export readback changed stored samples.")
    return replay, {"bundle": str(bundle), "corrected": str(record.path),
                    "replay_exact": True, "maximum_codec_error": record.maximum_codec_error,
                    "apply_seconds": apply_seconds, "verified_export_seconds": export_seconds}


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def report(output, result):
    (output / "results.json").write_text(json.dumps(result, indent=2, default=json_value) + "\n",
                                          encoding="utf8")
    print(json.dumps(result, indent=2, default=json_value))


def scores(metrics):
    cc = metrics["chart_CC"]
    return {"training_chart_de00_before_cc": cc["delta_e00_before_mean"],
            "training_chart_de00_after_cc": cc["delta_e00_after_mean"]}

