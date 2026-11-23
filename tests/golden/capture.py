"""Capture or compare characterization ("golden") outputs of the public pipeline.

The reference data was captured from the 2.2.0 release (main @ b398361) before any
review change, so later refactors can prove they preserve behaviour.

    python tests/golden/capture.py --write     # regenerate (only on intentional change)
    python tests/golden/capture.py             # compare against stored data

pytest drives the same scenarios through ``tests/test_golden.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
IMAGES = ROOT / "Images"
THUMB = (48, 64)


def _scenarios():
    from ColorCorrectionPipeline import CCOptions, Config, FFCOptions, Steps

    chart = {"device": "cpu"}
    one = {"frame_index": 0}  # sample JPEGs carry a second (preview) frame
    return {
        "v2_default": lambda: Config(chart=chart, input=one),
        "v2_rpcc2_grouped": lambda: Config(
            steps=Steps(ffc=FFCOptions(enabled=True), cc=CCOptions(fit={"model": "rpcc2"})),
            input={"frame_index": 0},
            chart=chart,
        ),
        "v2_ccm_3x3": lambda: Config(model="3x3", chart=chart, input=one),
        "v2_pls": lambda: Config(model="pls", chart=chart, input=one),
        "v2_high_precision": lambda: Config(
            steps=Steps(cc=CCOptions(fit={"model": "rpcc2"})),
            numerics={"precision": "high"},
            chart=chart,
            input=one,
        ),
        "v2_seeded_samples": lambda: Config(
            model="3x3",
            chart={"device": "cpu", "samples_per_patch": 20, "sampling_seed": 42},
            input=one,
        ),
        "v1_legacy_conv": lambda: Config(
            fitting_policy="legacy-v1",
            do_ffc=False,
            do_gc=False,
            do_wb=False,
            check_saturation=False,
            chart_detection=chart,
            CC_kwargs={"cc_method": "conv", "method": "Finlayson 2015", "degree": 2},
        ),
        "v1_legacy_custom_linear": lambda: Config(
            fitting_policy="legacy-v1",
            do_ffc=False,
            do_gc=False,
            do_wb=False,
            check_saturation=False,
            chart_detection=chart,
            CC_kwargs={
                "mtd": "custom",
                "degree": 2,
                "hidden_layers": [64],
                "max_iterations": 100,
                "random_state": 42,
                "use_lut": False,
            },
        ),
        # Error path: an unknown regression method must fail the run, not raise.
        "v1_invalid_mtd_fails": lambda: Config(
            fitting_policy="legacy-v1",
            do_ffc=False,
            do_gc=False,
            do_wb=False,
            check_saturation=False,
            chart_detection=chart,
            CC_kwargs={"mtd": "conv"},
        ),
        "v1_legacy_flat_all_steps": lambda: Config(
            do_ffc=True,
            do_gc=True,
            do_wb=True,
            do_cc=True,
            chart_detection=chart,
            CC_kwargs={
                "cc_method": "ours",
                "method": "Finlayson 2015",
                "mtd": "linear",
                "degree": 2,
                "random_state": 42,
            },
        ),
    }


def _scalars(obj, prefix=""):
    """Flatten numeric metric leaves to {path: float}."""
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_scalars(value, f"{prefix}{key}."))
    elif isinstance(obj, (bool, np.bool_)):
        pass
    elif isinstance(obj, (int, float, np.integer, np.floating)) and np.isfinite(obj):
        out[prefix[:-1]] = float(obj)
    return out


def _thumb(image):
    import cv2

    arr = np.asarray(image, dtype=np.float64)
    if arr.dtype.kind in "ui" or arr.max() > 1.5:
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    return cv2.resize(arr, THUMB[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)


def run_scenario(name):
    from ColorCorrectionPipeline import ColorCorrection

    config = _scenarios()[name]()
    pipe = ColorCorrection()
    chart, white, scene = (str(IMAGES / n) for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics, images, failed = pipe.run(chart, white, name_="chart", config=config)
        predicted = pipe.predict_image(scene)
    final_key = [k for k in ("CC", "WB", "GC", "FFC") if k in predicted][0]
    final = np.asarray(predicted[final_key])
    return {
        "failed": bool(failed),
        "image_keys": sorted(images),
        "predict_keys": sorted(predicted),
        "final_key": final_key,
        "final_dtype": str(final.dtype),
        "final_shape": list(final.shape),
        "final_mean": [float(v) for v in final.reshape(-1, final.shape[-1]).mean(0)],
        "metrics": _scalars(metrics),
    }, _thumb(final)


def compare(name, record, thumb, *, metric_rtol=1e-6, thumb_atol=1e-4):
    ref = json.loads((DATA / f"{name}.json").read_text())
    ref_thumb = np.load(DATA / f"{name}.npy")
    problems = []
    for key in ("failed", "image_keys", "predict_keys", "final_key", "final_dtype", "final_shape"):
        if record[key] != ref[key]:
            problems.append(f"{key}: {record[key]!r} != {ref[key]!r}")
    if not np.allclose(record["final_mean"], ref["final_mean"], rtol=1e-6, atol=1e-6):
        problems.append(f"final_mean {record['final_mean']} != {ref['final_mean']}")
    missing = sorted(set(ref["metrics"]) - set(record["metrics"]))
    if missing:
        problems.append(f"missing metrics: {missing[:8]}")
    for key, value in ref["metrics"].items():
        if "seconds" in key or "time" in key or key not in record["metrics"]:
            continue
        got = record["metrics"][key]
        if not np.isclose(got, value, rtol=metric_rtol, atol=1e-9):
            problems.append(f"metric {key}: {got} != {value}")
    diff = float(np.max(np.abs(thumb - ref_thumb)))
    if diff > thumb_atol:
        problems.append(f"thumbnail max abs diff {diff:.3g} > {thumb_atol}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    names = args.names or list(_scenarios())
    DATA.mkdir(exist_ok=True)
    status = 0
    for name in names:
        record, thumb = run_scenario(name)
        if args.write:
            (DATA / f"{name}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
            np.save(DATA / f"{name}.npy", thumb)
            print(f"wrote {name}: failed={record['failed']} metrics={len(record['metrics'])}")
        else:
            problems = compare(name, record, thumb)
            print(f"{name}: {'OK' if not problems else 'DIFF'}")
            for p in problems[:20]:
                print("   ", p)
            status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main())
