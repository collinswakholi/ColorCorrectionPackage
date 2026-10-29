"""Fresh-process frozen/candidate public pipeline and strict-export workload."""
import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

for name in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"


def measured(function):
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns()-started)/1e6


def plain(value):
    import numpy as np
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def stats(values):
    import numpy as np
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)), "raw_ms": values}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=plain), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["512", "1080p", "4k"])
    parser.add_argument("--modes", nargs="+", choices=["direct", "relaxed"], default=["direct", "relaxed"])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--kind", choices=["full", "cc_only"], default="full")
    parser.add_argument("--file-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output/"tmp"
    temporary.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    os.environ["NUMBA_CACHE_DIR"] = str(args.output/"numba-cache")
    os.environ["MPLCONFIGDIR"] = str(args.output/"matplotlib-cache")
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.site))
    import numpy as np
    from PIL import Image
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection, ImageData
    from ColorCorrectionPipeline.io import write_image_data
    from threadpoolctl import threadpool_info, threadpool_limits
    controller = threadpool_limits(limits=8)
    root = Path(package.__file__).parent
    model_path = args.models/(args.kind+".ccpb")
    pipeline = ColorCorrection().load_bundle(model_path)
    result = {"site": str(args.site), "package_file": str(package.__file__), "threads": 8,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.py")},
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "kind": args.kind, "threadpools": threadpool_info(), "runs": [], "preparation": {},
        "scope": "Public API; identical saved P78 portable calibration. Memory calls begin at float64 ndarray; file calls include decode and strict PNG8 atomic/fsync/byte/pixel verification. ImageData construction included unless explicitly marked reusable typed input."}
    try:
        import psutil
        result["host_memory"] = dict(psutil.virtual_memory()._asdict())
    except ImportError:
        pass
    for mode in args.modes:
        options = {"mode": "direct", "backend": "auto"} if mode == "direct" else {
            "mode": "combined-lut", "backend": "auto", "grid_sizes": (129,),
            "max_rgb_error": .006, "max_delta_e00": 1.0}
        prepared, elapsed = measured(lambda: pipeline.prepare_execution(options))
        assert prepared.preparation["qualification"]["selected_mode"] == ("direct" if mode == "direct" else "combined-lut")
        result["preparation"][mode] = {"ms": elapsed, "diagnostics": prepared.preparation}
        for size in args.sizes:
            path = args.inputs/(size+".png")
            image = None if args.file_only else np.asarray(Image.open(path).convert("RGB")).astype(np.float64)/255
            typed, typed_setup = (None, None) if args.file_only else measured(lambda: ImageData.from_array(image))
            warm = prepared.predict_image(path)
            del warm
            samples = {"memory_ms": [], "typed_ms": [], "file_predict_ms": [], "strict_export_ms": [], "file_total_ms": []}
            checks, first = None, None
            for repetition in range(args.repetitions):
                if not args.file_only:
                    memory, elapsed = measured(lambda: prepared.predict_image(image))
                    samples["memory_ms"].append(elapsed)
                    if repetition == 0:
                        checks = {"shape": list(memory.images["CC"].shape), "dtype": str(memory.images["CC"].dtype),
                            "memory_pixels_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(memory.images["CC"]))).hexdigest(),
                            "diagnostics": memory.diagnostics}
                    del memory
                    gc.collect()
                    reused, elapsed = measured(lambda: prepared.predict_image(typed))
                    samples["typed_ms"].append(elapsed)
                    if repetition == 0:
                        checks["typed_pixels_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(reused.images["CC"]))).hexdigest()
                    del reused
                    gc.collect()
                def file_workflow():
                    prediction, predict_ms = measured(lambda: prepared.predict_image(path))
                    exported, export_ms = measured(lambda: write_image_data(args.output/f"{mode}-{size}-{repetition}.png", prediction.images["CC"], dtype="uint8", allow_precision_loss=True))
                    return prediction, exported, predict_ms, export_ms
                (prediction, exported, predict_ms, export_ms), elapsed = measured(file_workflow)
                samples["file_total_ms"].append(elapsed)
                samples["file_predict_ms"].append(predict_ms)
                samples["strict_export_ms"].append(export_ms)
                if repetition == 0:
                    if checks is None:
                        checks = {"shape": list(prediction.images["CC"].shape), "dtype": str(prediction.images["CC"].dtype)}
                    first = exported.to_dict()
                    checks["file_pixels_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(prediction.images["CC"]))).hexdigest()
                    checks["file_diagnostics"] = prediction.diagnostics
                Path(exported.path).unlink()
                del prediction, exported
                gc.collect()
            row = {"mode": mode, "size": size, "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "reused_ImageData_setup_ms": typed_setup, "timings": {key: stats(values) for key, values in samples.items() if values},
                "checks": checks, "export_record": first}
            result["runs"].append(row)
            write(args.output/"results.json", result)
            print(mode, size, {key: value["median_ms"] for key, value in row["timings"].items()}, flush=True)
            del image, typed
            gc.collect()
        prepared.close()
    result["source_unchanged"] = result["source_sha256"] == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.py")}
    assert result["source_unchanged"]
    write(args.output/"results.json", result)


if __name__ == "__main__":
    main()
