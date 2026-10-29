"""Image-contract/strict-persistence timings; no verification is disabled.

Run in a fresh process with one explicit package site. Timed public calls are
separate from fixture generation, garbage collection, profiling and validation.
"""
import argparse
import cProfile
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
import tracemalloc

for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"


def timed(function):
    started = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns()-started)/1e6


def stats(values):
    import numpy as np
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)), "raw_ms": values}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", default=["512", "1080p", "4k"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--operations", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / "tmp"
    temporary.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib-cache")
    os.environ["NUMBA_CACHE_DIR"] = str(args.output / "numba-cache")
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.site))
    import importlib.metadata as metadata
    import numpy as np
    from PIL import Image
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ImageData
    from ColorCorrectionPipeline.io import read_image_data, write_image_data
    from threadpoolctl import threadpool_info, threadpool_limits
    controller = threadpool_limits(limits=8)
    root = Path(package.__file__).parent
    result = {"python": platform.python_version(), "executable": sys.executable,
        "site": str(args.site), "package_file": str(package.__file__), "threads": 8,
        "threadpools": threadpool_info(), "versions": {}, "inputs": {}, "runs": [],
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in root.rglob("*.py")},
        "scope": "Public calls including all validation/hash/persistence work; fixture creation and gc excluded. Outputs deleted after timed intervals.",
        "profiles": {}, "memory": {}}
    for distribution in ("numpy", "scipy", "Pillow", "opencv-contrib-python", "colour-science", "torch", "numba", "tifffile"):
        try:
            result["versions"][distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result["versions"][distribution] = None
    try:
        import psutil
        result["host_memory"] = dict(psutil.virtual_memory()._asdict())
        result["initial_rss"] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    out = args.output / "results.json"
    for size in args.sizes:
        path = args.inputs / (size + ".png")
        u8 = np.asarray(Image.open(path).convert("RGB")).copy()
        floating = u8.astype(np.float64)/255
        u16 = u8.astype(np.uint16)*257
        typed_float = ImageData.from_array(floating)
        typed_u8 = ImageData.from_array(u8)
        result["inputs"][size] = {"path": str(path), "shape": list(u8.shape),
                                  "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        cases = {
            "from_array_float64": lambda: ImageData.from_array(floating),
            "from_array_uint8": lambda: ImageData.from_array(u8),
            "constructor_float64": lambda: ImageData(floating),
            "typed_to_srgb": lambda: typed_float.to_srgb(),
            "read_png": lambda: read_image_data(path),
            "strict_png8_float64": lambda: write_image_data(args.output/"frame.png", floating, dtype="uint8", allow_precision_loss=True),
            "strict_png8_typed_float64": lambda: write_image_data(args.output/"frame.png", typed_float, dtype="uint8", allow_precision_loss=True),
            "strict_png8_typed_uint8": lambda: write_image_data(args.output/"frame.png", typed_u8),
            "strict_png16_uint16": lambda: write_image_data(args.output/"frame.png", u16),
            "strict_tiff_float64": lambda: write_image_data(args.output/"frame.tiff", floating),
        }
        for name, function in cases.items():
            if args.operations and name not in args.operations:
                continue
            measurements = []
            record = None
            for repetition in range(args.repetitions+1):
                value, elapsed = timed(function)
                if repetition:
                    measurements.append(elapsed)
                if hasattr(value, "pixel_digest"):
                    record = {"digest": value.digest, "pixel_digest": value.pixel_digest,
                              "shape": list(value.shape)}
                elif hasattr(value, "to_dict"):
                    record = value.to_dict()
                if hasattr(value, "path"):
                    Path(value.path).unlink()
                del value
                gc.collect()
            result["runs"].append({"size": size, "operation": name, "timing": stats(measurements), "record": record})
            print(size, name, result["runs"][-1]["timing"]["median_ms"], flush=True)
            write_json(out, result)
        if size == "4k" and args.profile:
            for name in ("from_array_float64", "read_png", "strict_png8_float64", "strict_png8_typed_uint8"):
                profiler = cProfile.Profile()
                profiler.enable()
                value = cases[name]()
                profiler.disable()
                if hasattr(value, "path"):
                    Path(value.path).unlink()
                del value
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(45)
                profiler.dump_stats(str(args.output/(name+".prof")))
                (args.output/(name+"-profile.txt")).write_text(stream.getvalue(), encoding="utf8")
                result["profiles"][name] = stream.getvalue()
                gc.collect()
        if size == "4k" and args.memory:
            for name in ("from_array_float64", "read_png", "strict_png8_float64", "strict_png8_typed_uint8"):
                gc.collect()
                tracemalloc.start()
                value = cases[name]()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                result["memory"][name] = {"python_numpy_traced_peak_bytes": peak,
                                          "traced_live_bytes_at_return": current,
                                          "note": "tracemalloc includes NumPy allocations exposed to Python tracing, not every native decoder allocation"}
                if hasattr(value, "path"):
                    Path(value.path).unlink()
                del value
        del cases, floating, u8, u16, typed_float, typed_u8
        gc.collect()
        write_json(out, result)
    result["source_unchanged"] = result["source_sha256"] == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                            for path in root.rglob("*.py")}
    assert result["source_unchanged"]
    write_json(out, result)


if __name__ == "__main__":
    main()
