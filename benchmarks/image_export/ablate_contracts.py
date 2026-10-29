"""Process-local exploratory ablations; never edits package source."""
import argparse
import gc
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

for name in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(args.output/"mpl")
    os.environ["NUMBA_CACHE_DIR"] = str(args.output/"numba")
    os.environ["TEMP"] = os.environ["TMP"] = str(args.output)
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.site))
    import cv2
    import numpy as np
    from PIL import Image
    from ColorCorrectionPipeline import ImageData
    import ColorCorrectionPipeline.contracts as contracts
    from threadpoolctl import threadpool_limits
    controller = threadpool_limits(limits=8)
    u8 = np.asarray(Image.open(args.input).convert("RGB")).copy()
    f64 = u8.astype(np.float64)/255
    f64_nonunit = u8.astype(np.float64)
    original_hash, original_normalize = contracts._image_digests, contracts._normalized_image_snapshot
    def sequential(metadata, pixels):
        return contracts._hash_parts({"color_space": metadata["color_space"], "channel_order": "RGB"}, (pixels,)), contracts._hash_parts(metadata, (pixels,))
    def preallocated(source, lo, hi):
        with io.BytesIO() as stream:
            stream.seek(source.size*8-1); stream.write(b"\0"); stream.seek(0)
            with np.nditer(source, flags=["external_loop", "buffered"], op_flags=["readonly"], op_dtypes=[np.dtype("<f8")], order="C", buffersize=65536) as iterator:
                for block in iterator:
                    values = np.array(block, dtype=np.dtype("<f8"), copy=True)
                    low, high = contracts._image_bounds(values)
                    if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
                        raise ValueError("out of bounds")
                    values -= lo; values /= hi-lo
                    stream.write(memoryview(values).cast("B"))
            payload = stream.getvalue()
        return np.frombuffer(payload, dtype=np.dtype("<f8")).reshape(source.shape)
    def full_inplace(source, lo, hi):
        values = np.array(source, dtype=np.dtype("<f8"), copy=True, order="C")
        low, high = contracts._image_bounds(values)
        if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
            raise ValueError("out of bounds")
        values -= lo; values /= hi-lo
        return values
    def native_snapshot(source, lo, hi):
        captured = np.array(source, copy=True, order="C")
        low, high = contracts._image_bounds(captured)
        if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
            raise ValueError("out of bounds")
        values = captured.astype(np.dtype("<f8"), copy=False)
        values -= lo; values /= hi-lo
        return values
    data = {"shape": list(u8.shape), "cv2_default_threads": cv2.getNumThreads(),
            "cv2_parallel_framework": [line for line in cv2.getBuildInformation().splitlines() if "Parallel framework" in line], "runs": []}
    try:
        import psutil
        data["host_memory"] = dict(psutil.virtual_memory()._asdict())
        data["initial_rss"] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    expected = {}
    def run(name, function, memory=False):
        times, cpus = [], []
        for repetition in range(4):
            gc.collect()
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            value = function()
            elapsed, cpu_ms = (time.perf_counter_ns()-wall)/1e6, (time.process_time_ns()-cpu)/1e6
            if repetition:
                times.append(elapsed); cpus.append(cpu_ms)
            if isinstance(value, ImageData):
                case = name.split("/")[-1]
                signature = (value.digest, value.pixel_digest)
                if case in expected:
                    assert signature == expected[case], name
                expected[case] = signature
            del value
        record = {"name": name, "median_ms": statistics.median(times), "raw_ms": times,
                  "cpu_raw_ms": cpus, "median_cpu_ms": statistics.median(cpus)}
        if memory:
            gc.collect(); tracemalloc.start(); value = function()
            record["peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop(); del value
        data["runs"].append(record)
        (args.output/"results.json").write_text(json.dumps(data, indent=2), encoding="utf8")
        print(name, record["median_ms"], record["median_cpu_ms"], flush=True)
    for hash_name, hash_function in (("parallel", original_hash), ("sequential", sequential)):
        contracts._image_digests = hash_function
        run(hash_name+"/float64_unit", lambda: ImageData.from_array(f64), True)
    for name, normalizer in (("growing", original_normalize), ("preallocated", preallocated), ("full_inplace", full_inplace), ("native_snapshot", native_snapshot)):
        contracts._image_digests = original_hash
        contracts._normalized_image_snapshot = normalizer
        run(name+"/uint8", lambda: ImageData.from_array(u8), True)
        run(name+"/float64_nonunit", lambda: ImageData.from_array(f64_nonunit, value_range=(0, 255)), True)
    contracts._image_digests, contracts._normalized_image_snapshot = original_hash, original_normalize
    typed = ImageData.from_array(f64)
    run("opencv_before/to_srgb_copy", typed.to_srgb)
    run("opencv_before/encode", lambda: cv2.imencode(".png", u8, [cv2.IMWRITE_PNG_COMPRESSION, 0]))
    cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    run("opencv_after/to_srgb_copy", typed.to_srgb)
    run("opencv_after/encode", lambda: cv2.imencode(".png", u8, [cv2.IMWRITE_PNG_COMPRESSION, 0]))
    cv2.setNumThreads(1)
    run("opencv_one_thread/to_srgb_copy", typed.to_srgb)
    run("opencv_one_thread/encode", lambda: cv2.imencode(".png", u8, [cv2.IMWRITE_PNG_COMPRESSION, 0]))


if __name__ == "__main__":
    main()
