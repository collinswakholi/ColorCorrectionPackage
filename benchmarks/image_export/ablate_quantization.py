"""Untouched app: compare strict integer quantization workspace reuse."""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"quantization-ablation"
OUT.mkdir(exist_ok=False)
for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["TEMP"] = os.environ["TMP"] = str(OUT)
os.environ["MPLCONFIGDIR"] = str(OUT/"mpl")
os.environ["NUMBA_CACHE_DIR"] = str(OUT/"numba")
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage")
import numpy as np
from PIL import Image
from ColorCorrectionPipeline.io.writers import _EXPORT_CHUNK_VALUES
from threadpoolctl import threadpool_limits
controller = threadpool_limits(limits=8)

def _quantize_strict(values, dtype, maximum):
    # Preserve the exact intermediate implementation used as A in the recorded
    # experiment, so replay still compares the same algorithms after B ships.
    values_flat = values.reshape(-1)
    storage = values if dtype == values.dtype else np.empty(values.shape, dtype=dtype)
    storage_flat = storage.reshape(-1)
    quantization = 0.0
    for start in range(0, values_flat.size, _EXPORT_CHUNK_VALUES):
        stop = min(start+_EXPORT_CHUNK_VALUES, values_flat.size)
        source = values_flat[start:stop]
        if dtype.kind == "u":
            scaled = source * maximum
            np.rint(scaled, out=scaled)
            storage_flat[start:stop] = scaled.astype(dtype)
        elif storage is not values:
            storage_flat[start:stop] = source.astype(dtype)
        if storage is not values:
            error = np.abs(storage_flat[start:stop].astype(np.float64) / maximum - source)
            quantization = max(quantization, float(error.max()))
    return storage, quantization

def reuse(values, dtype, maximum):
    if dtype.kind != "u":
        return _quantize_strict(values, dtype, maximum)
    source_flat = values.reshape(-1)
    storage = np.empty(values.shape, dtype=dtype)
    storage_flat = storage.reshape(-1)
    quantization = 0.0
    for start in range(0, source_flat.size, _EXPORT_CHUNK_VALUES):
        stop = min(start+_EXPORT_CHUNK_VALUES, source_flat.size)
        source = source_flat[start:stop]
        scaled = source * maximum
        np.rint(scaled, out=scaled)
        storage_flat[start:stop] = scaled.astype(dtype)
        scaled /= maximum
        scaled -= source
        np.abs(scaled, out=scaled)
        quantization = max(quantization, float(scaled.max()))
    return storage, quantization

def signature(value):
    storage, error = value
    return hashlib.sha256(memoryview(storage)).hexdigest(), struct.pack("!d", error).hex()

data = {"runs": [], "edge_checks": [], "changed_app": False}
for dtype, maximum in ((np.dtype("uint8"), 255.), (np.dtype("uint16"), 65535.)):
    integer = np.arange(int(maximum)+1, dtype=np.float64)
    edges = np.concatenate((integer/maximum, (integer[:-1]+.5)/maximum, np.nextafter((integer[:-1]+.5)/maximum, 0.), np.nextafter((integer[:-1]+.5)/maximum, 1.), np.array([-0., 0., np.nextafter(0., 1.), np.nextafter(1., 0.), 1.])))
    edge = np.resize(edges, edges.size+(-edges.size % 3)).reshape(1, -1, 3)
    for name, values in (("ties-and-neighbors", edge), ("strided", edge[:, ::-1, ::-1])):
        a, b = _quantize_strict(values, dtype, maximum), reuse(values, dtype, maximum)
        assert np.array_equal(a[0], b[0]) and struct.pack("!d", a[1]) == struct.pack("!d", b[1]), (dtype, name)
        data["edge_checks"].append({"dtype": str(dtype), "case": name, "values": values.size, "exact": True})
pixels = np.asarray(Image.open(ROOT.parent.parent/"ccp-patches9"/"ui"/"inputs"/"4k.png").convert("RGB")).astype(np.float64)/255
values = pixels*.973+.003
values.setflags(write=False)
del pixels
for dtype, maximum in ((np.dtype("uint8"), 255.), (np.dtype("uint16"), 65535.)):
    expected = signature(_quantize_strict(values, dtype, maximum))
    for label, function in (("A1", _quantize_strict), ("B1", reuse), ("B2", reuse), ("A2", _quantize_strict)):
        raw = []
        for index in range(4):
            gc.collect()
            started = time.perf_counter_ns()
            result = function(values, dtype, maximum)
            elapsed = (time.perf_counter_ns()-started)/1e6
            assert signature(result) == expected
            if index:
                raw.append(elapsed)
            del result
        gc.collect(); tracemalloc.start()
        result = function(values, dtype, maximum)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop(); del result
        row = {"dtype": str(dtype), "label": label, "median_ms": statistics.median(raw), "raw_ms": raw, "peak_traced_bytes": peak, "exact": True}
        data["runs"].append(row)
        (OUT/"results.json").write_text(json.dumps(data, indent=2), encoding="utf8")
        print(row, flush=True)
