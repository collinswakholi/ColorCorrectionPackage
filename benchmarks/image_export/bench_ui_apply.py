"""Retime real UI SDK apply-only against previously frozen native calibrations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

for name in ("RAYON_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"


def measured(function):
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns()-started)/1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--prior-benchmark", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["512", "1080p", "4k"])
    parser.add_argument("--backends", nargs="+", default=["cpu", "auto"])
    parser.add_argument("--kind", choices=["full", "cc_only"], default="full")
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output/"tmp"
    temporary.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.ui/"api"))
    import numpy as np
    from colorcorrect import Client, ApplyOptions
    prior = json.loads(args.prior_benchmark.read_text())
    data = {"threads": 8, "sdk": str(args.ui/"api"), "kind": args.kind,
        "worker": str(args.native/"ccp-engine.exe"), "worker_sha256": hashlib.sha256((args.native/"ccp-engine.exe").read_bytes()).hexdigest(),
        "scope": "Real UI persistent worker SDK file-to-PNG8 fast_mode=True combined65 renderer; no refit, no package strict publication guarantees.", "runs": []}
    assert data["worker_sha256"] == "123cb49377f0729e1c542dfbcdc096fff2a75f1cb6fea61f020cf7db7377a4fd"
    for backend in args.backends:
        old_row = next(row for row in prior["runs"] if row["backend"] == backend and row["kind"] == args.kind)
        model_path = Path(old_row["model"])
        with Client(root=args.native, backend=backend, timeout=120) as client:
            ready = sorted(client.capabilities)
            model = client.load_calibration(model_path)
            for size in args.sizes:
                path = args.inputs/(size+".png")
                cold, first_ms = measured(lambda: model.apply(path, output_dir=args.output,
                    options=ApplyOptions(output_format="png", fast_mode=True)))
                times = []
                for repetition in range(args.repetitions):
                    applied, elapsed = measured(lambda: model.apply(path, output_dir=args.output,
                        options=ApplyOptions(output_format="png", fast_mode=True)))
                    times.append(elapsed)
                data["runs"].append({"backend": backend, "size": size, "first_ms": first_ms,
                    "timing": {"n": len(times), "median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)), "raw_ms": times},
                    "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reply": applied.raw_reply, "output_sha256": hashlib.sha256(applied.outputs[0].read_bytes()).hexdigest(), "capabilities": ready})
                (args.output/"results.json").write_text(json.dumps(data, indent=2), encoding="utf8")
                print(backend, size, data["runs"][-1]["timing"], flush=True)


if __name__ == "__main__":
    main()
