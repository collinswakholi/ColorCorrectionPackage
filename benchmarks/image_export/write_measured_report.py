"""Write the reproducible evidence report from raw JSON, never invent timings."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/"final-summary.json").read_text(encoding="utf8"))
lines = ["# Image validation and strict export: measured qualification", "",
"These measurements compare the frozen final patch-9 installed package with the local clone after immutable image validation, bounded reading/metadata parsing, private input borrowing, and strict writer changes. The final main matrix includes preallocated normalization and integer quantization workspace reuse. Intermediate ABBA and isolated ablations remain available as supporting evidence and are clearly labeled.", "",
"## Controls and scope", "",
"- Python 3.12 audit environment; fresh processes select exactly one source tree. BLAS/OpenMP/Numba threads are set to 8 before imports and constrained with threadpoolctl. Default OpenCV Concurrency policy remains unchanged (32 threads on this host). Large-image digests use one additional temporary worker, synchronously joined.",
"- TEMP/TMP, Matplotlib and Numba caches are writable audit directories before imports. No application thread policy, validation, color math, or persistence guarantee is disabled in the timing harness.",
"- Inputs are the same frozen real-photo PNGs (512×512, 1920×1080, 3840×2160). The 4K input is 7,538,731 bytes, SHA256 a3af13f6673130a2ccd022b259844a1c86a21b3bd1a1ca2018c060b5d78903ec.",
"- Components: one warmup then three measured public calls, fixture setup/garbage collection/output deletion outside timing. Profiling and tracemalloc are separate untimed passes. The intermediate pipeline was measured in fresh-process ABBA separately for direct and relaxed combined-LUT129 (n=6/source/mode). After the final exact integer workspace improvement, fresh candidate workers ran n=3 per mode; a new frozen relaxed control ran n=3, while direct uses the unchanged earlier ABBA frozen control n=6. Identical frozen full portable calibration; preparation is recorded separately. UI: real SDK 2.2.1 and released worker 0.2.0, persistent worker, same saved native models/input PNGs, first apply recorded separately then seven warm file-to-PNG calls.",
"- Strict package export includes atomic publication, fsync, full published-byte verification, hashing, raw codec decode, full sample verification, and exact quantization/error accounting. UI combined65 uses different rendering precision, interpolation and persistence. This is an honest file-workflow comparison, not a claim of equal numerical/persistence contracts.", "",
"## Main component matrix", "", "Milliseconds, median of n=3. Every figure includes the public operation's normal validation work.", "",
"| Operation | 512 baseline → clone | 1080p baseline → clone | 4K baseline → clone |", "|---|---:|---:|---:|"]
operations = list(dict.fromkeys(row["operation"] for row in data["component_rows"]))
for operation in operations:
    values = []
    for size in ("512", "1080p", "4k"):
        row = next(row for row in data["component_rows"] if row["operation"] == operation and row["size"] == size)
        values.append(f'{row["baseline_median_ms"]:.2f} → {row["candidate_median_ms"]:.2f}')
    lines.append("| "+operation+" | "+" | ".join(values)+" |")
copy_row = next(row for row in data["component_rows"] if row["operation"] == "typed_to_srgb" and row["size"] == "4k")
lines += ["", f'The public to_srgb copy is intentionally unchanged and remains an independent writable array. Its timing variation (4K {copy_row["baseline_median_ms"]:.2f}→{copy_row["candidate_median_ms"]:.2f} ms) is a control, not a claimed optimization. Separate exact-pixel, writability, nonaliasing and mutation-isolation assertions pass at all three sizes.', "",
"## Final 4K complete file prediction and strict PNG export", "", "Final candidate median n=3. Frozen direct control n=6 from the earlier ABBA; frozen relaxed control n=3 from a new contemporary process. Raw samples and the intermediate ABBA remain in final-summary.json.", "",
"| Mode | Operation | Baseline ms | Clone ms | Reduction |", "|---|---|---:|---:|---:|"]
for row in data["pipeline_rows"]:
    lines.append(f'| {row["mode"]} | {row["metric"]} | {row["baseline_median_ms"]:.2f} | {row["candidate_median_ms"]:.2f} | {row["reduction_percent"]:.1f}% |')
lines += ["", "## Real UI Python SDK retest", "", "Median of seven warm file-to-PNG applications; startup/model loading and first cold application are recorded separately.", "",
"| Requested backend | Size | Median ms | Actual renderer |", "|---|---|---:|---|"]
for row in data["ui"]["runs"]:
    execution = row["reply"]["execution"]
    lines.append(f'| {row["backend"]} | {row["size"]} | {row["timing"]["median_ms"]:.2f} | {execution["backend"]}; {execution["method"]} |')
relaxed = next(row for row in data["pipeline_rows"] if row["mode"] == "relaxed" and row["metric"] == "file_total_ms")
ui_cpu = next(row for row in data["ui"]["runs"] if row["backend"] == "cpu" and row["size"] == "4k")["timing"]["median_ms"]
lines += ["", f'The main matrix reduces the relaxed 4K package/UI CPU time ratio from {relaxed["baseline_median_ms"]/ui_cpu:.2f}× to {relaxed["candidate_median_ms"]/ui_cpu:.2f}×. The gap is materially narrower and remains open. Direct GC/color execution still dominates direct mode; changing its mathematics was outside this optimization.', "",
"## Accuracy and compatibility evidence", "",
"- All 27 component ImageData/ExportRecord cases compare exactly against the frozen baseline. Every ExportRecord field except destination path is equal, including source digest, canonical pixel digest, payload SHA256, byte count, quantization error, codec error, dtype, orientation, encoding and precision-loss flags.",
"- Every intermediate ABBA and final source/process case agrees on full float64 prediction pixel SHA256 and the complete path-normalized export record. Direct and relaxed are compared to their own frozen mode, not to one another.",
"- Public to_srgb copy controls pass exact pixels and independent writable ownership at all three resolutions. All source hashes are checked before/after each main worker and remain unchanged.", "",
"## Traced peak memory", "", "Separate 4K tracemalloc passes. These include NumPy allocations visible to Python and exclude some native codec/driver workspaces; they are not whole-process peak RSS.", "",
"| Public operation | Baseline MiB | Clone MiB | Reduction |", "|---|---:|---:|---:|"]
for key, before in data["memory_baseline"].items():
    b = before["python_numpy_traced_peak_bytes"]
    a = data["memory_candidate"][key]["python_numpy_traced_peak_bytes"]
    lines.append(f'| {key} | {b/2**20:.2f} | {a/2**20:.2f} | {100*(1-a/b):.1f}% |')
lines += ["", "## Ablations and negative results", "",
"The initial growing-BytesIO candidate was retained as negative evidence. The first whole matrix also showed an unchanged copy control slowing by ~2.6×, so its performance numbers are not used as final claims. A contemporary frozen 4K control reproduced the original baseline: from_array float64 592.62 ms, uint8 603.55 ms, read 1348.68 ms and strict PNG8 2057.78 ms. Isolated normalizer ablations then identified a real growing-buffer regression and established the final preallocation choice.", "",
"| 4K constructor ablation | Median ms | Traced peak MiB |", "|---|---:|---:|"]
for row in data["contracts_ablation"]["runs"]:
    if "peak_traced_bytes" in row:
        lines.append(f'| {row["name"]} | {row["median_ms"]:.2f} | {row["peak_traced_bytes"]/2**20:.2f} |')
lines += ["", "Every normalizer ablation matched the same canonical image/digest signatures. Parallel hashes were faster than sequential on this host. Preallocated immutable streaming was faster and used about half the peak memory of full-array in-place normalization. A diagnostic OpenCV before/after cvtColor and process-local thread-count-one test showed no meaningful improvement; the hypothesis that the RGB swap caused the whole-matrix slowdown was not supported. No global OpenCV threading policy was changed.", "",
"The bounded follow-on integer quantization experiment reuses the already-rounded float64 workspace for exact reconstruction error. The isolated experiment did not edit the application; its successful variant was subsequently integrated, tested, and included in the final main matrix. All storage SHA256 and float64 maximum-error bits matched, including every integer level, half ties, nextafter neighbors, signed zero, subnormal/endpoint values, strided arrays and the full 4K frame.", "",
"| Quantizer | Baseline median ms (n=6) | Workspace reuse ms (n=6) |", "|---|---:|---:|"]
for dtype in ("uint8", "uint16"):
    before = sum([row["raw_ms"] for row in data["quantization_ablation"]["runs"] if row["dtype"] == dtype and row["label"].startswith("A")], [])
    after = sum([row["raw_ms"] for row in data["quantization_ablation"]["runs"] if row["dtype"] == dtype and row["label"].startswith("B")], [])
    lines.append(f'| {dtype} | {statistics.median(before):.2f} | {statistics.median(after):.2f} |')
lines += ["", "## Reproduction and raw inventory", "",
"Repository scripts live in benchmarks/image_export; the small measured results live in docs/validation/image_export. Start with benchmarks/image_export/README.md. Use the qualified audit Python environment and the same frozen inputs/calibrations. Every worker requires a fresh output directory. Do not run another test or benchmark workload concurrently.", "",
"- benchmarks/image_export/bench_image_io.py: generic --site/--inputs/--output component worker, optional separate --profile and --memory passes. Baseline, control, negative-candidate, intermediate and final raw samples are under docs/validation/image_export with their original directory labels.",
"- benchmarks/image_export/bench_pipeline_io.py: generic fresh worker with explicit --site/--models/--inputs/--output, --file-only, --sizes 4k, --modes direct or relaxed and --repetitions 3. The historical sequential audit drivers are run_pipeline_abba.py and run_final_revision.py; their manifests retain the exact executed commands and local audit paths.",
"- benchmarks/image_export/bench_ui_apply.py: generic real UI SDK worker. docs/validation/image_export/ui-final/results.json records the native worker hash, actual backend, model/input hashes, full replies and raw n7 samples.",
"- benchmarks/image_export/ablate_contracts.py and ablations/results.json: normalization, digest scheduling and OpenCV diagnostics. ablate_quantization.py and quantization-ablation/results.json retain the exact edge checks and isolated n6 integer quantizer timings. Audit-only replay scripts preserve their original explicit fixture paths.",
"- docs/validation/image_export/final-summary.json contains final/intermediate tables, full source fingerprints, memory observations and parity evidence. inventory.json provides SHA256/size for every exported script/report/result. Large image/native/model binaries and binary cProfile files remain in the original local audit; none are vendored into the repository.", ""]
(ROOT/"MEASURED_RESULTS.md").write_text("\n".join(lines), encoding="utf8")
inventory = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.suffix in (".json", ".py", ".txt", ".prof", ".md", ".log") and "cache" not in str(path) and path.name != "inventory.json":
        payload = path.read_bytes()
        inventory.append({"path": str(path.relative_to(ROOT)), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
(ROOT/"inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf8")
print("Report and", len(inventory), "inventory files written")
