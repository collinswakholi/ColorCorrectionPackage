# Image validation and strict export: measured qualification

These measurements compare the frozen final patch-9 installed package with the local clone after immutable image validation, bounded reading/metadata parsing, private input borrowing, and strict writer changes. The final main matrix includes preallocated normalization and integer quantization workspace reuse. Intermediate ABBA and isolated ablations remain available as supporting evidence and are clearly labeled.

## Controls and scope

- Python 3.12 audit environment; fresh processes select exactly one source tree. BLAS/OpenMP/Numba threads are set to 8 before imports and constrained with threadpoolctl. Default OpenCV Concurrency policy remains unchanged (32 threads on this host). Large-image digests use one additional temporary worker, synchronously joined.
- TEMP/TMP, Matplotlib and Numba caches are writable audit directories before imports. No application thread policy, validation, color math, or persistence guarantee is disabled in the timing harness.
- Inputs are the same frozen real-photo PNGs (512×512, 1920×1080, 3840×2160). The 4K input is 7,538,731 bytes, SHA256 a3af13f6673130a2ccd022b259844a1c86a21b3bd1a1ca2018c060b5d78903ec.
- Components: one warmup then three measured public calls, fixture setup/garbage collection/output deletion outside timing. Profiling and tracemalloc are separate untimed passes. The intermediate pipeline was measured in fresh-process ABBA separately for direct and relaxed combined-LUT129 (n=6/source/mode). After the final exact integer workspace improvement, fresh candidate workers ran n=3 per mode; a new frozen relaxed control ran n=3, while direct uses the unchanged earlier ABBA frozen control n=6. Identical frozen full portable calibration; preparation is recorded separately. UI: real SDK 2.2.1 and released worker 0.2.0, persistent worker, same saved native models/input PNGs, first apply recorded separately then seven warm file-to-PNG calls.
- Strict package export includes atomic publication, fsync, full encoded-byte read-back verification, hashing, raw codec decode, full sample verification, and exact quantization/error accounting. UI combined65 uses different rendering precision, interpolation and persistence. This is an honest file-workflow comparison, not a claim of equal numerical/persistence contracts.

## Main component matrix

Milliseconds, median of n=3. Every figure includes the public operation's normal validation work.

| Operation | 512 baseline → clone | 1080p baseline → clone | 4K baseline → clone |
|---|---:|---:|---:|
| from_array_float64 | 17.65 → 11.26 | 145.21 → 56.33 | 598.57 → 242.50 |
| from_array_uint8 | 17.84 → 14.49 | 148.44 → 70.84 | 604.28 → 299.84 |
| constructor_float64 | 11.47 → 10.73 | 92.23 → 50.78 | 367.17 → 217.44 |
| typed_to_srgb | 1.40 → 1.42 | 10.36 → 10.54 | 41.27 → 57.53 |
| read_png | 382.38 → 21.37 | 564.90 → 122.28 | 1385.68 → 485.10 |
| strict_png8_float64 | 83.53 → 42.25 | 513.38 → 235.29 | 2110.09 → 853.37 |
| strict_png8_typed_float64 | 66.15 → 28.98 | 358.88 → 162.30 | 1399.34 → 582.53 |
| strict_png8_typed_uint8 | 67.82 → 29.83 | 356.67 → 166.53 | 1382.21 → 582.73 |
| strict_png16_uint16 | 104.13 → 54.90 | 620.97 → 332.57 | 2387.03 → 1318.45 |
| strict_tiff_float64 | 77.21 → 50.74 | 456.85 → 203.99 | 2042.55 → 827.55 |

The public to_srgb copy is intentionally unchanged and remains an independent writable array. Its timing variation (4K 41.27→57.53 ms) is a control, not a claimed optimization. Separate exact-pixel, writability, nonaliasing and mutation-isolation assertions pass at all three sizes.

## Final 4K complete file prediction and strict PNG export

Final candidate median n=3. Frozen direct control n=6 from the earlier ABBA; frozen relaxed control n=3 from a new contemporary process. Raw samples and the intermediate ABBA remain in final-summary.json.

| Mode | Operation | Baseline ms | Clone ms | Reduction |
|---|---|---:|---:|---:|
| direct | file_predict_ms | 7524.49 | 7305.01 | 2.9% |
| direct | strict_export_ms | 1983.82 | 857.35 | 56.8% |
| direct | file_total_ms | 9519.03 | 8140.28 | 14.5% |
| relaxed | file_predict_ms | 1634.18 | 699.93 | 57.2% |
| relaxed | strict_export_ms | 2055.25 | 842.41 | 59.0% |
| relaxed | file_total_ms | 3679.20 | 1551.72 | 57.8% |

## Real UI Python SDK retest

Median of seven warm file-to-PNG applications; startup/model loading and first cold application are recorded separately.

| Requested backend | Size | Median ms | Actual renderer |
|---|---|---:|---|
| cpu | 512 | 16.72 | cpu; combined-lut65 |
| cpu | 1080p | 76.66 | cpu; combined-lut65 |
| cpu | 4k | 285.41 | cpu; combined-lut65 |
| auto | 512 | 18.76 | wgpu:dx12:NVIDIA GeForce RTX 4070 Ti SUPER; combined-lut65 |
| auto | 1080p | 76.88 | wgpu:dx12:NVIDIA GeForce RTX 4070 Ti SUPER; combined-lut65 |
| auto | 4k | 280.26 | wgpu:dx12:NVIDIA GeForce RTX 4070 Ti SUPER; combined-lut65 |

The main matrix reduces the relaxed 4K package/UI CPU time ratio from 12.89× to 5.44×. The gap is materially narrower and remains open. Direct GC/color execution still dominates direct mode; changing its mathematics was outside this optimization.

## Accuracy and compatibility evidence

- All 27 component ImageData/ExportRecord cases compare exactly against the frozen baseline. Every ExportRecord field except destination path is equal, including source digest, canonical pixel digest, payload SHA256, byte count, quantization error, codec error, dtype, orientation, encoding and precision-loss flags.
- Every intermediate ABBA and final source/process case agrees on full float64 prediction pixel SHA256 and the complete path-normalized export record. Direct and relaxed are compared to their own frozen mode, not to one another.
- Public to_srgb copy controls pass exact pixels and independent writable ownership at all three resolutions. All source hashes are checked before/after each main worker and remain unchanged.

## Traced peak memory

Separate 4K tracemalloc passes. These include NumPy allocations visible to Python and exclude some native codec/driver workspaces; they are not whole-process peak RSS.

| Public operation | Baseline MiB | Clone MiB | Reduction |
|---|---:|---:|---:|
| from_array_float64 | 403.42 | 189.86 | 52.9% |
| read_png | 1024.01 | 222.27 | 78.3% |
| strict_png8_float64 | 1044.23 | 308.67 | 70.4% |
| strict_png8_typed_uint8 | 854.39 | 118.82 | 86.1% |

## Ablations and negative results

The initial growing-BytesIO candidate was retained as negative evidence. The first whole matrix also showed an unchanged copy control slowing by ~2.6×, so its performance numbers are not used as final claims. A contemporary frozen 4K control reproduced the original baseline: from_array float64 592.62 ms, uint8 603.55 ms, read 1348.68 ms and strict PNG8 2057.78 ms. Isolated normalizer ablations then identified a real growing-buffer regression and established the final preallocation choice.

| 4K constructor ablation | Median ms | Traced peak MiB |
|---|---:|---:|
| parallel/float64_unit | 233.18 | 189.86 |
| sequential/float64_unit | 354.06 | 189.85 |
| growing/uint8 | 665.93 | 202.88 |
| growing/float64_nonunit | 668.04 | 202.38 |
| preallocated/uint8 | 280.87 | 191.35 |
| preallocated/float64_nonunit | 267.98 | 190.85 |
| full_inplace/uint8 | 294.39 | 379.70 |
| full_inplace/float64_nonunit | 305.59 | 379.70 |
| native_snapshot/uint8 | 289.81 | 379.70 |
| native_snapshot/float64_nonunit | 308.64 | 379.70 |

Every normalizer ablation matched the same canonical image/digest signatures. Parallel hashes were faster than sequential on this host. Preallocated immutable streaming was faster and used about half the peak memory of full-array in-place normalization. A diagnostic OpenCV before/after cvtColor and process-local thread-count-one test showed no meaningful improvement; the hypothesis that the RGB swap caused the whole-matrix slowdown was not supported. No global OpenCV threading policy was changed.

The bounded follow-on integer quantization experiment reuses the already-rounded float64 workspace for exact reconstruction error. The isolated experiment did not edit the application; its successful variant was subsequently integrated, tested, and included in the final main matrix. All storage SHA256 and float64 maximum-error bits matched, including every integer level, half ties, nextafter neighbors, signed zero, subnormal/endpoint values, strided arrays and the full 4K frame.

| Quantizer | Baseline median ms (n=6) | Workspace reuse ms (n=6) |
|---|---:|---:|
| uint8 | 327.01 | 125.06 |
| uint16 | 333.87 | 135.22 |

## Reproduction and raw inventory

Repository scripts live in benchmarks/image_export; the small measured results live in docs/validation/image_export. Start with benchmarks/image_export/README.md. Use the qualified audit Python environment and the same frozen inputs/calibrations. Every worker requires a fresh output directory. Do not run another test or benchmark workload concurrently.

- benchmarks/image_export/bench_image_io.py: generic --site/--inputs/--output component worker, optional separate --profile and --memory passes. Baseline, control, negative-candidate, intermediate and final raw samples are under docs/validation/image_export with their original directory labels.
- benchmarks/image_export/bench_pipeline_io.py: generic fresh worker with explicit --site/--models/--inputs/--output, --file-only, --sizes 4k, --modes direct or relaxed and --repetitions 3. The historical sequential audit drivers are run_pipeline_abba.py and run_final_revision.py; their manifests retain the exact executed commands and local audit paths.
- benchmarks/image_export/bench_ui_apply.py: generic real UI SDK worker. docs/validation/image_export/ui-final/results.json records the native worker hash, actual backend, model/input hashes, full replies and raw n7 samples.
- benchmarks/image_export/ablate_contracts.py and ablations/results.json: normalization, digest scheduling and OpenCV diagnostics. ablate_quantization.py and quantization-ablation/results.json retain the exact edge checks and isolated n6 integer quantizer timings. Audit-only replay scripts preserve their original explicit fixture paths.
- docs/validation/image_export/final-summary.json contains final/intermediate tables, full source fingerprints, memory observations and parity evidence. inventory.json provides SHA256/size for every exported script/report/result. Large image/native/model binaries and binary cProfile files remain in the original local audit; none are vendored into the repository.
