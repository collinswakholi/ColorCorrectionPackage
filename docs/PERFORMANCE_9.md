# Patch 9: measured execution and real UI Python API comparison

## Result

The final source preserves direct execution under the strict default error budget. An explicitly selected, less strict profile (`max_rgb_error=0.006`, `max_delta_e00=1.0`, `grid_sizes=(129,)`) passed the independent audits and the tested real image. On the same 4K photograph and frozen calibration, seven alternating public API calls measured **5,851 ms direct versus 822 ms with the qualified combined LUT**, a **7.12× median improvement**. These are in-memory float64 calls, not file-to-file timings.

Final file-to-strict-PNG processing measured **5,626 ms median over three calls**. The UI native Python API remains substantially faster for its file-to-PNG workload: **281 ms CPU rendering / 348 ms GPU rendering** for 4K GC/WB/CC. The fitting algorithms, precision, and persistence guarantees differ, so the measurements do not establish interchangeable outputs or physical calibration superiority.

The investigation found a real qualification gap before finalization: a one-code sampled RGB budget passed an initial audit but failed near-black image pixels. The final source adds a separate 572,733-point stress audit and now rejects that profile. The failed prototype measurements are retained as negative evidence.

Unit-suite and clean-distribution results are reported separately by the main patch-9 qualification report; this document covers the actual API experiments.

## Measurement conditions and scope

- Windows; registry CPU identity: Intel Core i9-14900K. Numerical application uses 8 Numba/OpenMP threads and 8 NumPy/SciPy OpenBLAS threads. Isolated package detector probes explicitly use 8 ORT CPU threads; whole P78 fitting used its detector default of 4 threads.
- Installed audit interpreter: Python 3.12.10; NumPy 2.4.6; SciPy 1.18.1; scikit-learn 1.9.1; Torch 2.13.0+cpu; ONNX Runtime 1.31.1; onnxruntime-ep-webgpu 0.4.0. This existing audit environment is not presented as a clean installation test.
- Host physical RAM: 34.03 GB. The diagnostic observed 6.3–7.2 GB available, or 78–81% memory load. No other active Python workload appeared in the process inventory. These are workstation measurements, with runtime variability; no thermal cause was measured.
- Numerical timing windows were sequential. The first benchmark process launched at approximately 23:04:09 UTC on 2026-10-28. The window was released at 23:51:07 UTC after final-source and actual-GPU checks. Source hashes remained unchanged during both final-source measurements.
- Application images derive from the public `Sample_3572.JPG`, resized once to RGB PNG at 512×512, 1920×1080, and 3840×2160. Both applications receive the same files. Calibration uses a common `ColorCard_Big.JPG` PNG resized to a maximum dimension of 1000 pixels. Fixture file SHA256 values are recorded in UI benchmark metadata.
- Controlled P78 models use explicit color semantics, modern stages, reproducible degree-1 affine CC, `use_lut=False`, no FFC, no saturation extrapolation, and no cross-validation. Both products receive the same nominal canonical Lab/D50 references. This is a controlled direct P78 baseline, not a latency claim for every historical/default configuration.
- UI options use degree-1 `linear`, one sample per chart cell, no FFC/saturation/cross-validation, and `fast_mode=True` for the main application matrix. The fast renderer combines GC/WB/CC into a 65³ LUT. Native UI detection is GPU-only even when its correction backend is `cpu`.
- All P9 application comparisons load the exact P78 `.ccpb` bundles, avoiding refitting variation. The final paired comparison alternates mode order over seven repetitions. p95 is the NumPy percentile over the stated sample count, not a confidence interval.

## Real UI implementation tested

The SDK is `CCP_New_UI/api`, version **2.2.1**. The real native worker is the extracted `candidate-win-x64-v3` package, SHA256:

`123cb49377f0729e1c542dfbcdc096fff2a75f1cb6fea61f020cf7db7377a4fd`

That worker matches the public 0.2.0 release manifest, whose source revision is `ee2522eaddb615dcbddc2ea4505e9e435bfc08ae`. The inspected UI checkout is `16c6f42176cbd392c167ecabea4bb2519984fda8`. The native worker is **not claimed to have been rebuilt from that HEAD**. The inspected revisions had no Rust `.rs`, model, or reference changes; dependency manifests/lockfiles differed. The current SDK was used directly through its persistent worker protocol, not a fake worker.

The primary 1280 detector weights are byte-identical in package and UI resources: 39,354,948 bytes, SHA256 `1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240`.

The UI rereads and decodes each input on repeated apply calls. `serve.rs:461` calls `apply_saved_to_image_ex_with_progress` per item; `run_pipeline.rs:2320` calls `load_image_with_icc`; `sample.rs:126` reads the bytes and creates a fresh decoder. Warm state includes model/LUT/GPU resources and OS file cache, not a decoded-image cache.

## Application timings

UI numbers are file decode + native correction + PNG8 encoding + SDK/RPC. Package file numbers additionally include the public strict writer's atomic publication, fsync, encoded-byte and pixel read-back, and hashes. In-memory numbers include the package's image contract preparation from a float64 ndarray. The compatibility `write_image` writer was not timed.

Median / p95, milliseconds; n=7 except the explicitly marked final strict export.

| Workload | Resolution | P78 direct memory | P78 strict PNG | UI CPU combined65 PNG | UI GPU combined65 PNG |
|---|---:|---:|---:|---:|---:|
| CC only | 512² | 32.97 / 34.34 | 420.27 / 433.63 | 16.11 / 16.48 | 17.31 / 18.18 |
| CC only | 1080p | 257.39 / 259.26 | 1176.63 / 1193.41 | 79.99 / 81.40 | 82.46 / 84.45 |
| CC only | 4K | 1071.18 / 1127.09 | 3941.65 / 4496.46 | 283.88 / 299.92 | 315.02 / 356.16 |
| GC/WB/CC | 512² | 199.38 / 232.74 | 576.85 / 588.31 | 15.65 / 15.96 | 20.86 / 21.11 |
| GC/WB/CC | 1080p | 1593.67 / 1613.61 | 2525.26 / 2545.28 | 77.00 / 77.77 | 94.66 / 95.58 |
| GC/WB/CC | 4K | 6218.30 / 6284.49 | 8900.90 / 9027.51 | 281.27 / 286.63 | 348.15 / 358.91 |

The raw JSON is authoritative for all digits. GPU rendering was slower than CPU rendering in these UI file workloads; no universal GPU speedup is claimed.

Final audit-v2 source, same-process 4K GC/WB/CC:

| Case | n | Median ms | p95 ms |
|---|---:|---:|---:|
| Direct float64 public API | 7 | 5851.34 | 7673.73 |
| Explicit relaxed 129³ LUT, float64 public API | 7 | 822.31 | 998.80 |
| Relaxed file → strict PNG8 | 3 | 5625.76 | 5672.39 |
| Prediction portion of final file call | 3 | 2482.93 | 2516.99 |
| Strict export portion of final file call | 3 | 3095.12 | 3173.74 |

Fresh-process preparation for the final relaxed PNG experiment was 6110.00 ms; immediate same-policy cache-hit preparations measured 52.78 / 53.58 ms median/p95 (n=3, all hits). In the paired experiment, where earlier checks had already warmed CPU/JIT resources, relaxed preparation was 2456.67 ms, with subsequent cache-hit preparations 41.10 / 41.40 ms. These are distinct startup conditions.

A short same-process diagnostic controlled the earlier large separately-run spread: direct versus default-auto fallback measured 1115 / 1084 ms at CC-only 4K and 1756 / 1793 ms at full-stage 1080p (n=3 per mode), with exactly matching pixels. Reusing an already validated immutable `ImageData` reduced the CC-only case to 431 / 423 ms, after separately paying source preparation. This amortized same-pixels scenario is not fresh-frame latency; the short GC case did not show a reliable typed-input gain.

## Qualification failure found and repaired

The original sampled/audit sets contained 4,941 and 4,967 points. For the full-stage 129³ LUT, they measured maximum RGB errors 0.002905 / 0.003157 and maximum ΔE00 0.175 / 0.260. These passed a one-code (`1/255`) RGB budget with ΔE00≤0.5.

Dense real images disproved a global interpretation: maxima reached **0.005348 RGB / 0.848027 ΔE00**. The 67³ cell-center grid alone also missed the problem (0.002625 / 0.333056). Near-black examples include input RGB `[3,1,0]/255`, `[2,1,2]/255`, and `[1,1,5]/255`. The 67-grid's first center lies near a 129-grid cell boundary, missing its midpoint.

Independent endpoint tests measured:

| Full-stage fixed129 stress | Max RGB | Max ΔE00 |
|---|---:|---:|
| Complete near-black byte cube `{0..16}/255` | 0.005376 | 0.868785 |
| First four LUT cells, 33³ samples | 0.005389 | 0.896735 |
| Independent 262,144 byte-RGB random points | 0.003539 | 0.573885 |

Final audit-v2 adds an independent **572,733-point** stress set: 67³ centers (300,763), complete near-black and near-white byte cubes (9,826), and independently seeded byte-RGB random points (262,144). The final one-code profile now falls back to direct with an explicit stress-audit rejection; it does not retune against the held-out audit.

| Final full-stage profile | Outcome |
|---|---|
| Default RGB≤0.002, ΔE00≤0.5 | Direct fallback; no default candidate meets qualification |
| RGB≤1/255, ΔE00≤0.5, 129 only | Direct fallback; selected table fails independent stress |
| RGB≤0.006, ΔE00≤1.0, 129 only | 129³ LUT accepted on tested qualification/audit/stress sets |

The final relaxed stress maximum is **0.005376 RGB / 0.868785 ΔE00**. The full 4K image maximum is **0.005348 / 0.848027**; mean errors are **0.00003345 RGB / 0.004925 ΔE00**, with p95 **0.00010579 / 0.017775**. These are measured output differences from direct execution, not physical color accuracy. The audits remain empirical, not a proof over every possible RGB value.

The pre-hardening one-code timing matrix is retained as a **failed prototype**, not a supported accuracy profile. Its numeric 129³ table can be independently examined under the final code using the deliberately loose fixed-table endpoint probe; that probe does not promote the table.

## Actual GPU detection and honest fallback

Initial sandbox runs could not reopen ORT's GPU profiling file under the AppContainer temporary path. The detector honestly rejected unverifiable GPU execution and fell back to CPU. Their auto medians (331.87 ms P78 / 368.81 ms initial P9) are **CPU-fallback measurements**, not GPU measurements. The raw failed attempts are retained.

A fresh P9 process with `TEMP` and `TMP` pointing to the writable audit directory succeeded without escalation. Auto tried unavailable CUDA/DirectML, then selected **WebGpuExecutionProvider / D3D12**. Node profiling verified **210 GPU nodes**, `execution_verified=True`, and `cpu_fallback=False`.

| Actual same-process detector case | First file call ms | Warm file median / p95 ms | Warm memory median / p95 ms |
|---|---:|---:|---:|
| Explicit CPU | 1243.23 | 638.50 / 751.56 | 596.80 / 619.92 |
| Actual verified GPU | 3578.95 | 226.66 / 317.94 | 201.87 / 345.55 |

Each warm result has seven calls and a fresh detection context, so geometry caching cannot replace inference. File calls include PNG decode; memory calls do not. The public package helper also samples one RGB per chart cell. UI uncached detection measured 199.72 / 209.41 ms under CPU-render policy and 208.64 / 210.58 ms under auto-render policy, both using actual DirectML inference.

UI stderr confirmed **DirectML DXGI adapter index 1**, corresponding to the RTX 4070 Ti SUPER. The static `devices` reply's planned index 0 is not the actual DirectML assignment. Package GPU node evidence identifies WebGPU/D3D12; the raw provider record does not identify a physical adapter name. Radeon and Apple hardware were not available for measurement here.

## Numerical comparison without assuming UI ground truth

The shared-cell experiment supplies exactly the UI report's 24 measured RGB means and the same canonical Lab/D50 references. UI degree-1 linear fitting uses **linear RGB, an intercept, and ridge 1e-5**; package reproducible fitting uses **encoded sRGB, a bias feature, and unregularized least squares**. Both are affine models in different domains. UI D50 x=0.34567 and package D50 x=0.3457 are separately recorded. Clipped display-reference RGB is not substituted for the unclipped fit target.

Independently reconstructing the actual saved UI affine matrix from those samples and native color constants matches its coefficients to **1.68e-15 maximum error**. Applying the actual UI matrix and package model to the shared samples produces normalized RGB differences: mean **0.027664**, p95 **0.068164**, maximum **0.104240**. Mean training-chart ΔE00 under common evaluation is **3.61287 package / 3.93372 UI**. These are training residuals under distinct objectives and do not establish better generalization or physical calibration accuracy.

UI fast-mode checks load the same frozen UI calibration for both renderers. `fast_mode=False` still uses its standard CC LUT33; it is not an analytic float64 reference. Output is PNG8, so differences include quantization:

| UI saved model/input | Maximum quantized RGB difference | Maximum output-pair ΔE00 |
|---|---:|---:|
| CC only / common chart | 1/255 | 1.09445 |
| CC only / seeded RGB probe | 10/255 | 1.66353 |
| GC/WB/CC / common chart | 9/255 | 3.50268 |
| GC/WB/CC / seeded RGB probe | 12/255 | 3.50268 |

CPU/GPU results were essentially identical for this comparison. These PNG8 differences are not compared directly to P9's float64 qualification budgets.

## Functional and persistence observations

- The real UI SDK detected 1 / 2 / 1 / 1 chart faces in the four full-resolution public chart photographs. Blank input raised `ProcessingError: no lattice`.
- All four calibrations saved and loaded successfully, with bit-identical reapplied PNG pixels. Both 33³ and 65³ cube exports succeeded for all four.
- Mixed 8/16-bit input preserved the successful 8-bit item and reported the 16-bit TIFF as unsupported. This is a UI limitation, not a package precision requirement.
- P78 warm bundle save medians were 179.84 ms CC-only / 229.24 ms full; warm load medians 140.90 / 174.35 ms. These include the package's validated portable format.
- UI warm `load_calibration` handles measured about 3.1 ms; native parsing/LUT/GPU initialization occurs on apply. Fresh-worker first apply at 512² measured 28.10 / 36.69 ms CPU and 720.86 / 681.63 ms GPU for CC-only/full. Do not describe the Python handle time as complete native load latency.
- The cold-resume raw `worker_ready_ms` field measures a capability read after context entry had already started the worker; it excludes process startup. The separate main startup measurement captured actual worker ready latency: 984.31 ms first CPU-policy worker / 50.70 ms later auto-policy worker, single samples.
- UI warm fit/save/report medians were 40.82 / 69.40 ms CPU and 45.22 / 77.03 ms GPU for CC-only/full. `write_output=False` still renders the calibration image. These calls reuse a detection handshake. P78 whole-run medians 969.73 / 1518.30 ms include detection, contract/observation work, fitting, and stage images, so they are not isolated fitting comparisons.
- UI 65³ cube export medians were 182.12 / 250.35 ms CPU for CC-only/full. The old prototype's 129³ package exports measured 8724 / 10346 ms, producing 114.9 / 118.0 MB text cubes. Final source format/export code is unchanged; no final-source export throughput claim is substituted for those old measurements.

## Reproduction and evidence

[The raw inventory](validation/patch9/ui/raw_inventory.json) lists the small JSON evidence files and their SHA256 values. Reproduction scripts and numeric fixtures are saved in [benchmarks/patch9](../benchmarks/patch9/). The large derived images, NumPy arrays and cube exports remain in the local audit directory.

The scripts were run from one audit directory so `inputs/`, `functional/`, and the frozen model directories align. Configure the UI/native resource paths near the top of the UI scripts for another checkout; no UI source changes are required. All reproduction outputs should use a fresh directory. The package scripts accept `--site` and `--models-from` explicitly.

Typical order:

```powershell
python bench_ui_api.py --functional --output functional
python bench_ui_api.py --output benchmark-ui
python bench_package_api.py --site <P78-wheel-site> --output benchmark-p78
python bench_final_source.py --site <local-clone> --models-from benchmark-p78 --output final-source
python bench_final_png.py --site <local-clone> --models-from benchmark-p78 --output final-png
# Set TEMP/TMP before Python imports for the actual GPU probe:
python bench_package_detect.py --site <local-clone> --output detect-p9-explicit-temp
python check_ui_fast_accuracy.py --benchmark benchmark-ui --output ui-fast-accuracy
python check_preview_stress.py --site <local-clone> --models-from benchmark-p78 --output preview-stress
python check_endpoint_stress.py --site <local-clone> --models-from benchmark-p78 --output endpoint-stress
```

After audit hardening, the old one-code full-stage profile correctly returns direct instead of reproducing the prototype acceptance. The endpoint probe deliberately exposes the fixed129 table using loose measurement-only budgets and preserves the counterexample independently of promotion.

Numerical source SHA256 at the final timing revision:

- `acceleration.py`: `71fcb846a4fb9955e2e047ba76159c097a1797320285a7150bd57b3da7fdf4be`
- `core/execution.py`: `dfc852cdcbf0353bc0c8a0bb05d5c43d47e2b723df76259a64f71482fdddd424`

Complete timing-revision and frozen-P78 hashes are in [source_fingerprints.json](validation/patch9/ui/source_fingerprints.json). Subsequent distribution qualification added compatibility adapters for Colour 0.4.1's chart constructors and stored conventional matrices, including singleton inputs. Current Colour APIs delegate unchanged; these adapters do not alter the affine GC/WB/CC math or kernels timed here. Final installed-file fingerprints and the completed regression results are recorded in [VALIDATION_9.md](VALIDATION_9.md). UI tracked git status remained clean. No repository push was performed.
