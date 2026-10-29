# Bulk direct pipeline optimization and final local qualification

This follow-up is implemented in the local `ColorCorrectionPackage` clone. No commit, push, tag, release, remote CI dispatch or publication was performed. It changes two application files relative to the previously qualified direct-color wheel and adds **34 adversarial regression cases**. The application source has been frozen for the measurements and installed-profile checks below.

## Measured result

Against the immediately preceding qualified direct implementation, 4K direct stages and output checks improved **2.12×**. Complete file→verified PNG8 improved **1.36×**, to **1.951 s**. No LUT, float32 substitution, model refit or relaxed accuracy budget is used.

Primary result: alternating order in one process, n=7 per engine and resolution, after preparation and one first-file call. All encoded PNG payloads and the measured real-photo float64 outputs match the preceding implementation exactly.

| Complete file→strict PNG8 | Preceding direct engine (ms) | Bulk direct engine (ms) | Speedup |
|---|---:|---:|---:|
| 512 | 116.86 | 101.30 | 1.15× |
| 1080p | 842.01 | 591.51 | 1.42× |
| 4k | 2662.10 | 1951.46 | 1.36× |

| 4K phase | Preceding direct engine (ms) | Bulk direct engine (ms) | Speedup |
|---|---:|---:|---:|
| Direct stages + output checks | 1378.42 | 650.28 | 2.12× |
| File prediction including decode/contracts | 1837.93 | 1133.53 | 1.62× |
| Strict PNG8 publication | 837.00 | 824.85 | 1.01× |
| Complete file→strict PNG8 | 2662.10 | 1951.46 | 1.36× |

Each phase is its own median; the complete total is measured per call. Raw samples, p95, call order, process CPU time, input/model identities, thread pools, host memory and first-use/preparation phases are in [paired results](validation/bulk_direct/paired-final/results.json). The strict writer is unchanged. Its timing variation is recorded rather than interpreted as another optimization.

The machine is the same Windows x64 i9-14900K workstation used in earlier reports. Both package engines use eight numerical threads. No competing test/build jobs ran during timing. The baseline is the qualified `d491fcdcd9ea5413663f611023eb7498b7cc0ec934d61bd1e9ae9fe1e89b0a08` wheel. The paired harness loads its exact prepared-execution class beside the candidate; the shared numerical helper functions have identical ASTs, except that RGB validation now uses scalar reductions instead of temporary Boolean arrays. A separate fresh-process candidate run records preparation at **3517.06 ms** and first 4K file prediction at **999.69 ms**. These are separate phases, not end-to-end cold-process startup. The paired engines share helper/JIT state, so their first-use figures are not independent cold-start measurements.

## Changes and compatibility

- Known, unmodified neutral-PCHIP tone models, positive diagonal WB and bounded portable/reproducible affine polynomial CC use whole-image scheduling. Runtime parameter derivation and parallel dispatch move out of the per-chunk loop. No polynomial feature image is allocated.
- Tone creates an owned writable result. WB and CC reuse that private buffer. The CC kernel reads each row completely before writing it; rows are independent. Requested GC/WB intermediates get independent copies. Source pixels, saved models and caller-owned arrays remain untouched.
- This keeps full-frame storage proportional to source, final output and explicitly requested intermediates. It avoids an additional full-frame CC result in the bulk path; it does not claim a measured whole-process peak-memory reduction from the preceding chunked engine.
- `chunk_pixels` continues to bound feature workspace for other models. Legacy stages, arbitrary WB matrices, unknown estimators, overridden kernels/models, narrow/steep curves and unbounded/high-sensitivity downstream models keep their established reference/chunked routes. Base installations and explicit NumPy selection retain their existing calculation.
- `fastmath=False`, float64 equations, feature order, coefficients, PCHIP knots, clipping, stage order, color semantics and model schemas remain unchanged. Qualification and independent audit retain the direct limits RGB≤5e-12 and ΔE00≤1e-8, independently checking GC as well as the final pipeline. These are empirical checks, not an all-continuous-domain proof.
- If in-place execution partially writes and then fails, fallback recomputes every requested stage from the untouched source. Strict mode raises. A qualification-time failure pins the prepared handle to the recorded reference, even if the kernel later recovers. Existing diagnostic keys are retained; bulk scheduling and fallback are explicitly reported. The runtime cache identity advances to `qualified-pointwise-v4-bulk-direct`; no kernel/derived plan is serialized.

The public usage remains:

```python
with cc.prepare_execution({"mode": "direct", "backend": "auto"}) as execution:
    final_rgb = execution.predict_image(input_path).images["CC"]
```

Numba is optional through `[performance]`. The original `ColorCorrection.predict_image`/training APIs keep their established behavior. See [execution policies](acceleration.md).

## Real UI Python API comparison

The real local SDK 2.2.1 and released native worker 0.2.0 were retimed, n=7, using the same input files and the respective previously frozen calibrations, without refitting. The worker SHA256 is `123cb49377f0729e1c542dfbcdc096fff2a75f1cb6fea61f020cf7db7377a4fd`. These are persistent-worker file→PNG8 calls.

| Resolution | UI standard CPU (ms) | UI standard auto GPU (ms) | UI combined65 CPU (ms) | UI combined65 auto GPU (ms) |
|---|---:|---:|---:|---:|
| 512 | 22.46 | 23.24 | 14.50 | 24.12 |
| 1080p | 137.48 | 135.51 | 74.27 | 86.98 |
| 4k | 551.40 | 606.92 | 275.87 | 308.27 |

At 4K, the package's complete strict file path is still **3.54×** the UI standard CPU time, and **7.07×** its combined65 CPU time. The end-to-end UI gap has narrowed; parity has not been reached. Decode/contracts and strict publication remain substantial measured costs. Different fitting, arithmetic precision, interpolation and persistence behavior prevent treating the UI timings as a float64 accuracy oracle or attributing the entire gap to any one policy. Standard UI mode also uses a CC LUT. No physical calibration superiority is inferred.

The two API implementations retain their respective frozen fits: package encoded-sRGB OLS and UI linear-light ridge/intercept. The native binary was not rebuilt from the current UI checkout. Source-revision limitations remain as recorded in [the earlier API comparison](VALIDATION_9.md). See [standard UI raw results](validation/bulk_direct/ui-standard/results.json), [fast UI raw results](validation/bulk_direct/ui-fast/results.json), and [fresh package results](validation/bulk_direct/fresh-final/results.json).

## Accuracy and persistence qualification

The full real calibration was checked against its unaccelerated recorded float64 reference on all 16,777,216 byte-RGB colors. Identity and plateau/steep tone kernels were also exhausted against their reference equations. CIEDE00 was evaluated on 1,048,816 probes per case, including each worst-RGB location. There were zero uint8 and uint16 quantization differences in every case.

| Case | RGB inputs | Maximum RGB difference | Maximum ΔE00 on probes | Changed uint8 / uint16 channels |
|---|---:|---:|---:|---:|
| real full pipeline | 16,777,216 | 7.92e-15 | 4.26e-13 | 0 / 0 |
| identity tone | 16,777,216 | 2.65e-14 | 3.4e-13 | 0 / 0 |
| plateau/steep tone | 16,777,216 | 5.44e-15 | 5.48e-13 | 0 / 0 |

An additional 100 monotonic curves were tested with continuous RGB and knot-adjacent Lab probes. Real 512², 1080p and 4K outputs match the preceding qualified engine's float64 values and PNG bytes exactly. These results quantify implementation differences, not physical accuracy on independent captures. Universal float64 TIFF byte identity against unaccelerated math is not claimed.

An independent export-parity run compared **every ExportRecord field except destination path**, source/output digests, quantization facts and encoded bytes for 512² PNG8, PNG16, float64 TIFF and 4K PNG8. All four cases match the preceding engine exactly. Existing precision consent, metadata checks, SHA256 facts, fsync, complete byte read-back, decoded sample verification, no-clobber/atomic publication and failure cleanup remain active. The writer, bundle codecs and model weights were not changed by this follow-up.

- **1,219 tests passed in 244.38 s**, zero failures/errors/skips: 1154 unit, 24 distribution, 17 dependency and 24 enabled external-photo cases. The 24 warnings are expected deliberately short NN/PLS convergence runs. The separate new bulk suite also passed all 34 cases. Coverage includes partial writes, nonfinite output before clipping, strict failures, qualification pinning, retained stages, stopping, normalized polynomials through degree eight, immutable inputs, concurrent calls, model snapshots and reloads. Initial synthetic-fixture normalization labels were corrected before this final run; no failed/interrupted attempt is counted as final qualification.
- Four isolated installed profiles passed: Python 3.8.20 and 3.12.10, each with and without optional Numba. Both accelerated profiles execute bulk GC/WB/CC; base profiles match NumPy reference. Fitting, batch, FFC, legacy behavior, all five older P78 bundles, image/export parity, and exact bundle/trusted-local-pickle reloads passed. No runtime kernel state is saved.
- Actual installed chart GPU execution was verified: **210 WebGPU/D3D12 nodes**, `execution_verified=true`, `cpu_fallback=false`. CUDA/DirectML were not advertised by that installed ORT; their failed attempts are reported before successful WebGPU selection. Radeon and Apple hardware were not available for new measurements. Cross-platform CI remains a future remote gate.
- Strict Twine, inference-asset/license checks, exact 43-file installed/source identity and dependency checks passed. The documentation-bearing sdist is refreshed after this report and built/installed offline using Python 3.8; final source-build hashes and results are recorded separately under [distribution evidence](validation/bulk_direct/distribution/), avoiding a self-referential archive hash.

## Local artifacts and push preparation

Qualified wheel SHA256: `e0a915bf606acc4ddfae4c4f36f6de3034c6fd9d48da9579b3dd01d929634f26`.

43-file application tree SHA256: `b91f9673d98e6bf0f311f6e7b506334f52f8af164f7060447f1f772b3bd3305d`.

Small raw JSON, complete final test XML/log, source fingerprints, profile logs and stress/export facts are retained in [validation/bulk_direct](validation/bulk_direct/). Large wheels, model bundles, source archives and input PNGs remain in the local audit directory. Portable paired, stress and installed-wheel probes are in [benchmarks/bulk_direct](../benchmarks/bulk_direct/); the existing generic direct/UI workers are in [benchmarks/direct_color](../benchmarks/direct_color/).

Reproduction requires the frozen baseline site, input PNGs and original calibration bundles identified by their hashes in the raw results:

```shell
python benchmarks/bulk_direct/bench_paired.py --site /candidate/clone --baseline-site /frozen/d491/site --models /frozen/package/models --inputs /inputs --output /new/paired --repetitions 7
python benchmarks/bulk_direct/stress_accuracy.py --site /candidate/clone --baseline-site /frozen/d491/site --models /frozen/package/models --inputs /inputs --output /new/stress
python -I benchmarks/bulk_direct/verify_installed.py --expect-numba yes --output /evidence/direct.json
```

The push-preparation inventory, draft PR description, base/version checks and remaining release gates are in [PUSH_PREPARATION.md](PUSH_PREPARATION.md). No remote action is authorized by this report.
