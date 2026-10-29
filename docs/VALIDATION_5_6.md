# Patches 5–6: local implementation and validation

> Subsequent results: [patches 7–8 validation](VALIDATION_7_8.md).

Validated **28 October 2026**, in the local ColorCorrectionPackage clone on
`codex/reliability-model-detection`, Windows x64/Python 3.12.10. All changes remain
local and uncommitted. No push, tag, release or publication was performed, and no
later patch was started. The comparison UI checkout remains clean.

This follows the [revised patch sequence](patch-sequence.md), with the completed
[patches 0–2](VALIDATION.md) and [patches 3–4](VALIDATION_3_4.md) preserved. The
[usage and policy guide](numerical-policies-and-validation.md) defines the new
behavior. The separately authorized [Taichi investigation](TAICHI_INVESTIGATION.md)
contains actual CPU/NVIDIA/Intel Vulkan timings, precision errors and cold costs.

## Implemented scope

| Area | New opt-in behavior | Compatibility boundary |
| --- | --- | --- |
| Saturation | Original sampled-pixel clipping evidence, per-cell/channel source reporting, explicit report/exclude policy, identity image operation | Historical fit-only 0.95 scaling remains in legacy mode |
| GC | Declared neutrals only, measured sufficiency before endpoints, duplicate/contradiction checks, monotonic PCHIP in explicit D65 Lab, exact assumed 0/100 endpoints | Earlier polynomial models and evaluator remain; degree-one fitting bug repaired |
| WB | Named usable neutrals, minimum signal and bounded positive gains, recorded configurable headroom | Default headroom remains 0.95; 1.0 is an intentional choice |
| FFC | Guarded Lab-lightness mode: raw signal/saturation, explicit crop/coverage, fit rank/condition, positive bounded observed/fitted/interpolated gains, declared no-plane fallback and dimensions | Existing valid gradient gain/image arrays match legacy bitwise; no radiometric rewrite |
| Fitting | Equal independent-cell means, deterministic CPU methods, effective controls, explicit weighted synthetic assumptions, full/weighted design checks and stable conventional root-polynomial fit | Legacy methods and numerical paths remain default |
| Evaluation | Deterministic grouped CC-only folds, canonical common-white CIEDE2000, exhaustive accounting, no aggregate on incomplete folds | No whole-pipeline or independent-capture accuracy claim |
| Persistence | Recorded policies, fitting controls/dependencies, observation selection and validation; incompatible modern model/version/basis/gain metadata rejected before activation | Earlier trusted pickle containers remain supported; portable format is patch 7 |

Enable new stages using `stage_policy='modern-v1'`, new fitting using
`fitting_policy='reproducible-v1'`, and `color_semantics={'version':'explicit-v1'}`.
New CC requires both policies. Existing configurations and earlier explicit-v1
models keep their prior interpretation. Modern GC/WB/CC reject interactive show
requests, unsupported/irrelevant estimator options and legacy pseudo-sampling
controls rather than accepting unused settings.

The numerical changes are not promoted to legacy defaults. Canonical references
stay unclipped and measurements stay distinct from synthetic endpoint assumptions.
Fit and apply share the exact new tone, WB, FFC and CC application paths.

## Complete regression result

```text
601 passed, 8 warnings in 150.14s (0:02:30)
```

| Group | Passed |
| --- | ---: |
| Unit tests | 565 |
| Dependency checks | 17 |
| Real-photograph integration | 19 |
| Total | 601 |

No failures, errors or skips. The earlier 411-test scope remains passing; 190 cases
were added. Eight warnings come from deliberately short NN/PLS estimator tests,
including two four-iteration MLP tests. These are interface/persistence checks,
not evidence that those short fits converged. Suite timing is not a performance
benchmark; building artifacts overlapped part of the final regression run.

The final command was equivalent to:

```powershell
$env:CCP_PHOTO_DIR = (Resolve-Path '..\CCP_New_UI\test_images').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
python -B -m pytest tests -q -o addopts='' -p no:cacheprovider
```

Temporary files, caches, basetemp and JUnit output were redirected into the audit
directory. Without `CCP_PHOTO_DIR`, photograph integration tests explicitly skip.
The retained [JUnit file](validation/p56/final-results.xml) and
[final log](validation/p56/final-tests.log) record the successful run.

## Adversarial qualification

- Source clipping cannot disappear after FFC/GC. A partly clipped patch with an
  unclipped-looking mean is still flagged and excluded by the default policy.
  Reporting mode deliberately retains flagged cells without changing measurements.
- Colored cells/targets cannot alter neutral-only GC. Too few neutrals, insufficient
  distinct measured range, contradictory targets and invalid duplicates fail
  calibration before artificial endpoint anchors can supply fake sufficiency.
- Dense neutral ramps remain finite/monotonic. Real photos exposed PCHIP rounding
  at 100 L*: `99.999999999999986` rather than exactly 100. The evaluator now enforces
  its declared boundary assumptions exactly instead of weakening the contract.
- WB rejects near-black/invalid observations and unbounded/nonpositive gains.
  Headroom 1.0 identity fixtures and recorded 0.95 operation are tested separately.
- FFC rejects black and single-black-pixel crops before smoothing, excessive
  saturation, invalid/empty/out-of-bounds crop, inadequate area, duplicate bins,
  invalid smoothing/degree/seed, rank/condition failures, invalid raw/fitted gains
  and interpolation overshoot. No-plane full-image fallback must be explicit.
- Safe valid-gradient FFC retains bitwise legacy gain/application behavior.
  Pipeline cache reuse preserves quality records and prediction requires calibrated
  dimensions. Missing/NaN/reversed limits, wrong dimensions and corrupt gains cannot
  bypass modern model loading.
- Repeated cell observations do not acquire extra weight; inconsistent targets
  fail. Explicit black/white assumptions carry their own weights and are train-only.
  Extreme weights must pass weighted design rank/conditioning checks.
- Linear rank, PLS centered-rank component limits, effective learning rate,
  iteration caps, method-specific controls and unsupported search are exercised.
- Repeated CPU fits reproduce in fresh processes for linear, PLS, sklearn MLP and
  custom Torch NN. Torch default dtype/device changes cannot move the new CPU fit
  or silently change its precision; CPU RNG/determinism state is restored even on
  optimizer failure. Library concurrent fits serialize and reproduce.
- Group folds do not share repeated colors or held-out targets. Synthetic rows
  stay in training. Partial failure and all-excluded cases have complete accounting;
  no aggregate is invented. A viable full high-degree fit can still have deficient
  folds; those scores are unavailable with the recorded reasons.
- Corrupt estimator versions, options, degree/method labels, absent effective
  reports and conventional matrix/basis mismatches fail loading without replacing
  active models. This is integrity validation, not a pickle security boundary.

## Photograph and installed-package evidence

All four public chart photographs pass new full GC/WB/CC fit, fixed-geometry reuse,
accounted validation, fit/apply parity and save/load. The six added photo cases
also cover a matching synthetic white-field gradient with a real chart image and
the expected blank-chart failure. Images use a bounded longest edge of 1000 pixels
for this tranche's integration tests. Synthetic white-field use validates pipeline
mechanics, not a real camera's illumination correction.

Fit/apply comparisons use absolute/relative tolerance `1e-12`; new-model reload
predictions are exact in this environment. Training-chart metrics resample actual
transformed pixels. Held-out evaluation is explicitly CC-only/direct-on-region-means,
with upstream stages fixed; its records cannot establish independent capture accuracy.

Wheel and source distribution build successfully. The wheel was installed into an
isolated target and imported outside the source checkout. Every installed Python
source file matches the final clone byte-for-byte; bundled detector bytes remain
the pinned asset. The installed default detector actually executed on
WebGPU/D3D12: **210 GPU graph nodes and zero CPU graph nodes**.

| Installed family | Fit/apply parity | Fresh-process reload |
| --- | --- | --- |
| Polynomial linear | Passed | Bitwise equal |
| PLS | Passed | Bitwise equal |
| sklearn MLP | Passed | Bitwise equal |
| Custom Torch CPU NN | Passed | Bitwise equal |
| Conventional Finlayson | Passed | Bitwise equal |

NN installed smokes intentionally use four iterations to verify execution and
persistence, not estimator convergence. An actual linear-CC model saved before
patches 3–4 was also compared with its earlier installed package. Both direct-size
and LUT-size predictions remained **bitwise identical** under the same dependencies.
This is bounded legacy-model evidence, supplemented by the frozen regression tests.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Wheel | 92,051,871 | `6362c3ea8cca988b388ea39758ff4d82bf554273ce89cdee170e580274ff481f` |
| Source archive | 96,255,515 | `dfb93576cf0511165ecc65bb70b2f1b52e365375acc54c5a96ef42f668a9347f` |

Artifacts contain final application source and the benchmark deliverables before
this concluding report was added. Validation reuses the recorded Windows scientific
dependencies/system site packages; it is not a clean OS/dependency installation or
cross-platform certification. Package version remains 1.4.6 locally. Final source
fingerprints and installed results are retained in the
[machine summary](validation/p56/summary.json) and
[installed evidence](validation/p56/installed-validation.json).

## Taichi result and scope boundaries

**128 actual timing cases** completed on CPU and two physically selected Vulkan
adapters: NVIDIA RTX 4070 Ti SUPER and Intel UHD770. They include real-photo and
seeded-random arrays, 512²/1080p/4K where supported, 3 warmups and 11 repetitions,
cold/JIT costs, median/p95, transfer-inclusive and resident timings, and numerical
agreement against frozen patches 0–4 original functions. No unexpected execution
failure was hidden. An actual float64 noninteger-power compilation rejection is
retained as an expected Vulkan qualification limit.

Taichi CPU float64 helps selected fused operations, while existing Numba LUT/Lab
kernels are faster in these tests. Vulkan host transfers/conversions frequently
erase resident-kernel gains. GPU color-transform experiments are explicitly float32;
CPU float64 and NVIDIA arithmetic-only float64 results are separate. Large direct
polynomial speedups include avoiding original feature materialization/garbage
collection with LUT disabled; the deployed LUT default does not receive that gain.
The new patch-6 reproducible CPU feature path was not the timing baseline.

The unchanged original ONNX chart detector measured **494.196 ms CPU** versus
**183.955 ms Vulkan** warm median on the original 6000×4000 photo. Those are ONNX
Runtime reference measurements, not Taichi neural-network inference. The current
ORT provider architecture and production numerical backend defaults are retained.
Taichi remains isolated research code, without a production import or dependency.

Radeon, Apple Silicon/Metal, native Linux/ARM and Taichi CUDA were not physically
measured. Full independent-capture accuracy, arbitrary primaries/HDR/automatic ICC,
linear/high-precision radiometric FFC, GPU-training reproducibility, a portable
non-executable model format and batch/resource redesign remain outside this tranche.
Torch RNG isolation does not protect unrelated external threads concurrently using
Torch global state. Existing trusted-file pickle remains the persistence format.

The [complete Taichi report](TAICHI_INVESTIGATION.md) contains every timing table,
precision limits and prioritized engineering recommendations. Reproduction code
and the frozen source baseline are under [benchmarks/taichi](../benchmarks/taichi/README.md).
Copied source hashes and numerical-function equivalence were independently checked.

`uv lock --check --offline` and `git diff --check` passed. No further patch or remote
operation was started after these local validation gates.
