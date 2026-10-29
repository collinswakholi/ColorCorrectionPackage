# Patch 9 — measured acceleration and distribution qualification

All implementation and qualification were performed in the local
`ColorCorrectionPackage` clone. No commit, push, tag, release, or external CI run
was performed. Package version remains 1.4.6. Patches 0–8 remain in the working
tree alongside this tranche.

## Implementation

- `prepare_execution()` owns one portable calibration snapshot and exposes direct
  or empirically qualified combined GC→WB→CC execution. FFC stays spatial and
  separate; declared color/precision boundaries stay outside the table.
- Known affine polynomial models have explicit NumPy and fused float64 Numba
  execution. Existing prediction entry points retain their numerical policies.
  Numba is optional through `[performance]`; unavailable/unsupported backends
  report their actual execution or reject a strict request.
- Combined LUTs require qualification, independent audit and a mixed stress
  audit. Default stress uses 572,733 colors: centers, byte-lattice near-black/white
  cubes and independent quantized random colors. Failure selects direct execution
  without retuning on audit data. These remain empirical tests, not all-domain
  error bounds. Strict defaults reject both tested real-photo calibrations.
- Immutable caches are bounded and keyed by full learned content and execution
  policy. Legacy per-model table construction is synchronized, detects producer
  changes and inference overrides, and omits stale tables when persisting.
  Closing prepared handles waits for active calls. Cube export streams and
  verifies an atomic file replacement; spatial FFC exclusion is explicit.
- Qualification exposed Colour 0.4.1 API gaps on Python 3.8. Capability-based
  chart construction preserves current geometry while accepting the older
  three-field type. Stored conventional matrices use the same historical feature
  expansion/dot expression without refitting. Singleton expansion is padded only
  for the older API. Current library application delegates unchanged.
- Builds disclose bundled-model license declarations, verify exact model bytes,
  exclude the unused training `.pt`, and qualify installed wheels. CI is defined
  for source/build/installed checks; publishing is gated behind an explicit
  release/manual action and a named environment requiring repository setup.

Usage: [prepared execution](acceleration.md),
[distribution profiles and limitations](distribution-qualification.md).

## Final regression results

The final complete run passed **1,027 tests**, with **zero failures/errors/skips**,
in **254.83 seconds**: **962 unit**, **24 distribution**, **17 dependency**, and
**24 real-photo** tests. Its 24 warnings were expected short-cap PLS/NN convergence
warnings. All 24 opt-in real-photo tests were enabled through `CCP_PHOTO_DIR`.
The old/current Colour compatibility cases also passed a separate targeted run.

Coverage includes all prior numerical/transactional/contract/bundle/batch/I/O
regressions, polynomial degrees 0–8, backend failure and fallback, complete
parameter/settings invalidation, concurrent single builds, owned immutable
tables, stale-before-save pickle/bundle cases, independent audit refusal,
intermediate-stage stopping, FFC separation, concurrent close/export, and atomic
failure preservation. A numeric real-model fixture reproduces and rejects the
near-black interpolation counterexample.

## Actual performance

Windows i9-14900K workstation. Numerical comparisons use eight CPU threads;
medians/p95/raw observations and the measured memory-pressure condition are in
[the complete UI/API report](PERFORMANCE_9.md). No parallel test/build workloads
ran during timing windows. These are measured host results, not universal speed
guarantees.

| Measured path | Reference median | New median | Conditions |
|---|---:|---:|---|
| Degree-1 CC, 4K | 354.13 ms | 135.77 ms | Frozen P78 direct vs fused Numba; n=5 |
| Degree-3 CC, 1080p | 429.97 ms | 85.44 ms | Frozen P78 direct vs fused Numba; n=5 |
| GC/WB/CC, 4K in memory | 5,851.34 ms | 822.31 ms | Alternating direct vs explicitly relaxed 129³ LUT; n=7 |
| Chart detection, file | 638.50 ms CPU | 226.66 ms GPU | Same process; seven warm calls, verified WebGPU |

Across the five direct-kernel cases, Numba improved median time **2.61×–5.40×**,
with maximum absolute output error **6.66e-16**. Explicit NumPy was bitwise equal
to reference but slower in those cases; it is a fallback, not a claimed speedup.
The first Numba invocation cost **1,068.16 ms**, including JIT/cache initialization.

The **7.12×** whole-pipeline in-memory gain requires explicit approximation budgets
**RGB≤0.006 / ΔE00≤1.0**, with a fixed 129³ grid. On the tested 4K image, maxima
were **0.005348 RGB / 0.848027 ΔE00**; means were **3.35e-5 / 0.004925**. Default
budgets stay at **0.002 / 0.5**, selecting direct for these calibrations. An early
one-code profile failed dense near-black testing and is retained as negative
evidence; the strengthened audit now rejects it.

Final file→strict-PNG measured **5,625.76 ms**, n=3, including decode, prediction,
atomic publication, fsync, read-back verification and hashes. UI Python API 4K
GC/WB/CC file→PNG medians were **281.27 ms CPU / 348.15 ms GPU**, n=7, using its
combined 65³ fast renderer and different model/precision/persistence policies.
The package still has a substantial end-to-end latency gap. Kernel-only gains
must not be presented as file-workflow gains. Cold relaxed preparation measured
**6,110 ms** in the fresh-process export experiment; cache-hit preparation was
**52.78 ms** median. Startup conditions and full distributions are preserved.

## Real UI API comparison

The current SDK **2.2.1** was exercised against the real released Windows **0.2.0**
worker, with its executable hash and source-revision limitations recorded in the
performance report. No UI source was changed. The same frozen P78 bundles feed
all package comparisons. A separate shared-24-cell experiment controls sampling
while acknowledging encoded-RGB least squares versus the UI's linear-light
ridge/intercept model. Reconstructed UI coefficients agree to **1.68e-15**.

UI tests cover four full-resolution photos, blank failure, saved-model reload,
33³/65³ cube export, and mixed 8/16-bit batch behavior. Six main timing rows cover
CC-only and full GC/WB/CC at 512², 1080p and 4K, on UI CPU/GPU rendering.
Fast/standard UI PNG8 comparisons expose quantized differences and do not treat
the UI standard path as a float64 oracle. Training residuals do not establish
physical accuracy or generalization.

## Installed artifacts and compatibility

All four final installed profiles passed. Qualification
uses truly isolated Windows Python **3.8.20** and **3.12.10** base and performance
environments, with no inherited system/user site packages. Actual pip installation,
dependency checks, model fitting/bundle/application, precision I/O, legacy gamma,
new prepared execution, honest unavailable-Numba handling, file/mixed batches and
a bundled real-photo workflow are exercised in every profile. Actual old P78
portable bundles for all five families are compared across these dependencies.

| Profile | NumPy | Numba | Colour | Result |
|---|---|---|---|---|
| Python 3.8 base | 1.24.4 | absent | 0.4.1 | Passed |
| Python 3.8 performance | 1.24.4 | 0.58.1 | 0.4.1 | Passed |
| Python 3.12 base | 2.5.4 | absent | 0.4.7 | Passed |
| Python 3.12 performance | 2.4.6 | 0.65.1 | 0.4.7 | Passed |

Every profile passed actual pip installation and dependency checks, exact
installed/wheel/source equality, photo fit/save/load/apply, legacy reference and
gamma checks, new backend/fallback behavior, precision I/O, and all five actual
old portable bundles (20 loads). Legacy reference RGB differs by at most
**1.11e-15**; legacy gamma by **7.49e-16** and GC/WB bundle reload by zero.
Separate actual Colour **0.4.1 and 0.4.7** audits each passed 11 conventional
option cases, including singleton/vector/image shapes and no-refit sentinels.
Whole-image application matches native bitwise; the old broken singleton oracle
requires duplicated rows and a **2e-14** GEMM/GEMV rounding tolerance.

The final wheel contains **43 Python files**, all byte-identical to the clone,
and is **71,407,971 bytes** (versus P78's 92,084,288 bytes). Wheel SHA256:

`24b22616af3214ce1784213744340d63e46b808560227fcf7f37218eb3dcb758`

Python-source tree SHA256:

`d411c65c1e76a3d4a4e227b950484b26f16e151072c9757b991ace15acf77728`

The source archive is refreshed after this report, without changing the qualified
wheel. Its final hash is recorded in the separate artifact inspection JSON to
avoid embedding a source archive's own hash inside itself.

The installed-target migration additionally loads actual P56 and P02 saved
calibrations. All five P56 families match frozen P78 bitwise under a matched
eight-thread environment. Portable reloads, fresh-process predictions and mixed
batches pass. P02 large/small outputs match its original installed code bitwise.
The historical stored P56 custom-NN array differs at 22 float32 values by at most
**5.96e-8** under this context; frozen P78 reproduces that same difference.
This is pre-existing context sensitivity, not a P9 regression. Prepared custom-NN
chunking differs by at most **1.19e-7**, within its recorded **2e-6** tolerance.

Actual default chart GPU execution is verified in both installed-target and
isolated-Python-3.12 workflows: **WebGPU/D3D12, 210 GPU nodes, zero CPU nodes**.
The first successful GPU detector file call cost **3,578.95 ms**. Initial sandbox
tests could not reopen an ORT profile under AppContainer temporary storage and
honestly fell back to CPU. Explicit writable `TEMP`/`TMP` before Python imports
resolved verification; those failed attempts remain in the evidence.

The final source suite and archive checks preserve exact hashes for the 39,354,948-
byte chart detector and 44,658,747-byte plane detector. Wheel inspection retains
their notices and excludes the unused `.pt` without changing the ONNX paths.
Twine strict checks, source-file equality and actual installed dependency versions
are recorded with the artifact/profile JSON.

## Limits and next decisions

- Windows was physically tested. Linux/macOS CI is defined, not executed here;
  Radeon and Apple hardware were unavailable. No new timing claim covers them.
- Python 3.8/3.12 resolved dependency sets are tested; lower dependency bounds,
  all intermediate versions, and interpreters above 3.12 are not exhaustively
  qualified. External five-photo tests require their corpus; default CI instead
  includes one bundled-photo installed smoke.
- LUT tests are empirical. Runtime direct execution remains the path for strict
  accuracy. FFC cannot be represented in a pointwise cube. Exact semantic parity
  with UI fitting is not claimed.
- Strict image-contract/verification work remains expensive. Further performance
  work should profile redundant copies and export read-back cost while preserving
  their guarantees. This tranche does not silently weaken those defaults.
- Taichi remains an experimental investigation from patches 5–6; it is not a
  production dependency/backend. See [the recorded investigation](TAICHI_INVESTIGATION.md).
- Bundled-model declarations are disclosed and retained. Artifact checks do not
  resolve missing training-data/base-weight rights records. Repository/PyPI
  protected-environment and trusted-publisher settings require separate setup;
  no publication or remote qualification was performed.

Raw timing and UI evidence: [inventory](validation/patch9/ui/raw_inventory.json),
[direct kernels](validation/patch9/timing-direct.json). Reproduction scripts:
[benchmarks/patch9](../benchmarks/patch9/). Final source/test/profile/artifact JSON
and XML live in `docs/validation/patch9/`; large media and frozen historical
installations remain in the local audit directory.
