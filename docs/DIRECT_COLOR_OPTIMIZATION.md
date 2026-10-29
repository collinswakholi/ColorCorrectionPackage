# Direct GC/color execution: implementation and qualification

All work remains in the local clone. Nothing was committed, pushed, tagged or published. This follow-up changes two application files and adds 61 regressions. It accelerates **prepared direct execution**; the established reference prediction APIs and saved numerical model parameters remain unchanged.

## Result and use

The final 4K paired file→strict-PNG median improved **3.74×**, from **11.483 s to 3.066 s**. Direct numerical execution improved **6.39×**, from **10.098 s to 1.581 s**. This uses no combined LUT and no relaxed interpolation profile.

```python
with cc.prepare_execution({"mode": "direct", "backend": "auto"}) as execution:
    corrected = execution.predict_image(input_path).images["CC"]
```

Numba remains optional through the existing `[performance]` extra. Base installations retain the reference NumPy calculation. Use `backend="numpy"` for that reference calculation; strict sampled error budgets can also select the recorded reference. Existing `ColorCorrection.predict_image`/training behavior is retained for compatibility.

## What changed

- A bounded, parallel float64 kernel evaluates the original sRGB→XYZ→Lab→saved neutral PCHIP→XYZ→sRGB equations. It uses the exact runtime D65 matrices and white point, original piecewise thresholds, signed output transfer, clipping and endpoint assumptions. `fastmath=False`; no lookup table, float32 conversion, refitting or rounded replacement matrix is introduced.
- Runtime PCHIP coefficients are derived by SciPy from the original saved knots. Accumulation follows [SciPy's piecewise-polynomial evaluator](https://github.com/scipy/scipy/blob/v1.18.1/scipy/interpolate/_ppoly.pyx#L602). Derived coefficients, locks and compiled code are not added to calibration artifacts.
- Positive diagonal WB is an exact channel scale when fused GC is active. Reference and arbitrary matrix routes remain available. Its equivalence to matrix application is tested bit for bit.
- Overridden models, very narrow/steep curves, unsupported CC estimators and high downstream sensitivity retain reference GC. A conservative downstream gain bound uses the WB matrix and known affine polynomial derivatives on [0,1]. The admission cutoff is 32; narrow intervals below 0.001 L* or secants above 100 also retain reference. These are conservative eligibility limits, not modifications to a model.
- Whole-pipeline qualification and a separate audit enforce **RGB≤5e-12 and ΔE00≤1e-8** for the new direct tone path, in addition to any tighter requested budget. GC is checked independently so a later constant/clipped CC cannot conceal error. These checks are empirical; they are not a proof over arbitrary continuous inputs and models.
- A runtime backend failure during qualification pins the prepared handle to the recorded reference. A later successful JIT retry cannot bypass that decision. Per-frame diagnostics aggregate all chunks, preserve earlier fallback reasons and distinguish mixed execution. Existing GC/WB diagnostic labels remain present. Runtime engine identity is versioned so prior table/proof entries cannot be reused under the new policy.

## Real timings and UI Python API

Windows i9-14900K; Python 3.12.10 audit environment; 8 NumPy/SciPy/OpenMP/Numba threads, all 32 logical CPUs available. No other test/build jobs ran during timings. The final paired process observed 84.4% memory load, 5.32 GB available. Identical frozen calibration and image files were used without refitting. Warm calls follow preparation and one first-file prediction; source hashes are checked before/after.

Final source, alternating baseline/candidate order, n=3 each:

| 4K phase | Frozen image/export baseline | Final source |
|---|---:|---:|
| Direct numerical stages | 10098.29 ms | 1580.70 ms |
| File prediction, including decode/contracts | 10588.49 ms | 2060.64 ms |
| Strict PNG8 export | 972.00 ms | 969.82 ms |
| Complete file→strict PNG8 | 11482.83 ms | 3066.25 ms |

Strict export includes its existing temporary write, flush/fsync, complete encoded-byte read-back, hash, decode, sample/metadata verification and atomic publication. Its unchanged timings are a useful control. The three final PNG payloads match baseline bytes exactly.

The real SDK 2.2.1 / released native 0.2.0 worker was exercised again, n=7, CPU and actual DX12 RTX 4070 Ti SUPER rendering:

| UI renderer / 4K | CPU | Auto GPU |
|---|---:|---:|
| Standard GC, CC LUT33 | 606.82 ms | 631.86 ms |
| Combined65 fast mode | 277.80 ms | 302.77 ms |

The final direct package path remains **5.05×** the UI standard CPU time, or **11.04×** its fast CPU time. The gap remains open. UI uses different fitting, float32/interpolation and publication behavior; even standard mode uses a CC LUT and is not an analytic float64 oracle. Physical calibration superiority is not inferred from this timing experiment.

An earlier n=5 same-process numerical revision measured 617.82→257.87 ms at 512², 4405.66→1561.51 ms at 1080p, and 17642.21→6378.95 ms at 4K. Subsequent hardening affected backend-failure decisions and diagnostics, not the nominal numerical kernels. The final-source 4K recheck above is the primary claim. Fresh-process ABBA controls had drifted from 7.09 to 17.53 s for unchanged baseline code; normal host execution also showed variance. Those raw runs are retained as diagnostic evidence, not selected as the final speedup. No thermal, scheduler or sandbox cause was established.

Within the final paired process, preparation took 1501.40/2513.63 ms baseline/final. The first file predictions took 10699.16/4843.95 ms, including first-use work. They are separate phases, not complete cold-process startup measurements. JIT startup is not hidden inside the warm medians.

## Accuracy, compatibility and persistence

- The exact final source exhaustively processed **16,777,216 byte-RGB inputs** through the real full calibration: maximum RGB difference **7.94e-15**, with **zero uint8/uint16 quantization differences**. CIEDE00 probes covered 1,048,816 colors, maximum **5.03e-13**. These are differences from the frozen implementation, not physical color accuracy.
- Additional kernel checks exhaustively covered the byte cube for identity and plateau/steep curves, plus 100 random monotonic curves with continuous RGB and knot-adjacent Lab probes. The largest byte-cube kernel RGB difference was 2.66e-14. All tested 512/1080p/4K real-photo PNG payloads matched baseline bytes. Float64 arrays may differ by a few ulps; universal float64 TIFF byte identity is not claimed. The reference route remains available.
- **1,185 tests passed in 232.16 s**, zero failures/errors/skips: 1120 unit, 24 distribution, 17 dependency, 24 enabled real-photo cases. The 24 warnings are expected short-iteration PLS/NN convergence warnings. Final targeted execution suite: 143 passed.
- Four isolated installed profiles passed: Python 3.8.20 and 3.12.10, each with/without optional Numba. Actual accelerated tone execution was verified in both performance profiles; base profiles matched the NumPy reference exactly. Each passed model fitting, bundle/batch/FFC/legacy/prepared execution, all five actual older P78 bundles, the image/export parity suite, and direct bundle/trusted-pickle reload equality. No runtime kernel state is serialized.
- Installed 43-file source identity matches the clone. Strict Twine and model/license/archive checks pass. Actual installed chart GPU execution was verified (210 WebGPU/D3D12 nodes, no CPU fallback); detector code and weights are unchanged. Radeon and Apple hardware were not available for measurement. The original wheel, source fixtures and UI checkout are retained.
- Model schemas, fitted knots/coefficients, color semantics, input ownership, writable outputs, intermediate stopping, spatial FFC, atomic calibration/export persistence and CPU fallback remain covered. No fsync/read-back/verification step was removed.

## Artifacts and reproduction

Qualified wheel SHA256: `d491fcdcd9ea5413663f611023eb7498b7cc0ec934d61bd1e9ae9fe1e89b0a08`.

43-file application tree SHA256: `bd6e9d782602c5f8cd2dae5e125201dd0ffe48e40e9b4d9d0f20548c145550f6`.

The final source archive is refreshed after this report; its hashes and Python 3.8 source-build proof are recorded separately under [distribution evidence](validation/direct_color/distribution/), avoiding a self-referential archive hash. Package version stays 1.4.6; artifacts remain local.

Small raw measurements, full source fingerprints, full test XML/log, stress data and installed-profile evidence are in [validation/direct_color](validation/direct_color/). Scripts are in [benchmarks/direct_color](../benchmarks/direct_color/). Large binaries, input PNGs, frozen model bundles and the baseline wheel remain in the local audit directory. Do not run competing workloads during timing. Generic workers accept explicit site/model/input/output paths; historical local drivers retain the original audit paths.
