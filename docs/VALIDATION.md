# Local patches 0–2 and model detector validation

> See [patches 3–4 and GPU validation](VALIDATION_3_4.md) for the subsequent 411-test result and hardware evidence.

Validated on **28 October 2026**, Windows x64, Python 3.12.10, CPU execution.
Local branch: `codex/reliability-model-detection`. Baseline revision:
`ca1935776707e3373693941bc3b9349c57768029` (package version 1.4.6).
Changes remain local and uncommitted. No push, tag, release, or publication was performed.

## Implemented scope

- Recorded the original 84-test baseline and froze detector-independent reference,
  gamma-application, WB-headroom, and saturation fixtures.
- Fit candidates separately, commit complete state only after success, preserve
  prior models/reference context on failures, and give each prediction/batch a
  consistent model-container snapshot.
- Copy caller configuration and calibration pixels; stop at failed required stages;
  record stage status, reason, timings, artifacts, and detector diagnostics in `last_run`.
- Treat missing enabled FFC input, missing required charts, unknown correction
  methods, and invalid candidate outputs as failures. Optional FFC chart metrics
  can be unavailable without invalidating an otherwise valid flat-field fit.
- Repair real-estimator persistence by reconstructing transient LUT callbacks.
  Write/read-check/fsync temporary model files before atomic replacement. Preserve
  previous files and active state on failure. Retain recovery backups if run-export
  restoration itself fails, and report cleanup failures separately.
- Preserve direct prediction for legacy model states that predate LUT settings.
- Default to the bundled primary 1280 ONNX chart model on CPU, covering neutral,
  conventional and extended CC sampling. Preserve canonical cell identities and
  reuse geometry from the original source independently of optional metric flags.
  Keep explicit MCC selection and opt-in, reported fallback.
- Make FFC cache identity cover full white-image content and automatically used
  detector bytes, preventing stale gains after small input or same-path asset changes.
- Include the model, identity manifest, original notices and AGPL text in local
  source/wheel distributions; update the dependency lock for ONNX Runtime.

See [behavior and compatibility](local-reliability.md),
[detector configuration](chart-detection.md), and [original baseline](BASELINE.md).

## Final test result

```text
202 passed, 6 warnings in 211.49s (0:03:31)
```

| Group | Passed |
| --- | ---: |
| Unit tests, including new regression/adversarial cases | 177 |
| Dependency checks | 17 |
| Real-photograph integration tests | 8 |
| Total | 202 |

There were no failed or skipped tests. Six convergence warnings came from short
NN/PLS estimator tests. These validate interfaces, finite predictions and
persistence, rather than convergence or general color accuracy. Suite timing
includes shared-machine activity and is not a controlled performance measurement.

The recorded audit dependencies are listed in BASELINE.md; this tranche also
used ONNX Runtime 1.31.1. The audit venv uses system site packages. The complete
dependency graph was not installed in a fresh operating-system environment.

From the clone, with the recorded dependencies available, the final invocation
was equivalent to:

```powershell
$env:CCP_PHOTO_DIR = (Resolve-Path '..\CCP_New_UI\test_images').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
python -B -m pytest tests -o addopts='' -p no:cacheprovider -q
```

The actual audit additionally redirected TEMP/TMP, Numba/Matplotlib caches,
pytest basetemp and JUnit output into the disposable audit directory.
Without `CCP_PHOTO_DIR`, the eight photo integration tests explicitly skip.

Coverage includes disabled-stage reuse, failed fitting and saving, full context
rollback, concurrent prediction during fit, refitting during a batch, malformed
loads, failed serialization/fsync/replace/read-back, incomplete restoration,
legacy model migration, and linear/PLS/sklearn-NN/custom-Torch model round-trips
with both unbuilt and built LUTs.

## Real photographs and geometry

Separately from the 202 pytest cases, CPU detection ran on five public photographs
at exact 0/90/180/270-degree rotations: **20 cases**.

| Photograph | Expected Classic faces per rotation | Result |
| --- | ---: | --- |
| ColorCard_Big.JPG | 1 | All four detected |
| ColorCard_Combo.JPG | 2 | All four detected |
| Sample_3572.JPG | 1 | All four detected |
| Sample_3573.JPG | 1 | All four detected |
| Blank.JPG | 0 | All four rejected with NoChartDetectedError |

All 15 rotated-face comparisons retained canonical order using the tested
flip/transpose comparisons and nearest measured-color correspondence. Rotated
sampling is not numerically invariant: mean normalized RGB absolute differences
ranged from 0.000424 to 0.009355, with a maximum single-channel difference of
0.061572. Source image hashes and per-case results are in
[detector-rotation-results.json](validation/detector-rotation-results.json).
These are demonstration/geometry regression inputs, not independent physical
color truth or a broad chart-detection benchmark.

The eight integration tests resize photographs to at most 1600 pixels on the
longest edge to bound memory. They cover all four chart photographs, FFC with
both conventional/custom CC, failed blank-image fitting preserving the previous
model file, and loading/prediction in a fresh process. FFC uses an explicit
white-plane crop; automatic plane detection is not newly qualified here.

Fit/apply results match within absolute/relative tolerance 1e-12; saved/reloaded
results match exactly in the recorded environment. Toggling optional FFC metrics
produces identical fitted output. These parity tests disable the legacy
saturation operation, whose fit/apply mismatch is intentionally deferred.

## Package validation

Both wheel and source distribution built successfully. An isolated installed
wheel target was imported from outside the source directory and passed a real
photo fit/apply/save/reload smoke test. Every installed Python source file was
checked byte-for-byte against the final clone's application source.

The wheel and source archive include the bundled model; its exact SHA-256 is:

```text
1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240
```

The model is 39,354,948 bytes, IR version 8, default-domain opset 17. Model notices,
license text and manifest are packaged. Wheel metadata declares ONNX Runtime.
The installed smoke test reported `CPUExecutionProvider`. Package validation
reused the recorded audit dependencies; it is not a clean dependency-install or
cross-platform qualification.

The retained code-validated wheel SHA-256 is
`e154309d76ffb2ea0bbf075917e5af483a6b1b442470a5414940c87419c76820`
(91,999,183 bytes). Source archive SHA-256 is
`a31702cb0cd8c20a8e3abad10916b73f6bf9f50e2716ac45a00931f361c6d43a`
(95,962,675 bytes). Artifacts were built from final application source before
adding this report. Rebuilds may differ in archive metadata.

`uv lock --check --offline` passed, and `git diff --check` found no whitespace
errors. The comparison UI checkout was left clean.

## Scope limits before later patches

- Classic24 references, including Passport Classic faces, are supported. Spyder
  and Digital SG are not relabeled as Classic. Unsupported-only layouts fail;
  mixed images can select supported faces and report ignored layouts.
- The new detector changes measured patch values. This tranche establishes
  bounded functionality and geometry behavior, not improved physical accuracy.
- Legacy GC fitting, saturation policy/fit-apply mismatch, WB headroom, NN
  configuration/reproducibility work, color-space contracts, high-precision I/O,
  and broad scientific validation remain later patches. Existing prediction
  helper input/LUT-domain semantics are not comprehensively redesigned here.
- `.pkl` remains a trusted-file format that can execute code during loading.
  Atomic persistence does not turn it into a safe untrusted interchange format.
  Run exports have ordinary-failure recovery, not crash-atomic multi-file commits.
- GPU execution, other operating systems, other Python/runtime versions, optional
  1536 recall inference and production-scale batch performance were not qualified.
- No later pipeline patches or remote actions were started.

Machine-readable counts and artifact identities are in
[summary.json](validation/summary.json).
