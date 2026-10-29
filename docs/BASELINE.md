# Baseline for the local reliability and detector transition

This records the pre-change baseline and the acceptance boundaries for patches
0--2. The model-based chart detector is an additional, separately tested change.
All implementation and test work is local. Remote pushes and publication are
outside this tranche.

## Recorded source and test environment

The Python package baseline is version 1.4.6, revision
`ca1935776707e3373693941bc3b9349c57768029`. The comparison UI is revision
`16c6f42176cbd392c167ecabea4bb2519984fda8`. A comparison implementation is a
source of design ideas and fixtures, not independent numerical ground truth.

The unchanged Python package unit suite passed on 28 October 2026:

```text
84 passed, 2 warnings in 58.48s
```

Both warnings were sklearn convergence warnings in deliberately short,
20-iteration NN tests. Timing included fresh per-run caches and concurrent
baseline fixture generation; it is not a controlled performance measurement.

| Component | Recorded version |
| --- | --- |
| Python | 3.12.10, Windows x64 |
| NumPy | 2.4.6 |
| OpenCV contrib | 4.13.0.92 |
| colour-science | 0.4.7 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| Torch | 2.13.0+cpu |
| pandas | 3.0.5 |
| Numba | 0.65.1 |
| pytest | 9.1.1 |
| Plotly | 6.7.0 |
| scikit-image | 0.26.0 |

The audit used a temporary virtual environment with system site packages.
These results establish the recorded environment's baseline, not a clean-wheel
installation, every advertised Python version, GPU correctness, or photographed
color accuracy. Clean installation and supported-version qualification need
separate results.

From the repository root, the unit-test invocation was equivalent to:

```powershell
$taskAuditRoot = Join-Path $env:TEMP 'ccp-local-tests'
New-Item -ItemType Directory -Force -Path $taskAuditRoot | Out-Null
$env:TEMP = $taskAuditRoot
$env:TMP = $taskAuditRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NUMBA_CACHE_DIR = Join-Path $taskAuditRoot 'numba-cache'
$env:MPLCONFIGDIR = Join-Path $taskAuditRoot 'matplotlib'
$env:MPLBACKEND = 'Agg'
python -B -m pytest tests/test_unit -o addopts='' -p no:cacheprovider --basetemp (Join-Path $taskAuditRoot 'pytest') -q
```

Use the recorded dependencies when reproducing the historical result. A test
run with newly resolved package versions must record its own environment.

## Numerical compatibility snapshots

`tests/fixtures/legacy_numeric_baseline.json` was generated from the unchanged
baseline revision in the environment above. Its SHA-256 is
`5f60963b49829abff71356747ff07c3b3e63ebf734a156efad4ae897aa0e5e9f`.

`test_legacy_numerical_baseline.py` checks four detector-independent contracts:

- Classic24 reference names, ordering, reference RGB, and default illuminant.
- Applying a supplied, nonidentity gamma polynomial to recorded RGB values.
- The existing `0.95` WB headroom factor, including matched input/reference data.
- The existing saturation helper's pixel scaling and threshold behavior.

These fixtures prevent a reliability or detection patch from silently changing
the meaning of existing transform parameters. They do not endorse the physical
accuracy of the legacy saturation or gamma algorithms. Intentional future
algorithm versions should get separate fixtures; do not regenerate this fixture
merely to make a failing compatibility test pass.

Reference values use absolute tolerance `1e-10`; WB and saturation use `1e-14`.
Gamma application uses `2e-6` to allow small supported CPU-backend differences in
the existing fast color conversions. These are numerical compatibility limits,
not perceptual accuracy thresholds.

## Reproduced pre-change defects

| Defect | Observed behavior | Required regression coverage |
| --- | --- | --- |
| Transient LUT callback is pickled | A real fitted default linear model cannot be saved | Unbuilt and built LUT round-trips using real estimators |
| Saving opens the destination first | Failed serialization truncates an existing file to zero bytes | Serialization and replacement failures preserve previous bytes |
| Reused pipeline retains disabled stage models | An old WB gain remains active after an all-disabled successful run | A successful new calibration owns a complete stage snapshot |
| Configuration is mutated | Missing white input changes caller `do_ffc` | Caller config and nested dictionaries stay unchanged |
| Error aggregation occurs after saving | Partial or failed calibration can reach save logic | Failed required stages do not publish successful artifacts |
| Missing charts may return identity models | GC, WB, and conventional CC can appear successful | Required detection failure is explicit and inspectable |
| Unknown CC method returns a false error flag | Invalid request appears successful | Invalid method is rejected or reports failure |

## Transaction and error acceptance contract

The transaction boundary is a complete `ColorCorrection.run()` call. Existing
individual stage methods are lower-level operations; their ownership behavior
must not be confused with the full-run guarantee.

A successful new run replaces the previous calibration, including disabled
stages. A failed run retains the previous complete model and reference context.
That context includes illuminant, chart, reference DataFrames, input/white image
attributes, and model-path state where they are owned by the pipeline. An
unexpected exception must preserve state just as a reported stage error does.

Failures must identify their stage and reason. A disabled stage, a successful
identity transform, a permitted fallback, and a failed fit are different outcomes.
Required chart or white-field inputs must not be silently converted into
successful fitting. A fallback, where supported, must report which backend
actually ran and why the preferred one was unavailable.

Saving requested as part of the run is part of that run's success contract. A
save failure must not publish a candidate as the successful active calibration.
Serialization must complete to a temporary file in the destination directory,
then replacement publishes the new file. Cleanup must not remove the previous
destination. Existing results must survive errors before replacement.

Atomic replacement of one model file is not a multi-file transaction. Reports
or CSVs written by a run must not be described as an all-or-nothing export set
unless that property is implemented and failure-tested explicitly. In
particular, saturation CSV output must not be written before later required
stages can fail.

Pickle loading remains a trusted-file compatibility operation. Type/shape or
container validation after unpickling does not make arbitrary pickle files safe
to open. Loading failure must leave all active model attributes unchanged.

## Model detector transition boundary

Detection tests need both deterministic decoder/geometry fixtures and actual
model inference on photographs. Mocked outputs alone do not establish model
compatibility. The supported model contract, model identity, inference provider,
fallback policy, and selected chart layout must be recorded.

The original pipeline's references are Classic24. SG and Spyder detections must
not be reinterpreted as Classic24 rows. Multi-layout reference support belongs
to a later, explicit capability. Model absence, inference failure, no chart,
unsupported layout, and low-confidence rejection are different conditions.

The existing UI sample photographs are `ColorCard_Big.JPG`,
`ColorCard_Combo.JPG`, `Sample_3572.JPG`, `Sample_3573.JPG`, and `Blank.JPG`.
They are useful local integration inputs, not independently annotated detector
ground truth. Their UI location and the model assets must not become implicit
runtime dependencies of an installed Python package.

## Work deliberately left for later patches

This tranche does not change the fitting domain, canonical reference values,
tone-curve algorithm, WB headroom, legacy saturation transformation, or FFC
physics. Known limits remain: incomplete NN seeding, ignored NN parameters,
weak sampled-image cache identity, fit/apply saturation mismatch, and differences
between LUT backends outside their documented input range. These require their
own versioned numerical changes or focused reliability patches.

The wider plan also includes complete color contracts, independent evaluation,
a non-executable versioned model format, higher-precision/ICC I/O, bounded batch
execution, and release gates. Passing this tranche's tests does not imply those
later capabilities are complete.
