# Patches 7–8: portable calibration, I/O, batch and reporting validation

Validated **28 October 2026**, in the local ColorCorrectionPackage clone on
`codex/reliability-model-detection`, Windows x64/Python 3.12.10. Changes remain
uncommitted. No push, tag, release or publication was performed. No patch 9 work
or production accelerator/default change was started. The comparison UI remains clean.

This extends the completed [patches 5–6](VALIDATION_5_6.md). Usage and precise
contracts are documented in [calibration bundles](calibration-bundles.md),
[image I/O](image-io-policies.md), and [batch/reporting](batch-and-reporting.md).

## Implemented scope and compatibility

| Area | New behavior | Preserved compatibility |
| --- | --- | --- |
| Portable calibration | Version-1 `.ccpb`: strict JSON plus numeric NPY arrays, known inference adapters, hashes, limits and atomic read-back-validated save/load | Existing pickle methods remain separate trusted-file APIs; no format guessing or pickle fallback |
| Model fidelity | FFC gains, polynomial/PCHIP GC, WB, conventional bases and linear/PLS/sklearn MLP/package Torch parameters, including normalization, activations, BN, LUT values/policy and provenance | Earlier model interpretation remains identified; missing legacy capture metadata stays unknown |
| I/O | Explicit precision-preserving PNG/TIFF, declared encoding, orientation/ICC/alpha/gray/frame decisions, atomic publication and export facts | Legacy reader/writer defaults and return types remain, including legacy uint8 writing |
| Batches | One owned calibration snapshot, bounded generator consumption/reordering, per-item success/failure/cancellation, callbacks, selected stages and file mode | Legacy `predict_images` list/exception behavior and default four-stage single-image results remain |
| Reporting | Atomic finite JSON with item identities, errors, timings, source/encoding/export facts and aggregate accounting; no pixels/opaque models | `run` still returns its existing tuple and retains diagnostics in `last_run` |

`Config(model_format='bundle', save=True, ...)` selects bundle output from a fitted
run; pickle remains default. `MyModels` and `ColorCorrection` also expose explicit
`save_bundle(path)`/`load_bundle(path)` methods. Loading constructs and validates a
complete replacement before activation. Unsupported estimators, hooks and versions
fail rather than being serialized opaquely. Bundle reconstruction is inference-only;
optimizer/training state is not resumed.

Portable CC feature generation has an explicit CPU float64 schema. Torch forward
operations use CPU float32. Unbundled legacy models retain their previous feature
path. Migrating a legacy CUDA-float32 feature path is not qualified as bitwise equal;
other legacy stages, including GC execution choices, are not rewritten by this format.

## Complete regression suite

```text
872 passed, 23 warnings in 255.90s (0:04:15)
```

| Group | Passed |
| --- | ---: |
| Unit tests | 831 |
| Dependency checks | 17 |
| Real-photograph integration | 24 |
| Total | 872 |

There are no failures, errors or skipped tests. The prior 601-test scope remains
passing; **271 new cases** cover this tranche:

| Added module | Cases |
| --- | ---: |
| Calibration bundle | 107 |
| Explicit image I/O | 89 |
| Bounded batch engine/reporting | 40 |
| Public prediction workflows | 30 |
| Portable/batch photograph integration | 5 |

The 23 warnings are expected convergence warnings from deliberately short NN/PLS
tests. Those tests verify parameters, interfaces and persistence; they do not claim
converged color accuracy. Suite elapsed time is not a performance benchmark.

The final run was equivalent to:

```powershell
$env:CCP_PHOTO_DIR = (Resolve-Path '..\CCP_New_UI\test_images').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
python -B -m pytest tests -q -o addopts='' -p no:cacheprovider
```

Caches, temporary files, pytest basetemp and JUnit output were redirected into the
audit directory. Photograph cases explicitly skip without the corpus variable;
the reported run supplied it. Retained evidence: [JUnit](validation/p78/complete-results.xml),
[log](validation/p78/complete-tests.log), [machine summary](validation/p78/summary.json).

## Adversarial and fidelity checks

**Bundle format.** Known legacy/modern families were exercised with built/unbuilt
LUTs, direct and image-size prediction, PCHIP endpoints, scalar intercept precision,
PLS scaling, dense MLP activations and Torch BN/dropout eval behavior. Conventional
tests include legacy Finlayson `root_polynomial_expansion=None` and `terms=None`,
alongside explicit root choices. Integer legacy GC/constant models retain exact
representable parameters. Unsupported classes, hooks, instance inference overrides,
versions, feature orders and incompatible stage/model metadata fail clearly.

Malformed cases include duplicate/traversal/undeclared entries, links, compression,
encryption, ZIP64/multi-disk/comment/prefix forms, header/directory disagreement,
overlap/unclaimed bytes, invalid JSON/duplicate keys/depth/nonfinite values, unknown
arrays, object/structured/subarray NPY dtypes, forged sizes/shapes/hashes, truncated
payloads and expanded network alias/width/parameter limits. NPY declared header
length is checked **before** NumPy's header reader can allocate/decode it. Disk
reads are bounded and re-check regular-file identity/size. No archive member is
extracted and no artifact-specified Python class, pickle or Torch object loader is
invoked. Rejected candidates do not retain self-bound LUT callback cycles.

Hashes detect inconsistency, not authorship or authenticity. Payload/parser limits
are configurable safeguards, not strict process-RSS or comprehensive security
certification. Finite parameters and loader probes also do not prove finite output
on every possible input; actual new batch outputs are checked for shape/range/finiteness.

**I/O.** Asymmetric RGB uint16 PNG and uint8/16/float32/64 TIFF round-trips verify
sample order and preservation. Float TIFF uses uncompressed IEEE samples through
tifffile, avoiding OpenCV float RGB LogLuv loss. Quantization, integer encoding
requantization and lossy codec choices require/record explicit decisions. Transfer
metadata is honored or a conflicting declaration is rejected. EXIF orientations,
grayscale, alpha/compositing, frames, valid/malformed ICC and unsupported high-bit
ICC conversion are covered. Metadata and pixels use one captured byte payload.
File/pixel/allocation caps are checked before copying/decoding large data. Encoder,
readback, no-clobber and replacement fault cases preserve prior destinations.

**Batches.** Tests cover mixed valid/missing inputs, duplicates, callbacks failing
under both policies, source iteration/close failure, cancellation before submission
and during export, early close, changing caller models and arrays, and closed-stream
input/model lifetime. Both pending and ordered-ready records count toward the same
in-flight bound. Weak-reference tests verify no growing result history or retained
closed calibration buffers; no hard RSS benchmark is claimed.

Configured LUT preparation occurs once on the owned snapshot; failed preparation
records direct fallback and cannot race worker retries. Chunked CC selects LUT or
direct evaluation from the whole image once, including short final chunks. This
bounds per-call feature workspace without a tail silently selecting another path.
It is not a new accelerator backend. Existing default single-image inference is
unchunked; old numerical expectations remain covered.

Each file-mode item stages selected images plus its JSON manifest, then publishes
one unique index directory. Second-stage encoder failure or observed cancellation
does not leave a visible partial item. Cleanup failures expose recovery facts without
hiding the original export error. Summary-report failures preserve prior reports
and successful item outcomes. Running native work/publication cannot be forcibly
interrupted; cancellation/early-close accounting distinguishes unsubmitted, queued,
completed and abandoned-not-emitted work.

## Photograph and installed-wheel qualification

All four chart photographs pass modern bundle fit/save/load, fixed-geometry reuse,
fit/apply parity and mixed-input file batches. The added fifth case compares uint16
TIFF file and equivalent-array calibration/application. These use a bounded longest
edge of 900 pixels. The complete 24-photo suite also retains prior FFC,
negative-chart and color-contract coverage; rotation fixtures remain in the wider
regression suite. These verify mechanics, not independent
capture transfer or physical color accuracy.

Wheel and source archive built successfully. An isolated wheel target was imported
outside the source checkout; every installed application Python file matches the
final clone byte-for-byte. New modules, tifffile requirement and pinned detector
bytes were checked in the artifact. The installed default detector physically used
WebGPU/D3D12 with **210 GPU nodes and zero CPU graph nodes**.

Actual models saved by patches 5–6 were loaded and converted, rather than simulated
old serialization layouts:

| Earlier fitted family | Existing pickle pixels | Bundle conversion | Fresh-process bundle pixels | Mixed new batch |
| --- | --- | --- | --- | --- |
| Polynomial linear | Bitwise unchanged | Bitwise equal | Bitwise equal | Passed |
| PLS | Bitwise unchanged | Bitwise equal | Bitwise equal | Passed |
| sklearn MLP | Bitwise unchanged | Bitwise equal | Bitwise equal | Passed |
| Custom Torch CPU NN | Bitwise unchanged | Bitwise equal | Bitwise equal | Passed |
| Conventional Finlayson | Bitwise unchanged | Bitwise equal | Bitwise equal | Passed |

Whole-image bundle conversion was checked against `2e-12` absolute/relative tolerance
and actually had zero difference for all five artifacts. Fresh-process reloads were
exact. Chunked mixed batches used `2e-12` for float64 paths and `2e-6` for the Torch
float32 path, without a general cross-hardware bitwise promise. The installed wheel
also fitted a new GPU-detected chart, saved/reloaded its bundle and completed a file
batch; uint16 PNG samples round-tripped exactly. An actual patches 0–2 legacy linear
model retained bitwise-equal direct-size and LUT-size predictions against its earlier
installed package.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Wheel | 92,084,288 | `1e63b86dedcbf1686d2d3d5a13a21eb7bda99ce1c619779eefcfc2cca885242b` |
| Source archive | 96,326,818 | `00a6cdbefe63f8bf51082851185e6b6b15f58a4a4a4fcdf895c80992ad71c84f` |

Artifacts were built from final application source before this concluding report.
The package version remains 1.4.6 locally. Validation reused recorded scientific
dependencies/system site packages; it is not a clean operating-system/dependency,
minimum-version or cross-platform qualification. Versions, source fingerprints and
installed results are in [installed evidence](validation/p78/installed-validation.json).

## Remaining boundaries

- A collected `BatchResult` intentionally retains results. Use streaming/file mode
  and discard emitted arrays for memory bounded with respect to batch length.
  Working memory still depends on image dimensions, decoder/intermediate buffers,
  estimator width, calibration arrays, chunk size and concurrency.
- ICC conversion supports the documented uint8 path; associated alpha and unsupported
  high-precision profile conversion fail explicitly. Arbitrary primaries/HDR and
  complete external-viewer color management are not claimed. Some image-format
  encoding descriptions require a receiving application that honors them.
- Portable Torch inference is CPU float32; legacy CUDA migration and untested
  architecture/version bitwise identity are not certified. Existing chart-provider
  qualification boundaries remain; no new Radeon/Apple/Linux hardware claim is made.
- Legacy pickle is still trusted-file-only. Unsupported estimators remain available
  through existing Python APIs, not an opaque fallback in the new format.
- Independent-capture scientific validation, high-precision radiometric FFC, broad
  accelerator optimization and distribution/release qualification remain later work.

`uv lock --check --offline` and `git diff --check` passed. No patch 9 or remote
publication work was started after these local gates.
