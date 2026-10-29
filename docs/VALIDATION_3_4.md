# Patches 3–4 and GPU chart detection: local validation

> Subsequent results: [patches 5–6 validation](VALIDATION_5_6.md).

Validated **28 October 2026**, in the local ColorCorrectionPackage clone on
`codex/reliability-model-detection`. Changes remain uncommitted. No push, tag,
release or publication was performed. The comparison UI checkout remains clean.
The [earlier report](VALIDATION.md) records patches 0–2; this report covers the
subsequent contracts, explicit semantics and GPU extension.

## Implemented behavior

- Immutable version-1 `ImageData`, `ReferenceProfile`, `ChartObservation` and
  `ObservationSet`, with validated ownership, shape, encoding, precision,
  orientation, canonical cell IDs, references and content identities.
- Canonical unclipped Lab D50 references, separate clipped display swatches,
  explicit source white/scale and named neutral membership. Known layouts reject
  misordered cell IDs instead of relabeling detector geometry from reference rows.
- Original-source geometry reused across stages, multiple Classic faces retained,
  consistent `chart_index`, sampled-pixel clipping fractions, and full-precision
  explicit measurements independent of the detector's uint8 preview.
- Default `legacy-v1` numerical behavior preserved. Opt-in `explicit-v1` declares
  sRGB transfer/primaries, working D65, canonical reference D50, Bradford/CAT02,
  evaluation white, encoded-RGB fitting and output clipping/boundary encoding.
- Explicit gamma uses direct canonical working-white lightness and pure D65 Lab
  conversion, retaining its existing all-patch polynomial algorithm. This fixes
  negative/out-of-gamut reference re-decoding and global colour-library scale leakage.
- Actual output observations are scored against canonical truth with common-white
  CIEDE2000. These are training-chart residuals, not independent accuracy estimates.
- New model metadata persists contracts, source facts, observations, stage policies
  and detector execution. Loading validates supported policies, reference/source
  assignment and stage/model consistency before activation. Older metadata-free
  models retain legacy interpretation without invented capture provenance.
- Explicit white encoding is consistent across file, byte, float and typed inputs.
  Explicit FFC application rejects mismatched image dimensions; legacy resizing
  remains. Precision/capture-quality improvements to FFC remain later work.
- GPU-first chart execution with observable CPU recovery using the same ONNX model.
  GPU initialization, runtime and malformed-output failures are covered. Failed
  chart geometry also retains completed inference/fallback provenance.

Usage details: [integration and compatibility](contracts-and-color-integration.md),
[contracts](contracts.md), [color semantics](color-semantics.md),
[GPU configuration](gpu-detection.md).

## Complete regression suite

```text
411 passed, 6 warnings in 103.52s (0:01:43)
```

| Group | Passed |
| --- | ---: |
| Unit tests | 381 |
| Dependency checks | 17 |
| Real-photograph integration tests | 13 |
| Total | 411 |

There were no failures or skipped tests. Six warnings were convergence warnings
from deliberately short NN/PLS persistence/interface tests. Their purpose is not
to establish estimator convergence or scientific color accuracy. Timing is not a
controlled performance benchmark.

Coverage includes all preceding reliability regressions plus immutable arrays
after JSON/pickle/deepcopy, uint16 and explicit-range handling, all eight EXIF
orientations, invalid schema/hash/order/template inputs, partial sampled-pixel
clipping, multi-face selection, reference/metadata corruption, unsupported color
policies, output encoding after reload, white-input transport parity, source/global
scale independence and exact gamma canonical-lightness forwarding.

The pure CIEDE2000 implementation passes all 34 published Sharma pairs in both
directions. Matrix/transfer/adaptation/scale tests include known vectors,
round-trips, common-white equivalence, unclipped reference preservation and a
double-adaptation counterexample. Fixture provenance is retained alongside the
tests and documented in color-semantics.md.

One regression demonstrates why the canonical path matters: Lab D50
`[65,120,-110]` has working D65 L* `66.1384717873`; historical RGB re-decoding
produces `68.8221292362`. Explicit mode avoids this `2.68366 L*` error. This is a
constructed color-contract test, not a photographed accuracy result.

The photo suite includes all four public chart photographs, FFC/conventional and
custom CC, fixed-cell measurements, common-white scoring, encoded/linear boundary
equivalence, failure preservation, fresh-process loading and saved interpretation.
Images are resized to bounded longest-edge sizes of 800–1600 pixels. Fit/apply
parity uses absolute/relative tolerance `1e-12`; reload checks are exact in this
environment. Those parity tests disable the unchanged fit-only saturation operation.

The final run was equivalent to:

```powershell
$env:CCP_PHOTO_DIR = (Resolve-Path '..\CCP_New_UI\test_images').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
python -B -m pytest tests -o addopts='' -p no:cacheprovider -q
```

The actual audit redirected temporary directories, Numba/Matplotlib caches,
pytest basetemp and JUnit output into the audit directory. Without the corpus
environment variable, the photo integration tests explicitly skip.

## Real GPU evidence

Host: Windows x64 with NVIDIA RTX 4070 Ti SUPER and Intel UHD770. NVIDIA driver
reported by `nvidia-smi`: 610.88. Core runtime: ONNX Runtime1.31.1 plus official
WebGPU plugin0.4.0. DirectML was tested in a separate environment with runtime1.24.4,
avoiding conflicting ONNX Runtime distributions in one installation.

Three fresh processes each tested the five original-resolution public photos.
All **15 hardware cases passed**: each route detected four chart photos with face
counts **1 / 2 / 1 / 1**, and correctly rejected Blank.JPG. All five face
comparisons per route preserved canonical cell correspondence.

| Route | Observed initial graph execution | Maximum normalized sampled RGB channel difference vs CPU |
| --- | --- | ---: |
| Default auto → native WebGPU/D3D12 | 210 WebGPU nodes, zero CPU nodes | `3.64054e-5` |
| Strict Vulkan, fresh process | 210 WebGPU nodes, zero CPU nodes | `3.64054e-5` |
| Strict DirectML, NVIDIA DXGI adapter1 | One fused DML node, zero CPU nodes | `2.60038e-6` |

Session profiling verifies the first forward; subsequent corpus forwards reuse
that verified, no-CPU-fallback session. ORT provider registration alone is not used
as evidence of execution. WebGPU registration descriptors and Dawn's physical
adapter selection are recorded separately; descriptor identity is not independent
proof of the physical adapter used. DirectML automatic selection chose the NVIDIA
adapter with greater dedicated video memory; explicit Intel adapter0 was also
checked separately.

Actual unavailable-CUDA-runtime recovery selected CPU, detected the expected chart,
and recorded the failed attempt and `cpu_fallback=true`. Unit fault injections
cover failed initialization/forwards, malformed outputs, resource release and
strict no-fallback behavior. A physical GPU OOM/driver-fault campaign was not run.

These are same-photo CPU/GPU equivalence checks, not independent chart-corner or
physical color ground truth. The files retain source/model hashes and test-time
source fingerprints. A later diagnostics-only detector change did not alter
inference, decoding, assembly or sampling. Final full-suite and installed auto-GPU
tests exercised the final application source.

## Provider policy and qualification boundaries

| Platform | Default attempt order |
| --- | --- |
| Windows | CUDA → DirectML → native WebGPU/D3D12 → CPU |
| Linux | CUDA → native WebGPU/Vulkan → CPU |
| macOS | native WebGPU/Metal → CoreML → CPU |

Explicit GPU selections recover to CPU by default; `allow_cpu_fallback=False`
requests strict qualification. Vulkan/Metal use the official native WebGPU plugin,
not an invented Vulkan EP. Changing its process-wide backend/adapter requires a
new process and is reported rather than silently relabeled.

Current native WebGPU defaults are installed for supported Python≥3.11 platforms
with ORT≥1.31 and plugin≥0.4. Older supported Python keeps its compatible runtime
and CPU recovery; GPU use depends on available providers. CUDA/DirectML alternate
runtime wheels must not be co-installed over the base runtime. The conditional
dependency markers were checked for Windows x64/ARM64, Linux x64/aarch64, macOS
ARM64, older Python and excluded macOS Intel plugin cases.

**Radeon hardware, Apple Silicon/Metal/CoreML, CUDA execution and native Linux GPU
hardware were not physically qualified.** Their selection/options/failure policies
have unit coverage. CoreML profiling proves CoreML provider use but cannot identify
its internal GPU/CPU/Neural Engine scheduling. The implementation supports those
provider paths; this report does not claim their host qualification.

## Built and installed package

Wheel and source distribution built successfully. An isolated wheel target was
imported outside the source tree, with every installed Python source file compared
byte-for-byte to the final clone. Model/hash/notices and runtime dependency markers
were verified. The installed wheel actually selected D3D12 and recorded 210 GPU
nodes/no CPU nodes, then passed explicit fit/apply/save/reload.

An actual linear-CC model saved by the earlier patches0–2 package was loaded in the
new package. Direct-size and LUT-size predictions were **bitwise identical** to
the earlier installed package under the same audit dependencies. This is bounded
legacy-model compatibility evidence, supplemented by the frozen legacy fixtures.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Wheel | 92,029,974 | `069f2d16db87368f27c12b4c13224227396ec01f963d410a28e4fcad5052119e` |
| Source archive | 96,024,275 | `4332b185a55693b3c1fd721d02947a1eaf6e23a71155846bd3c18a5dab34578b` |

Artifacts were built from final application source before adding this report.
The audit reused recorded scientific dependencies and system site packages; it
was not a clean operating-system/dependency installation or cross-platform build
qualification. The package version remains1.4.6 locally; no release was planned.

`uv lock --check --offline` and `git diff --check` passed. Model bytes/notices
remain unchanged from the earlier pinned asset.

## Remaining scope

- Classic24 references and supported Classic faces only; other layouts are not
  relabeled. Canonical values remain nominal declared reference data.
- No automatic ICC conversion, arbitrary RGB primaries/HDR pipeline, spectral
  accuracy certification or independent photographed color validation.
- Legacy saturation, monotonic/neutral-only gamma algorithms, NN configuration
  and reproducibility work, regression quality, high-precision FFC and broad batch
  performance remain later patches. FFC shape validation does not prove matching
  lens/crop/lighting/capture conditions.
- Pickle remains trusted-file-only; metadata validation is not an untrusted-file
  security boundary. A portable non-executable model bundle is later work.
- No later patches, remote push, release or publication were started.

Machine-readable [summary](validation/p34/summary.json),
[hardware summary](validation/p34/hardware-summary.json),
[auto](validation/p34/auto-corpus.json), [Vulkan](validation/p34/vulkan-corpus.json),
[DirectML](validation/p34/directml-corpus.json), and
[CPU recovery](validation/p34/cpu-fallback-hardware.json) are retained locally.
Absolute audit paths were reduced to filenames in the copied evidence; numeric
results, content hashes and execution records were retained.
