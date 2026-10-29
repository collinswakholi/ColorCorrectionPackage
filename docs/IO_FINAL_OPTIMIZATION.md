# Decode/validation and strict-export optimization

This local follow-up preserves the preceding implementation's default encoded bytes, numerical outputs, image identities and export facts for the tested inputs. It adds an explicit PNG representation that changes file bytes while preserving decoded integer samples and every strict publication barrier. All measurements use the frozen model/input files without refitting. Nothing is pushed, tagged, published or dispatched to remote CI. The complete authorized patch set is prepared as a local review commit; see [push preparation](PUSH_PREPARATION.md).

## Final measured result

Final balanced, same-process public API comparison: n=12 per policy and resolution, covering every policy permutation twice, after preparation and one first complete file call. Times are wall-clock medians, in milliseconds.

| File→verified PNG8 | Frozen preceding engine | Compatible default | Explicit fast-lossless PNG |
|---|---:|---:|---:|
| 512 | 93.63 | 99.26 | 91.82 |
| 1080p | 522.30 | 492.99 | 451.77 |
| 4k | 1880.50 | 1674.84 | 1505.32 |

| 4K phase | Frozen preceding engine | Compatible default | Explicit fast-lossless PNG |
|---|---:|---:|---:|
| Decode, normalization and image contracts | 471.82 | 403.33 | 419.74 |
| Direct stages + output checks | 563.11 | 532.55 | 575.93 |
| Strict export | 835.53 | 735.20 | 520.93 |
| Complete file→strict PNG8 | 1880.50 | 1674.84 | 1505.32 |

The 4K complete-file speedup is **1.12×** by default and **1.25×** with the explicit PNG profile. Phase medians are independent; the complete total is measured directly. Small images do not show a consistent end-to-end gain. No direct-color kernel was changed, so variation in that control phase is not attributed to an I/O optimization.

The baseline is the qualified wheel `e0a915bf606acc4ddfae4c4f36f6de3034c6fd9d48da9579b3dd01d929634f26`. The paired harness loads its exact reader, writer and `from_array` method; the shared initializer, post-init, hash functions, decoder and color equations are unchanged. The shared type also has the candidate's ownership-only pixel-header isolation. An independent fresh process exercises the complete original wheel as an additional control. The benchmark does not treat the shared-helper first-call timings as independent cold-process startup.

The same Windows x64 i9-14900K workstation and eight numerical threads are used. No competing test/build jobs ran during final timings. Raw call order, all samples, p95, preparation/first-file costs, process CPU time, thread pools, host memory and source/input/model hashes are retained in [paired results](validation/io_final/paired-balanced-final/results.json). Separate candidate fresh-process runs cover [compatible](validation/io_final/fresh-final-compatible/results.json) and [fast-lossless](validation/io_final/fresh-final-fast-lossless/results.json); the [full original-wheel control](validation/io_final/fresh-baseline/results.json) is separate from the primary paired result. Earlier cyclic-order runs showed variation in the unchanged direct stage; the final comparison balances both pair directions and positions instead of selecting a favorable earlier run.

## What changed

1. Large normalization fills a private preallocated float64 buffer directly, releases writable views and exposes immutable bytes. It avoids repeated chunk allocation/write/setup. Generic/endian/unusual-range inputs retain NumPy conversion and the original subtraction/division order.
2. Full-range native uint8/uint16 RGB can use an optional fused float64 CPU normalizer. Every possible integer code is intrinsically finite and within its declared full range; first-use qualification checks every code and a strided channel permutation. The public ImageData constructor still validates the complete owned normalized image and computes both existing SHA256 identities.
3. Integer export quantization fuses multiply→round-to-even→cast and the original divide→subtract→abs→maximum error equation. `fastmath=False`; no float32 intermediate, approximate division, omitted error fact or reduced source digest is introduced. First-use qualification checks all code values, all half-bin boundaries and their adjacent representable floats, plus independent probes. Failed qualification selects NumPy. A partially written runtime destination is completely overwritten from the source before NumPy recovery returns a result; debug logs record fallback.
4. Arrays below **8 MiB canonical float64 size** keep NumPy arithmetic, avoiding unhelpful small-image JIT startup. Main-thread large calls use parallel CPU kernels; background workers use GIL-releasing serial kernels without nested native worker pools. No global thread count, CPU affinity or process priority is changed by library code. Numba remains optional; base installations retain NumPy.
5. Each public `ImageData.pixels` access now gets an independent NumPy header over the same immutable bytes. A read-only ndarray otherwise allows shape/dtype/stride edits, which could invalidate typed-image digest and export facts. Local reshaping of a returned view remains possible, while the canonical contract stays unchanged. Metadata schemas, values and normal-input digests are unchanged; no caller-supplied trusted flag bypasses validation.

The native pixel decoder, ICC/alpha/grayscale/orientation policies, allocation caps, file growth checks, digest formats and model/bundle formats remain unchanged. The additional header isolation is an explicit ownership hardening, not a new data conversion.

## Optional fast-lossless PNG

```python
from ColorCorrectionPipeline.io import write_image_data

record = write_image_data(
    "corrected.png", corrected_float_rgb,
    dtype="uint8", allow_precision_loss=True,
    png_profile="fast-lossless",
)
```

`png_profile="compatible"` is the default and keeps the established encoder/quality behavior. `fast-lossless` writes truecolor RGB PNG8/PNG16 using filter None and zlib compression level zero, with the same transfer/primaries/project metadata. It intentionally overrides PNG quality/compression settings. The format follows [PNG3 filtering and compression](https://www.w3.org/TR/png-3/). Non-PNG destinations reject this option. Batch `export_options` accepts the same profile.

Lossless here describes the codec relative to the selected integer samples. Float-to-integer quantization still requires explicit precision-loss consent and reports the same maximum quantization error. Different encoded bytes, SHA256 and file size are expected for the opt-in profile. Default bytes are preserved. No lossy codec, reduced bit depth or interpolation tolerance is substituted for speed.

Both profiles retain temporary same-directory output, flush/fsync, complete byte read-back, encoded SHA256, bounded decoding, full shape/dtype/format/metadata checks, every lossless sample comparison, atomic no-clobber/replace publication and cleanup after failure. Quantization/encoding finish before publication. No source identity, precision consent or validation pass is removed to make the timing competitive.

## Real UI Python API comparison

The real local SDK 2.2.1 and released native worker 0.2.0 were exercised again, n=7 per mode/backend/resolution, with previously frozen respective calibrations. The worker executable SHA256 remains `123cb49377f0729e1c542dfbcdc096fff2a75f1cb6fea61f020cf7db7377a4fd`.

| Resolution | UI standard CPU | UI standard auto GPU | UI combined65 CPU | UI combined65 auto GPU |
|---|---:|---:|---:|---:|
| 512 | 25.51 | 23.20 | 14.77 | 16.01 |
| 1080p | 138.34 | 135.73 | 75.49 | 74.49 |
| 4k | 589.31 | 623.26 | 278.34 | 276.80 |

At 4K the package's default strict workflow takes **2.84×** the standard UI CPU time; its explicit fast PNG workflow takes **2.55×**. The end-to-end gap remains open. The UI uses different fitting, float32/interpolation and publication behavior; even standard mode uses a CC LUT. These timings do not establish equal numerical functions or physical calibration accuracy. The package retains analytic float64 direct execution and its existing strict persistence contract.

Auto UI replies report actual DX12 GPU execution, accelerated pixels and zero CPU pixels. The current native binary was not rebuilt from the UI checkout; source-revision limitations remain in [the earlier comparison](VALIDATION_9.md). Raw [standard](validation/io_final/ui-standard-final/results.json) and [fast](validation/io_final/ui-fast-final/results.json) replies are retained. The UI checkout remains clean.

## Accuracy and battle tests

- **1,261 tests passed**, zero failures/errors/skips. The 25 warnings comprise 24 deliberately short NN/PLS convergence warnings and one intentional deprecated stride-assignment test. There are **42 new regression cases** covering all integer codes, striding/endianness/orientation, mutable input capture, public header mutation, exact rounding/error facts, rejected qualification, partially written backends, worker execution, both PNG transfer encodings/depths, no-clobber, wrong samples/metadata and fsync/read-back/publish failure cleanup.
- **3,277,506 source values** were checked for both uint8/uint16 on main-parallel and worker-serial routes against the frozen quantizer. They include all nonnegative float16 representations through one, two million finite float64 bit patterns, one million uniform values, uint16 code/half-bin/neighbor probes, signed zero and endpoints. Quantized bytes and maximum error facts match exactly. Separate singleton checks exercise subnormals and per-value error facts, preventing a larger half-bin error from concealing a difference.
- Every final paired run matches complete real-photo float64 output bytes. Compatible PNG bytes and every ExportRecord field except destination path match the preceding implementation. Fast PNG source/error facts match; its common publisher verifies decoded samples exactly. No physical accuracy advantage is inferred from implementation equality. These tests are empirical, not a proof over every possible floating-point/platform combination.
- Four isolated installed profiles pass: Python 3.8.20 and 3.12.10, base and performance. They verify actual eligible I/O kernel execution, exact normalized bytes/metadata against frozen contracts, compatible encoded bytes/all facts against the frozen writer, fast decoded pixels/precision/source facts, serial worker behavior and partial-failure recovery. Existing fitting/bundle/batch/FFC/legacy/prepared execution, all five older P78 bundles and strict-image suites also pass.
- Actual installed chart inference again verifies 210 WebGPU/D3D12 nodes with no CPU fallback. Model bytes and provider code are unchanged. Radeon/Apple hardware and remote OS CI are not newly certified by these Windows tests.
- Strict Twine, exact model/license/archive checks, dependency checks and 44-file installed/source identity pass. The final documentation-bearing source archive is rebuilt and installed offline using Python 3.8 after this report; its evidence is recorded separately under [distribution](validation/io_final/distribution/) to avoid a self-referential archive hash.

## Decisions and artifacts

The early NumPy vector-buffer normalization prototype improved its isolated median only modestly. Fused quantization showed a large isolated gain; complete-file measurements above remain the primary performance claims. A final all-serial I/O scheduling experiment differed by only 1–2% in complete-file median and showed no robust reason to replace the chosen main/worker policy. Small-frame JIT crossover was raised to the shared 8 MiB threshold. Raw exploratory timings are retained and are not substituted for final-source results.

Qualified wheel SHA256: `c26d3e1906a3cc26e8ebef2d1d12269d6e28828efd647a799359154775740b2f`.

44-file application tree SHA256: `fac4e692ad5664513ef3ec282121e0f1269be20855a4838bcb95b47bb52aa48a`.

Small final raw evidence is in [validation/io_final](validation/io_final/). Portable installed, paired and fresh-process workers are in [benchmarks/io_final](../benchmarks/io_final/). The large original/candidate wheels, archives, image inputs and frozen models remain in the local audit directory. Earlier reports are historical qualification snapshots; this report and [PUSH_PREPARATION.md](PUSH_PREPARATION.md) describe the latest local state.
