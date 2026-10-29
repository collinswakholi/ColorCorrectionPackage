# Image validation and strict export optimization

This follow-up optimizes the local patch-9 clone without changing fitting, color equations, execution policy, LUT qualification budgets, or the public writable-array adapters. Nothing was pushed.

## Implementation

- Canonical image construction validates an owned immutable snapshot with scalar min/max reductions instead of allocating several full-frame boolean masks. Native float64 unit-range inputs avoid redundant cast/subtract/divide arrays. Other large inputs normalize in bounded float64 blocks into a once-reserved immutable buffer; the subtraction and division order remains unchanged. Orientation and channel permutation operate on views before normalization. Reuse is limited to exact immutable byte-backed storage with an independent ndarray header. Mutable owners and merely read-only views are copied.
- Both historical SHA256 identities still cover every canonical pixel byte with their original metadata prefixes. Large images overlap the two independent scans using one short-lived worker. There is no cache keyed by a mutable array's identity. Thread submission failure falls back to sequential hashing.
- File reads use bounded chunks through EOF and retain the growth check against `max_file_bytes`. The old one-shot `read(max_file_bytes + 1)` could request approximately 1 GiB for a small PNG. Static PNG metadata inspection avoids Pillow's implicit pixel inflation before the actual OpenCV decode, while preserving header and late EXIF, project encoding, ICC and transparency handling. Ambiguous, animated, malformed or unusual streams retain the conservative path. Allocation limits are checked before PNG pixel decode and before metadata operations that can load pixels.
- Final-only prepared execution borrows an independent read-only view of canonical sRGB. Public `ImageData.to_srgb()`, requested intermediate images, and prediction outputs retain their writable independent-array behavior. Legacy float32 execution keeps its original precision.
- Strict export borrows immutable canonical pixels, quantizes with bounded temporary buffers using the original rounding equation, and swaps RGB channels using OpenCV. Exact sample equality proves zero codec error for lossless output; lossy error calculation retains the original per-value operations in bounded regions. Float64 TIFF avoids an unnecessary full-frame copy.
- Publication still writes a same-directory temporary file, flushes and fsyncs it, reads back every encoded byte, decodes the saved snapshot, verifies all lossless pixels and the requested shape/dtype/format, and only then atomically publishes. No-clobber and overwrite behavior are retained. Read-back now also rejects contradictory encoding, orientation, frame, profile and TIFF photometric metadata before publication.

## Compatibility and accuracy

The regression oracles are frozen patch-9 source snapshots, retained with their tests. They compare complete canonical pixel bytes, full metadata, both image digests, encoded payload bytes and complete export records except the destination path. Transfer conversion, EXIF orientation, uint8/uint16/float16/float32/float64, endian changes, noncontiguous inputs, unusual declared ranges and lossy error reporting are covered.

The owned-float64 range check also rejects rare invalid low-precision inputs that the old NumPy scalar-comparison behavior could incorrectly admit when a declared range endpoint rounded to the source dtype. Valid input results retain the historical byte identity. Late ICC metadata is honored even when a header EXIF chunk previously prevented Pillow from scanning later chunks; explicit rejection policies therefore remain effective.

No new approximate arithmetic or LUT profile was introduced. Existing direct and relaxed-LUT modes are measured separately. The relaxed profile remains an explicit user choice with the same empirical qualification and its existing limits.

## Measurements

Warm public API medians on the same Windows i9-14900K workstation, Python 3.12.10, 8 numerical threads. OpenCV's default policy remains unchanged. Each component has one warmup and three measured calls; profiling and memory tracing run separately. No test/build jobs competed during timing. The baseline is the preserved final patch-9 wheel, not a reimplementation. Inputs and each application's saved calibration remain frozen.

| 4K operation | Frozen P9 | Final clone |
|---|---:|---:|
| Float64 `ImageData.from_array` | 598.57 ms | 242.50 ms |
| `read_image_data` PNG | 1385.68 ms | 485.10 ms |
| Float64 array → strict PNG8 | 2110.09 ms | 853.37 ms |
| Typed uint8 image → strict PNG8 | 1382.21 ms | 582.73 ms |
| Float64 array → strict float64 TIFF | 2042.55 ms | 827.55 ms |
| Existing relaxed 129³ file prediction + strict PNG8 | 3679.20 ms | 1551.72 ms |

The last row is a final-source comparison against a contemporary frozen-P9 control, n=3 per source. Prediction changed from 1634.18 to 699.93 ms and strict export from 2055.25 to 842.41 ms. Total time fell **57.82%**. It uses the existing **explicit** relaxed policy (`mode='combined-lut'`, `grid_sizes=(129,)`, `max_rgb_error=0.006`, `max_delta_e00=1.0`). Its empirical qualification and error budgets are unchanged. It is not the default accuracy policy.

Default/direct numerical execution still dominates its workload. Final direct file processing measured 8140.28 ms, n=3, versus 9519.03 ms for the earlier frozen-P9 ABBA samples, n=6; this is a less tightly paired comparison. Direct remains selected when the original strict budget rejects a LUT.

The real UI Python SDK 2.2.1/released worker 0.2.0 retest measured **285.41 ms CPU / 280.26 ms actual DX12 GPU** for the same 4K file workload, n=7. The optimized relaxed package path therefore remains **5.44×** the UI CPU time, down from **12.89×** for the contemporary P9 control. The gap is narrower, not closed. UI rendering uses its combined65 LUT and different precision, fit and publication behavior.

All 27 final component records match the baseline exactly; all final pipeline prediction pixel SHA256 values and path-normalized export records also match their own frozen direct/relaxed mode exactly. Every encoded payload hash and size matches. No accuracy tradeoff was introduced by this optimization.

Separate 4K traced allocations show large reductions; these are Python-visible/NumPy peaks, not complete process RSS or native codec/driver memory. The reader's old approximately 1 GiB temporary allocation is gone. Exact final memory values, 512/1080p/4K tables, raw samples, source hashes, controls, rejected prototypes and reproducibility instructions are in [the measurement report](IMAGE_EXPORT_TIMINGS.md). The extra quantization reuse was selected only after an isolated ABBA experiment proved exact integer/error bits and a material speedup; final source was then retimed.

## Qualification

The final complete source run passed **1,124 tests in 253.76 seconds**, with **zero failures/errors/skips**: 1,059 unit tests, 24 distribution tests, 17 dependency tests and 24 enabled real-photo tests. The 24 warnings are the existing expected short-iteration PLS/NN convergence warnings. This includes **97 new regression cases** for the contracts, reader and writer. The separate last writer/I/O focused run passed 129 tests; the last image-contract focused run passed 40.

Failure coverage includes fsync, byte corruption, decoded pixel corruption, encoding/orientation/frame/profile contradictions, allocation caps, atomic replacement failures and competing no-clobber destinations. Existing files survive failed strict exports, temporary files are cleaned, and a successful lossless export still proves every sample, not a sampled subset.

The local wheel and source archive pass strict Twine and exact model/license/archive checks. All **43 application Python files** in the wheel match the frozen clone; all 43 also match the source archive. Installation uses isolated `python -I` probes outside the checkout, with no dependency changes or source shadowing:

| Installed profile | Result |
|---|---|
| Python 3.8.20, base (NumPy 1.24.4 / Colour 0.4.1) | Pass |
| Python 3.8.20, performance extra | Pass |
| Python 3.12.10, base (NumPy 2.5.4 / Colour 0.4.7) | Pass |
| Python 3.12.10, performance extra (NumPy 2.4.6) | Pass |

Each profile passes dependency checks, exact installed-source fingerprints, complete fit/save/load/apply/batch/FFC/legacy/prepared-backend smoke checks, and the five actual older P78 portable calibrations. Each additionally passes 208 small and 7 large exact image identity cases, 20 exact-payload/full-record exports and 2 transfer exports, four single-decode PNG cases, three late-metadata cases, two cap refusals and two atomic wrong-encoding refusals. Older cross-version model-output differences stay within existing tolerances (2e-12, or 2e-6 for the custom float32 model); the largest observed custom difference is 5.960e-8. This is separate from the exact same-runtime P9 optimization comparisons above.

The clean Python 3.12 base wheel's default auto detection selects **WebGpuExecutionProvider / D3D12**, with **210 profiled GPU nodes**, `execution_verified=True`, `cpu_fallback=False`, and saved/reloaded prediction maximum difference 0. This proves actual execution, not merely provider availability. Radeon and Apple execution were not measured on this Windows host.

Qualified wheel SHA256: `37851c00f7a8497442344483362fd1c9472b2e1515af3bd67db19cf0cb7a68ea`.

Application source-tree SHA256: `682a8182a45f092eee1d01f8b22ee9d774e6555d4b2a59b4ef07ac0b61e679a8`.

The wheel remains local, version 1.4.6. Distribution JSON, model hashes, dependency versions, installed fingerprints and the final source-archive record are retained in [the distribution evidence](validation/image_export/distribution/). [Qualification reproduction](../benchmarks/image_export/QUALIFICATION.md) describes the isolated image/export probe. The original patch-9 baseline wheel remains unchanged. The UI checkout has no tracked changes; Git whitespace validation passes. The source archive is refreshed after this report so it includes the final documentation and tests.

## Limits

The UI SDK uses a native float32 renderer and a different fitted calibration/LUT implementation. Its file-to-PNG path does not establish the package's strict publication contract, and its outputs are not an analytic accuracy oracle. The comparison measures the same input files and each application's frozen calibration without refitting.

The reader's decoded-byte cap covers decoded source plus canonical float64 RGB; decoder workspaces and stage intermediates remain outside that documented budget. Unusual animation streams retain the conservative behavior. A pre-existing Pillow GIF `seek()` disposal allocation can occur before a post-seek dimension check for malformed later frames; this change does not claim complete allocation preflight for every malformed animation.

Measurements describe this Windows workstation. Radeon and Apple hardware were not available for timing. No remote publication, commit or push was performed.
