# Measured performance

Measured on Windows with an Intel Core i9-14900K, Python 3.12.10 and eight
numerical worker threads on 2026-10-29. These are wall-clock measurements of
actual package calls, with no simultaneous test/build workload. Disk cache and
thermal state can affect timings. Medians use seven repeated warm calls;
initial compilation, preparation and model loading are reported separately.
Existing on-disk JIT caches were available. Python imports are outside these timings.

## Bundled photographs

A fresh calibration used the Classic24 chart in `Images/Image_1.JPG` and the
white-field photograph `Images/white.JPG`, with all
FFC/GC/WB/CC stages enabled, explicit color semantics, modern stage policies,
reproducible degree-1 linear fitting and no regressor LUT. Frame 0 was selected
explicitly. Both photographs are 1152×768. Chart inference actually executed
210 nodes on the WebGPU provider through D3D12; this is execution evidence,
not merely an advertised GPU provider.

The first fit, including detector initialization and durable bundle saving,
took 5.36 s. Two subsequent fresh fits took 2.48 and 2.55 s. Preparing direct
CPU execution for the reloaded full calibration took 0.34 s, including initial
qualification/compilation. Application timing includes strict JPEG decode,
image contracts and direct float64 correction; export timing includes PNG8
quantization, fsync, complete byte readback and decoded-sample verification.
Precision loss from float64 to PNG8 was explicitly consented to.
PNG profiles were warmed and alternated in order across repetitions.

| Photograph | PNG profile | Predict, ms | Strict export, ms | Complete call, ms |
| --- | --- | ---: | ---: | ---: |
| Sample_1.JPG | compatible | 188.08 | 117.67 | 294.16 |
| Sample_1.JPG | fast-lossless | 172.17 | 95.49 | 252.43 |
| Image_1.JPG | compatible | 184.48 | 125.24 | 303.32 |
| Image_1.JPG | fast-lossless | 178.97 | 85.80 | 280.64 |

Complete-call medians are measured directly, so they need not equal the sum of
the individual phase medians. Direct output agreed with the reloaded reference
path within 2.84e-15 maximum RGB difference. Decoded exports exactly matched
the reference uint8 quantization for every pixel. The two PNG profiles produce
the same samples; `fast-lossless` uses uncompressed PNG deflate and larger files.

## Larger input workloads

The following runs use three resized versions of the same photographed input
and one fixed saved GC/WB/CC calibration. These are separate scaling workloads,
not the fresh bundled-photo calibration above. Public direct execution includes
decode/validation and verified PNG8 export. Each row is a seven-call warm median.

| Input | Profile | Predict, ms | Strict export, ms | Complete call, ms |
| --- | --- | ---: | ---: | ---: |
| 512×512 | compatible | 47.36 | 50.26 | 98.39 |
| 1920×1080 | compatible | 325.01 | 249.39 | 574.41 |
| 3840×2160 | compatible | 1129.78 | 802.70 | 1944.99 |
| 512×512 | fast-lossless | 40.55 | 47.65 | 95.35 |
| 1920×1080 | fast-lossless | 301.92 | 185.46 | 497.23 |
| 3840×2160 | fast-lossless | 1094.09 | 585.29 | 1703.70 |

These measurements do not establish independent-capture color accuracy or
performance on other processors/GPUs. Prepared color execution uses the CPU;
chart detection uses its separate GPU provider policy. UI native engines use
different fitting, arithmetic and export guarantees and are not a numerical
accuracy oracle. Taichi remains experimental and is not a runtime dependency
or an automatically selected backend.

To measure another capture, time calibration and `prepare_execution` separately,
warm the prepared handle, then measure `predict_image(path)` followed by
`write_image_data`. Keep input pixels, saved calibration, selected stages,
thread counts, output precision and persistence checks fixed. Record raw samples,
dependency versions, actual backend diagnostics and input/model hashes outside
the repository. Compare decoded output samples, not only elapsed time.

## UI execution qualification (2026-10-30)

On Windows Core i9-14900K, Python 3.12.10, eight numerical threads, common
1152×768 PNG8 inputs, seven alternating paired warm public file-to-PNG calls
per input. Import, calibration, loading, JIT and LUT preparation are excluded.
The table uses explicit `fitting_policy="ui-native-v1"`, full FFC/GC/WB/CC,
matched WebGPU detections and physical XYZ D65 references, and CPU rendering.
Both use the same plane FFC (bins 30/smoothing 7), 9×9 samples and CC LUT33.

| Method | Input | Package strict PNG8, ms | UI API PNG8, ms | Package/UI |
| --- | --- | ---: | ---: | ---: |
| CCM | Image_1 | 195.26 | 85.54 | 2.28× |
| CCM | Sample_1 | 190.37 | 82.99 | 2.29× |
| RPCC3 | Image_1 | 190.06 | 83.37 | 2.28× |
| RPCC3 | Sample_1 | 208.98 | 91.62 | 2.28× |
| NN100 | Image_1 | 164.68 | 70.75 | 2.33× |
| NN100 | Sample_1 | 170.08 | 74.61 | 2.28× |

Package uses explicit `fast-lossless` PNG (level 0/NoFilter) with verified
quantization, immutable identity hashes, fsync, complete readback, sample and
metadata checks, and atomic publication. UI uses its normal fixed Fast/NoFilter
encoder, whose compression level is not exposed. The differing export work is
included; this table does not claim native UI speed parity.

Thirty matched configurations cover CCM, RPCC2/3, polynomial2/3, PLS d1/c3
and d2/c3/c6, and residual NN32/NN100 across CC-only, GC/WB/CC and full stages.
All sixty outputs pass exact per-package safe reload checks. Twenty existing
staged-model float-array golden comparisons also remain exact. Existing staged
calibrations and defaults retain their original equations.

Bounded CPU workers execute nogil image kernels without an OpenMP spin phase.
No global thread environment is changed. Older Numba uses separate helpers to
avoid pathological nested inlining. Exact byte-code decode caching adds no
transfer-function approximation. Unsupported/failed optional execution retains
the reference implementation and strict persistence checks.

The first complete native NN100 image with an empty JIT cache took 12.00s;
the next took 164.13ms. Imports and model loading were excluded.
Optional JIT therefore benefits repeated execution, not every one-shot use.

Identical-observation NN100 training matches all 1,403 native parameters in the
qualified Windows 10,000-epoch control. Independent native full-pipeline fits
differ by 0.026268 RGB8 levels/channel on the scene (maximum 1);
the chart maximum is 2 levels. These are small remaining differences, not a
promise of bit-identical independently fitted networks or independent-capture
accuracy. See [UI-compatible fitting](ui-compatible-fitting.md) for native-mode
requirements, fusion semantics and prepared-execution limits.

Final regression qualification passed 1,450 checks. Wheel/sdist and isolated
Python 3.8/3.12 installations, with and without optional performance kernels,
passed source-equivalence, numerical, safe-reload and distribution checks.
