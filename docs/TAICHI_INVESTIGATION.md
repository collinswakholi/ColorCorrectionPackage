# Taichi CPU/Vulkan investigation: actual local timings

Measured 28 October 2026 on Intel Core i9-14900K (32 logical processors visible), NVIDIA RTX 4070 Ti SUPER and Intel UHD Graphics 770, Windows x64. NVIDIA driver 610.88. Taichi 1.7.4, Python 3.12.10, NumPy 2.4.6 and Numba 0.65.1. This is an experimental benchmark, with **no production Taichi dispatch or dependency added**.

The useful result is selective: a fused CPU FFC or direct polynomial evaluator can help, but replacing the existing Numba LUT/color-conversion functions indiscriminately is not beneficial. GPU launch, transfer and float64 boundary conversion costs must be included. The current chart neural network stays in ONNX Runtime.

## Measurement contract

- Frozen source from the validated patches 3–4 package supplied the original functions. The original acceleration module and direct-prediction function hashes are in every result file. Experimental Taichi kernels preserve the mathematical algorithms for the tested operations, subject to the explicitly reported precision.
- Three image sizes: 512×512, 1920×1080 and 3840×2160. Each has a resized real ColorCard_Big.JPG RGB array and independent seeded random RGB input. Intel Vulkan covers the first two sizes. These are per-pixel compute tests, not chart-detection accuracy tests; resizing is outside timing.
- 11 measured repetitions follow 3 warmups. Original/Taichi calls alternate in seeded randomized order within each run. CPU original Numba and Taichi use 32 threads. Raw samples, median and p95 are retained. No other team member ran compute tests during this window. Clocks/power state were not locked; these are local observations, not universal hardware guarantees.
- Original functions receive float64. Taichi CPU float64 is the precision-preserving comparison. Separate Taichi CPU float32 controls use the **same kernels and precision** as Vulkan float32; all float32 errors are measured against the original float64 reference. Float32 is not silently substituted for the existing package contract.
- GPU **host-to-host**: input is converted from float64 where necessary, uploaded each call, processed, synchronized, downloaded and converted back to float64. The LUT/coefficients/white-balance matrix/FFC gainmap are retained on the GPU. Persistent allocation/upload is measured separately. This matches reuse of a calibrated model; first-use cost is higher.
- GPU **resident**: a synchronized launch plus kernel execution on already uploaded inputs/parameters, without output download or host conversion. It is a lower bound for a future resident pipeline, not current pipeline latency.
- LUT inference uses the same 33³ sampled fitted degree-3 model. Direct regression explicitly disables LUT and includes the original polynomial feature generation, sklearn prediction and existing garbage collection. Most deployed large-image predictions already use the LUT; a direct-regression speedup is not a speedup of that default path.
- WB times only the existing image@diagonal-matrix application expression, excluding chart detection/fitting and any other stages. FFC uses a matching spatial gainmap and the original legacy D65 Lab constants; it does not qualify the separate explicit-v1 color engine or estimate a gainmap.

## Measured median milliseconds: real-photo input

Original is the CPU-f64 reference pass below. Other runs repeat their own paired original baseline; exact paired speedups and timing variation are retained in JSON. Rows labeled f32 change arithmetic precision and must not be read as drop-in float64 replacements.

| Size | Operation | Original f64 | Taichi CPU f64 | Taichi NVIDIA Vulkan | GPU precision |
| --- | --- | ---: | ---: | ---: | --- |
| 512×512 | 33³ LUT correction inference | 1.844 | 2.671 | 9.567 | f64 |
| 512×512 | sRGB → Lab D65 | 6.027 | 8.285 | 12.905 | f32 |
| 512×512 | Lab-luminance FFC apply | 16.433 | 10.660 | 17.039 | f32 |
| 512×512 | WB matrix apply only | 2.249 | 1.676 | 5.545 | f64 |
| 512×512 | Direct degree-3 regression inference | 257.891 | 2.148 | 12.207 | f64 |
| 1920×1080 | 33³ LUT correction inference | 6.993 | 8.554 | 121.797 | f64 |
| 1920×1080 | sRGB → Lab D65 | 43.440 | 52.451 | 81.612 | f32 |
| 1920×1080 | Lab-luminance FFC apply | 112.870 | 71.641 | 68.224 | f32 |
| 1920×1080 | WB matrix apply only | 18.526 | 3.195 | 92.719 | f64 |
| 1920×1080 | Direct degree-3 regression inference | 727.522 | 5.383 | 94.590 | f64 |
| 3840×2160 | 33³ LUT correction inference | 21.519 | 28.117 | 272.951 | f64 |
| 3840×2160 | sRGB → Lab D65 | 114.974 | 151.312 | 289.887 | f32 |
| 3840×2160 | Lab-luminance FFC apply | 392.132 | 219.843 | 266.956 | f32 |
| 3840×2160 | WB matrix apply only | 77.436 | 11.167 | 232.043 | f64 |
| 3840×2160 | Direct degree-3 regression inference | 2122.775 | 18.477 | 366.041 | f64 |

## Same-precision float32 controls: real-photo input

All entries include host-to-host conversion back to the package float64 interface. This comparison isolates CPU versus Vulkan using the same experimental float32 kernels, while the previous table retains the actual original float64 baseline.

| Size | Operation | Taichi CPU f32 | NVIDIA Vulkan f32 | Intel Vulkan f32 |
| --- | --- | ---: | ---: | ---: |
| 512×512 | 33³ LUT correction inference | 10.682 | 12.764 | 13.700 |
| 512×512 | sRGB → Lab D65 | 11.561 | 12.905 | 13.242 |
| 512×512 | Lab-luminance FFC apply | 17.281 | 17.039 | 13.503 |
| 512×512 | WB matrix apply only | 5.218 | 10.074 | 6.553 |
| 512×512 | Direct degree-3 regression inference | 5.210 | 10.452 | 7.151 |
| 1920×1080 | 33³ LUT correction inference | 63.116 | 83.704 | 68.253 |
| 1920×1080 | sRGB → Lab D65 | 66.756 | 81.612 | 67.766 |
| 1920×1080 | Lab-luminance FFC apply | 73.406 | 68.224 | 58.369 |
| 1920×1080 | WB matrix apply only | 27.295 | 53.259 | 40.477 |
| 1920×1080 | Direct degree-3 regression inference | 27.093 | 53.225 | 38.891 |
| 3840×2160 | 33³ LUT correction inference | 160.487 | 289.426 | — |
| 3840×2160 | sRGB → Lab D65 | 199.817 | 289.887 | — |
| 3840×2160 | Lab-luminance FFC apply | 217.049 | 266.956 | — |
| 3840×2160 | WB matrix apply only | 86.829 | 259.875 | — |
| 3840×2160 | Direct degree-3 regression inference | 100.316 | 258.113 | — |

## GPU latency distribution and residency: 1920×1080 photo

| Route | Operation | Host median ms | Host p95 ms | Resident median ms | Resident p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| nvidia-vulkan-f64 | 33³ LUT correction inference | 121.797 | 130.966 | 7.387 | 9.786 |
| nvidia-vulkan-f64 | WB matrix apply only | 92.719 | 96.004 | 5.995 | 6.899 |
| nvidia-vulkan-f64 | Direct degree-3 regression inference | 94.590 | 98.857 | 8.310 | 8.822 |
| nvidia-vulkan-f32 | 33³ LUT correction inference | 83.704 | 86.497 | 4.079 | 6.621 |
| nvidia-vulkan-f32 | sRGB → Lab D65 | 81.612 | 86.189 | 4.239 | 4.298 |
| nvidia-vulkan-f32 | Lab-luminance FFC apply | 68.224 | 78.843 | 5.018 | 6.281 |
| nvidia-vulkan-f32 | WB matrix apply only | 53.259 | 55.040 | 3.816 | 5.092 |
| nvidia-vulkan-f32 | Direct degree-3 regression inference | 53.225 | 54.897 | 3.871 | 4.046 |
| intel-vulkan-f32 | 33³ LUT correction inference | 68.253 | 71.658 | 4.513 | 4.914 |
| intel-vulkan-f32 | sRGB → Lab D65 | 67.766 | 69.494 | 2.695 | 3.085 |
| intel-vulkan-f32 | Lab-luminance FFC apply | 58.369 | 68.723 | 3.096 | 3.320 |
| intel-vulkan-f32 | WB matrix apply only | 40.477 | 42.427 | 2.136 | 2.855 |
| intel-vulkan-f32 | Direct degree-3 regression inference | 38.891 | 40.422 | 2.279 | 3.222 |

## Numerical differences

Maximum absolute difference across both datasets and all tested sizes, against the unchanged original float64 function. Lab errors are in Lab units; the other rows use normalized RGB channel units. These numbers measure numerical agreement on these inputs, not independently measured physical color accuracy.

| Route | LUT RGB | Lab | FFC RGB | WB RGB | Direct poly3 RGB |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu-f64 | 4.44089e-16 | 1.66978e-13 | 1.71807e-14 | 0 | 2.22045e-16 |
| cpu-f32 | 2.00694e-07 | 7.49747e-05 | 7.44843e-06 | 1.32321e-07 | 1.08336e-07 |
| nvidia-vulkan-f32 | 1.77709e-07 | 0.000111125 | 1.19265e-05 | 1.32321e-07 | 1.08189e-07 |
| nvidia-vulkan-f64 | 3.33067e-16 | — | — | 0 | 0 |
| intel-vulkan-f32 | 1.24811e-07 | 0.000103467 | 9.44962e-06 | 1.32269e-07 | 1.06966e-07 |

### Actual float64 Vulkan boundary

A strict float64 non-integer power probe failed compilation in Taichi 1.7.4 Vulkan: `[glsl450] the operand type of instruction Pow(26) must <=32bits`. The retained failure log is an expected qualification failure, not a silent CPU recovery. Consequently Lab/FFC Vulkan results use a clearly declared float32 route. NVIDIA float64 arithmetic-only LUT/WB/polynomial kernels did execute. Metal does not provide Taichi float64 according to the [official type table](https://docs.taichi-lang.org/docs/type); broad scientific float64 parity therefore cannot be assumed across vendors/backends.

## Cold costs

Offline Taichi cache was disabled. First call includes JIT and its normal call work, after persistent model allocation. Imports and backend initialization are separate. Imports include the complete reference package/scientific stack and cannot be attributed entirely to Taichi. These are first calls in the scripted operation sequence, not separate pristine OS processes per operation. Original Numba also records its first-call time, with a fresh cache for the first measured CPU pass; later processes may use that cache.

| Route | Imports ms | Backend init ms | LUT first ms | Lab first ms | FFC first ms | WB first ms | Poly3 first ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu-f64 | 8663.993 | 217.411 | 1001.651 | 156.294 | 258.603 | 123.601 | 446.482 |
| cpu-f32 | 10047.045 | 219.232 | 1025.255 | 173.044 | 216.754 | 125.370 | 452.505 |
| nvidia-vulkan-f32 | 10186.375 | 876.437 | 125.362 | 130.572 | 170.195 | 88.841 | 394.358 |
| nvidia-vulkan-f64 | 10963.044 | 721.870 | 166.386 | — | — | 95.507 | 420.071 |
| intel-vulkan-f32 | 8702.901 | 635.242 | 169.413 | 144.934 | 166.803 | 99.641 | 420.435 |

## Original chart neural-network reference

The unchanged ONNX detector was also timed on the original-resolution photo. This includes letterbox preprocessing, native ONNX forward, decoding and chart assembly, excluding disk decoding. It is **not Taichi neural-network inference**. CPU uses 4 ORT threads; strict Vulkan uses the existing native WebGPU provider, with CPU fallback disabled.

| Existing detector route | First incl. session init ms | Warm median ms | Warm p95 ms | Detected faces |
| --- | ---: | ---: | ---: | ---: |
| cpu | 814.548 | 494.196 | 520.245 | 1 |
| vulkan | 1768.396 | 183.955 | 200.089 | 1 |

## Engineering decision

1. Keep the current ONNX Runtime provider architecture for the chart network. Taichi is a kernel DSL, not an ONNX graph importer. [Official external-array documentation](https://docs.taichi-lang.org/docs/external) also makes clear that decorating existing NumPy calls does not compile them. Reimplementing the network would require operator coverage, weight conversion and independent inference qualification.
2. Keep current Numba LUT/color conversions by default. CPU f64 timings show where Taichi loses as well as wins. A faster fused direct-regression evaluator is promising, but first compare a fused Numba equivalent and remove unnecessary allocation/collection costs. That is an algorithm/dataflow improvement, not evidence that every Taichi backend is faster.
3. If adding Taichi later, make it an optional explicitly selected application backend with declared precision. Keep calibration/reference truth/metrics in the existing float64 contract. Never silently change legacy or explicit semantics to make a GPU kernel compile.
4. A resident fused application chain is the main GPU opportunity: reuse calibrated gainmap/model arrays, combine compatible per-pixel stages, upload once and download once. The resident figures quantify only one kernel; no full resident multi-stage or neural-network pipeline speedup has been measured here.
5. Production work would also need lazy runtime ownership, backend/device-stable cache keys, transactional output/failure fallback, memory budget limits and batch scheduling. Taichi initialization/backend selection is process-wide; changing it while arrays are live requires careful ownership. Do not add global initialization at package import.
6. Hardware qualification here covers Windows x64 CPU, NVIDIA Vulkan and Intel Vulkan. Radeon, Apple Silicon/Metal, Linux/ARM and Taichi CUDA were not physically measured. The [official backend matrix](https://docs.taichi-lang.org/docs/hello_world) describes available platform routes, not a guarantee of identical precision or performance.

## Evidence and reproduction

All 128 matrix cases executed with finite outputs and no unexpected runtime errors. There are 30 CPU-f64, 30 CPU-f32, 30 NVIDIA-Vulkan-f32, 18 NVIDIA-Vulkan-f64 arithmetic cases and 20 Intel-Vulkan-f32 cases. The expected float64-power failure is retained separately. No skip has been relabeled as a successful GPU measurement.

Each JSON contains all individual times, p95, cold costs, dataset/source hashes, precision and explicit adapter selection. Native Vulkan enumeration identified Intel index 0 and NVIDIA index 1; `TI_VISIBLE_DEVICE` was set accordingly. `actual_arch` is Vulkan and `enable_fallback=False` was enforced. These are genuine GPU kernel executions, not a CPU fallback labeled GPU.

The scripts were run using an isolated Taichi target package directory with the previous audit Python 3.12 scientific dependencies; package environments and production dependencies were unchanged. Use a fresh process for every backend/dtype, set `TI_VISIBLE_DEVICE`, and run `bench_taichi.py --backend cpu|vulkan --dtype float64|float32 --repeats 11 --output <file>`. Select `--operations lut,wb,poly3` for supported f64 Vulkan arithmetic. `--repo` and `--photo` allow another local checkout/corpus. `vulkan_devices.py` inventories adapters; `bench_onnx_reference.py` performs the separate unchanged-detector reference measurement.

The exact scripts used for timing are retained as `bench_taichi_measured.py` and `bench_onnx_reference_measured.py`, with hashes in the machine summary. Only repository/photo path arguments were added to the runnable copies after measurement; kernel function definitions remain byte-identical. The frozen reference Python source, its artifact provenance and instructions are under [benchmarks/taichi](../benchmarks/taichi/README.md).

- [cpu-f64 raw measurements](validation/p56/taichi/cpu-f64.json)
- [cpu-f32 raw measurements](validation/p56/taichi/cpu-f32.json)
- [nvidia-vulkan-f32 raw measurements](validation/p56/taichi/nvidia-vulkan-f32.json)
- [nvidia-vulkan-f64 raw measurements](validation/p56/taichi/nvidia-vulkan-f64.json)
- [intel-vulkan-f32 raw measurements](validation/p56/taichi/intel-vulkan-f32.json)
- [Compact machine summary](validation/p56/taichi/summary.json)
- [Vulkan adapter inventory](validation/p56/taichi/vulkan-devices.json)
- [Expected float64-power compile rejection](validation/p56/taichi/fp64-power-failure.log)
- [Original ONNX CPU reference](validation/p56/taichi/onnx-cpu.json)
- [Original ONNX Vulkan reference](validation/p56/taichi/onnx-vulkan.json)
