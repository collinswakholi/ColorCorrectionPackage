# Experimental Taichi benchmarks

These scripts are isolated research code. The application never imports them and
Taichi was not added to production dependencies. The report is
`docs/TAICHI_INVESTIGATION.md`; raw measurements are under
`docs/validation/p56/taichi/`.

The measured runtime was Python 3.12.10, Taichi 1.7.4, NumPy 2.4.6, Numba 0.65.1
and PyTorch 2.13.0+cpu. The original direct regression path therefore used CPU
feature expansion. An unrelated exploratory Python 3.11 environment was not used
for the reported measurements.

`baseline_pkg` is a Python-source-only snapshot of the validated local patches
0–4 wheel (28 files, about 479 KB); it preserves the old direct regression behavior
while the working clone evolves. The original license applies. No detector/model
assets are duplicated. `baseline-provenance.json` records every source hash and
the source wheel identity. `original_accel.py` is a byte-identical snapshot of its
acceleration module. The unused plane-model warning from this source-only import
does not affect any benchmarked function.

`bench_taichi_measured.py` and `bench_onnx_reference_measured.py` are the exact
scripts used for timing. The runnable copies add `--repo` and `--photo` path
arguments only. Source-function equivalence and separate script hashes are in
the provenance file. Numerical code was not changed after measurements.

Use an independent Python environment with the project's dependencies and
`taichi==1.7.4`. Match the recorded numerical package versions for close
reproduction. Run from the repository root, one process at a time, with other
compute tasks stopped. For a CPU original reference, disable Torch CUDA device
visibility; this does not disable Vulkan:

```powershell
$benchPython = 'C:\path\to\isolated-environment\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = ''
$env:NUMBA_CACHE_DIR = 'C:\path\to\audit\numba-cache'
$env:MPLCONFIGDIR = 'C:\path\to\audit\matplotlib-cache'
$env:MPLBACKEND = 'Agg'

& $benchPython benchmarks/taichi/vulkan_devices.py
# Select the desired native Vulkan index from that inventory.
$env:TI_VISIBLE_DEVICE = '1'

& $benchPython benchmarks/taichi/bench_taichi.py --backend cpu --dtype float64 `
  --photo 'C:\path\to\ColorCard_Big.JPG' --output cpu-f64.json
& $benchPython benchmarks/taichi/bench_taichi.py --backend cpu --dtype float32 `
  --photo 'C:\path\to\ColorCard_Big.JPG' --output cpu-f32.json
& $benchPython benchmarks/taichi/bench_taichi.py --backend vulkan --dtype float32 `
  --photo 'C:\path\to\ColorCard_Big.JPG' --output vulkan-f32.json
& $benchPython benchmarks/taichi/bench_taichi.py --backend vulkan --dtype float64 `
  --operations lut,wb,poly3 --photo 'C:\path\to\ColorCard_Big.JPG' `
  --output vulkan-f64.json
```

Repeat the float32 GPU command in a fresh process with the other adapter index.
`--sizes 512x512,1920x1080` reproduces the Intel subset. Defaults are 3 warmups and
11 measurements for photo and seeded-random inputs at 512², 1080p and 4K.
`--smoke --repeats 3` runs small numerical smoke comparisons. Results include
individual samples, p95, first-use/JIT time, conversion/upload/download costs,
resident timings, input hashes and numerical differences from original float64.
Use `--repo` if the clone cannot be inferred from the script location.

`probe.py` is the expected float64-power Vulkan compile rejection. It intentionally
exits nonzero on the tested runtime; retain its stderr when checking another
runtime. It does not silently switch to CPU.

The original chart network reference is separate:

```powershell
& $benchPython benchmarks/taichi/bench_onnx_reference.py --device cpu `
  --photo 'C:\path\to\ColorCard_Big.JPG' --output onnx-cpu.json
& $benchPython benchmarks/taichi/bench_onnx_reference.py --device vulkan `
  --photo 'C:\path\to\ColorCard_Big.JPG' --output onnx-vulkan.json
```

Those two commands use the existing ONNX Runtime providers and the pinned model
in the clone. They are not Taichi neural-network timings. Neither per-kernel GPU
resident timings nor direct regression with LUT disabled should be presented as
a measured full-pipeline speedup.

`vulkan_devices.py` currently uses Windows Vulkan loader APIs. The Taichi kernels
are portable experiments, but native Linux/ARM, Radeon, Apple Silicon/Metal and
Taichi CUDA hardware were not measured here.
