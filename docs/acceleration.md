# Prepared execution (local patch 9)

Existing prediction methods retain their numerical/default policies. Repeated
application can opt into an owned, reusable execution snapshot:

```python
from ColorCorrectionPipeline import ColorCorrection, LutCache

cc = ColorCorrection().load_bundle("calibration.ccpb")
cache = LutCache(max_bytes=64 * 1024 * 1024, max_entries=16)
with cc.prepare_execution({"mode": "auto", "backend": "auto"}, cache=cache) as execution:
    print(execution.preparation)  # cold preparation, chosen grid, errors, cache hit
    result = execution.predict_image("input.tiff", stages="final")
    final_rgb = result.images["CC"]
    print(result.to_dict())       # descriptors/diagnostics, no image pixels
```

## Numerical policy and fallback

`mode="direct"` uses the recorded stages without interpolation. Known portable
affine polynomial models (linear and PLS) have an explicit NumPy adapter and an
optional fused float64 Numba adapter, with fastmath disabled. Other known model
families retain their recorded inference adapter. Unsupported explicit backend
requests fail when `allow_fallback=False`; automatic selection reports the actual
backend, output precision, and any fallback. Install the optional accelerator
with `pip install "ColorCorrectionPipeline[performance]"`. The base wheel runs
without Numba. No Taichi backend is enabled by this patch.
Saved legacy polynomial GC retains its existing runtime-dependent NumPy/Numba
Lab conversion and Torch CUDA polynomial helper, when available. An explicit
CPU backend controls the new CC/LUT kernels; it does not redefine those legacy
stage helpers. Prepared modern neutral-PCHIP GC can use the qualified float64 Numba
implementation of the recorded equations. The reference NumPy route remains
available. Eligible tone + positive diagonal WB + portable/reproducible affine
CC models use one owned working image and in-place CC. Model parameters are
derived per image rather than per chunk; no full polynomial feature image is allocated. Requested
intermediates receive independent copies. Other model families and arbitrary WB
matrices retain chunked execution; `chunk_pixels` still bounds their workspace.
The bulk path uses constant scalar kernel workspace beyond source, output and
requested intermediates, so it does not allocate a chunk-sized feature expansion.
See [direct qualification](DIRECT_COLOR_OPTIMIZATION.md) and the latest
[bulk optimization evidence](BULK_DIRECT_OPTIMIZATION.md).

`mode="auto"` attempts a combined GC→WB→CC trilinear LUT at grids 17, 33, 65 and
129, in that order. The default maximum sampled RGB error is 0.002 in the declared
output encoding and maximum sampled CIEDE2000 is 0.5 (D65). Qualification uses
4,096 seeded random colors plus neutral/edge probes. A separate seed and edge
spacing audit the first qualifying candidate. Audit failure selects direct
execution without tuning further on the audit points. These are sampled checks,
not a global error guarantee or physical calibration accuracy claim. Boundary
clipping, including clipping between stages, is part of the reference function.
The direct backend is also checked against the recorded reference on both sets.
An additional post-selection stress audit uses misaligned 67³ cell centers,
complete near-black and near-white byte cubes (17³ each), and 262,144 independently
seeded byte-lattice RGB colors. Its default 572,733 points target narrow clipping
transitions that random colors, edges and centers alone can miss. Smaller
`validation_samples`/`stress_grid_size` settings reduce this dose and are recorded.
Stress failure also selects direct execution without retuning on the audit data.

`mode="combined-lut"` requests the same qualification. When no candidate passes,
the default falls back honestly to direct execution. Add `allow_fallback=False`
to require a qualified LUT or raise `AccuracyBudgetError`. Budgets, candidate
grids, sample count/seed, and chunk size can be supplied through `ExecutionOptions`
or a dictionary. Increasing grids costs preparation time and memory. Report cold
preparation/JIT separately from warm application; a LUT may lose on short jobs.
Automatic selection qualifies interpolation error; it does not benchmark and
choose the fastest implementation. Affine CC can favor direct execution. In the
local photo calibrations even 129³ failed the default RGB budget; the reference
fallback is intentional. An early, weaker audit accepted a full-pipeline preview
at `max_rgb_error=1/255`, `max_delta_e00=0.5`, while dense image tests found RGB
error 0.00535 and Delta-E 0.848 near black. That counterexample is retained as a
regression fixture. The strengthened audit rejects it. Explicitly relaxed budgets
are measured separately; even the stronger sampled audit is not an all-pixel or
all-continuous-color guarantee.

Spatial FFC always runs separately before the pointwise function. Image decoding,
orientation and declared linear/sRGB boundary conversions stay outside the table.
Combined mode returns only the final image; prepare `mode="direct"` to request
intermediate stages. Requested intermediate stages stop execution at the latest
requested stage. Input range/shape/finite checks remain active.

## Ownership, cache, and export

Each prepared handle owns a validated portable calibration snapshot. Later owner
mutation does not change it. The shared immutable LUT cache is bounded by bytes
and entries and keyed by complete pointwise model content, color/white semantics,
runtime version and execution options. Different spatial FFC maps can share a
pointwise table. Live handles retain their table even after cache eviction, so
close unused handles. `close()` waits for active calls and releases their models.
There is no background worker or cache file. Use `cache.clear()` to evict shared
entries; `cache.info` reports retained bytes, builds and hits.

Legacy per-regressor LUT construction is synchronized and invalidates on learned
parameter/settings changes. Its one-table cache is immutable and excluded from
lock/callback serialization. Saving after a producer mutation omits a stale
derived table instead of blessing it on load. Pre-patch artifacts cannot prove
their original table producer; exact old table values are adopted for compatibility
and current producer content is pinned for subsequent invalidation.

`execution.export_cube("correction.cube")` writes a qualified pointwise table
atomically, checks a streamed SHA-256 readback, and uses red-fastest row ordering.
A cube cannot represent spatial FFC: explicitly pass `exclude_ffc=True` when it is
present. For declared linear input/output, explicitly pass `working_space=True`;
that cube still accepts and returns encoded sRGB and requires boundary conversion
by the consumer. Export preserves existing files on pre-publication failure.

Chart inference GPU selection remains independent of pointwise execution. Its
existing CUDA/DirectML/WebGPU/CoreML selection and CPU fallback remain in place.
This patch promotes only measured CPU kernels; no unqualified GPU/Taichi color
kernel is silently selected. See [the benchmark report](VALIDATION_9.md) for actual
host/backend timings and [distribution qualification](distribution-qualification.md)
for installed-wheel checks and the unrun platform matrix.
