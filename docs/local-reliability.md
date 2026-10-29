# Local reliability changes and compatibility

`ColorCorrection.run` still returns `(metrics, images, error)`. It now fits a
separate candidate and commits all active models/reference context only after
the required stages and requested persistence succeed. A failed fit preserves
the previous successful calibration. A successful all-disabled fit replaces it
with an identity calibration. Prediction takes one consistent active snapshot.
Direct `do_*` stage calls remain low-level mutating operations.

Inspect `pipeline.last_run` for `success`, per-stage `status`/timings, structured
`errors`, detector diagnostics, and saved `artifacts`. Processing fails at the
first failed required stage; later stages remain `not_run`. Invalid configuration,
invalid input and filesystem errors raise after recording failure. Stage errors
retain the legacy `error=True` return. Calibration and input arrays are not
published on a failed candidate run.

Enabled stages are required. In particular, `do_ffc=True` with no readable white
image now fails instead of silently changing the configuration. Set
`Config(do_ffc=False)` when no white image is available. GC, WB and conventional
CC now require a real chart when invoked by `run`; their standalone helpers keep
the historical identity fallback unless `require_chart=True` is requested.
Unknown correction/regression methods no longer appear to succeed.
FFC itself does not require a chart: unavailable optional chart metrics are
recorded as warnings while a valid flat-field calibration can still succeed.

Configuration owns copies of supplied dictionaries/illuminants, `to_dict`
returns an independent snapshot, and `from_dict` leaves its argument unchanged.
FFC cache identity covers every white-image pixel and, when automatic plane
selection is used, the detector's bytes. Changed inputs or replaced weights at
the same path cannot reuse a stale gain map through a thumbnail-only cache key.
Floating RGB input must be finite, nonempty HxWx3 and in `[0,1]`. White arrays
retain the documented uint8-BGR / floating-RGB conventions. Integer main-image
arrays should be read/normalized using `read_image` first.

## Persistence

The existing `.pkl` API remains available. **Only load trusted pickle files**;
container validation after unpickling does not prevent arbitrary code execution.
This patch does not introduce the future portable non-executable inference bundle.

Model saves serialize to a temporary file beside the destination, flush/fsync,
validate a read-back, and atomically replace the destination. Derived LUT
callbacks are reconstructed after loading; fitted estimator parameters and
already built LUTs are retained. Older trusted model states receive missing LUT
defaults. Failed saves preserve the existing model file; failed container loads
preserve the active model set.

Run exports are prepared before destinations change. Ordinary publication
failures restore replaced files from temporary backups. If restoration itself
fails, `last_run.recovery` identifies retained files for recovery. Each replacement
is atomic; the collection of CSV/model files is not crash-atomic across a process
termination or power loss. The durable single-bundle design remains later work.

## Detector transition

The default detector is now the bundled Opt2 1280 ONNX chart model, executed with
ONNX Runtime using GPU-first selection and CPU fallback. See [GPU detection](gpu-detection.md).
It covers both neutral/conventional extraction and extended
CC sampling. Geometry is chosen from original source pixels independently of
optional metrics. Model geometry is reused within one run; transformed images resample
the same canonical cells. See [chart-detection.md](chart-detection.md) for settings,
explicit MCC compatibility, fallback policy and supported layouts.

The detector and sampling algorithm change measured patch values. Existing
calibrations still apply through their saved coefficients; reliability fixes do
not intentionally change the legacy GC/WB/CC transform mathematics. The baseline
fixtures freeze those detector-independent operations, including the historical
0.95 WB headroom and saturation behavior. Their scientific redesign is deferred.

The bundled chart model is copied unchanged from the inspected local UI source:
`models/opt2/best_assemble_fp32_1280.onnx`, upstream comparison revision
`16c6f42176cbd392c167ecabea4bb2519984fda8`. Its SHA-256 is
`1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240`, size
39,354,948 bytes. Its declared model terms are AGPL-3.0; the original notices and
license text are retained beside the model. Package MIT terms do not replace
model terms. Existing provenance records and its limits are preserved; no learned
weights, operators or model metadata were modified for this transition.

All changes in this tranche are local. Test results and remaining limitations
are recorded before any later patch or remote push is considered.
