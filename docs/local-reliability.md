# State, persistence and compatibility

`ColorCorrection.run` returns `(metrics, images, error)`. It fits an isolated
candidate and commits the complete model/reference context only after required
stages and requested persistence succeed. Failed fits preserve the previous
successful calibration. A successful all-disabled fit installs an identity
calibration. Prediction takes one consistent state snapshot. Standalone `do_*`
methods remain low-level mutating operations.

`last_run` records success, stage status/timing, errors, detector diagnostics and
artifacts. Required-stage failure stops processing; later stages stay `not_run`.
Invalid configuration, input and filesystem errors raise after recording failure.
Stage errors retain the legacy `error=True` return. Enabled FFC requires a
readable white image: explicitly disable it when that image is unavailable.
Calibration names must be filename stems. Windows drive-relative and NTFS
alternate-stream qualifiers are rejected before decoding or persistence;
POSIX filenames containing a colon remain supported.
GC, WB and CC require a real chart during `run`; standalone legacy helpers retain
their historical identity fallback unless `require_chart=True` is requested.
FFC can succeed without chart metrics, which are reported as unavailable.

Configuration owns supplied dictionaries and illuminants. `to_dict` returns an
independent snapshot and `from_dict` does not mutate its argument. The FFC cache
identity covers all white-image pixels, relevant configuration and automatic
plane detector bytes, preventing stale reuse when weights change at the same path.

## Durable output

Prefer validated non-executable [`.ccpb` bundles](calibration-bundles.md).
Legacy `.pkl` loading remains available for trusted files only: validation after
unpickling cannot prevent arbitrary code execution. Failed container loads
preserve active state. Saved explicit metadata carries color, reference,
observation, policy and detector context; older metadata-free models retain
legacy interpretation.

Model saves serialize beside the destination, flush/fsync, validate a readback
and atomically replace the destination. Derived LUT callbacks are reconstructed
after loading, with fitted parameters and valid cached values retained.
Run exports are prepared before publication. Ordinary multi-file publication
failures restore replaced files from backups. If restoration fails, `last_run`
identifies retained recovery files. Each replacement is atomic; a CSV/model
collection is not crash-atomic across power loss or process termination.
Strict [image exports](image-io-policies.md) verify encoded bytes, metadata and
decoded samples before publication and honor no-clobber/overwrite policy.

## Numerical compatibility

Legacy ndarray callers retain encoded-sRGB stage equations, including the
historical saturation scaling and WB headroom. [Modern policies](numerical-policies-and-validation.md)
and [explicit semantics](color-semantics.md) require opt-in. Use `ImageData` to
declare precision, channel order and encoding; floating raw RGB must be finite,
nonempty HxWx3 and in `[0,1]`. Legacy uint8 white arrays retain their BGR convention.

The default Classic24 ONNX detector supplies geometry from original pixels,
independently of optional metrics, and reuses it within a run. Transformed images
resample those canonical cells. Its measurements differ from OpenCV MCC; existing
saved coefficients still apply without chart detection. See [detector settings](chart-detection.md),
[GPU selection and recovery](gpu-detection.md) and the packaged model provenance.
