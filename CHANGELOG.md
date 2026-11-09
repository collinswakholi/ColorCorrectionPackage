# Changelog

## [2.0.0] - 2026-11-09

- Default new calibrations to UI-native RPCC2 with calibration-time image precision selection.
- Expose model, basis, degree and precision directly in `Config`; support polynomial, root and hybrid features through degree four for linear, PLS and NN fits.
- Add immutable step options and categorized configuration requests with explicit grouped JSON; retain flat constructors, mutable fields and saved numerical recipes.
- Retire `ours`/`conv` terminology from the primary API while retaining historical ML/conventional training, effective settings and saved-model evaluation.
- Preserve native byte-image equations and add an explicitly recorded float64 precision runtime.
- Ship pip-installable tutorials, sample resources, migration documentation and v2 numerical/image qualification checks.
- Qualify standard CPython 3.13 and 3.14 with base and Numba installations; expand CI and document the Python 3.15 binary-dependency constraint.
- Centralize runtime sampling rules, keep optional legacy backends lazy, and update packaged resource loading and architecture documentation.

## [1.5.0] - 2026-10-30

- Clarify encoded-RGB fit objectives and out-of-gamut targets; use the existing linear-light CCM workflow in the sample example.
- Transactional calibration state, structured failures and verified atomic persistence.
- Reject Windows drive-relative and alternate-stream calibration names before decoding or persistence, preserving POSIX colon filenames.
- Model-based chart detection with GPU provider diagnostics and CPU recovery.
- Explicit image/reference/observation and color contracts, opt-in modern numerical policies and reproducible grouped fitting.
- Portable calibration bundles, bounded batch processing and precision-preserving image export.
- Qualified direct CPU execution, optional Numba acceleration and an explicit fast-lossless PNG profile.
- Consolidated current documentation and isolated distribution qualification.
- Added opt-in UI-compatible CCM, RPCC2/RPCC3, polynomial, PLS and residual NN fitting, with ordered float64 NN training arithmetic.
- Added `ui-native-v1` for recorded uint8 sRGB decoding, ordered sampling, reference conversion and joint FFC/GC; older calibration semantics remain unchanged.
- Added bounded CPU image workers and exact byte-code decode caching, preserving stage precision and strict persistence. Native-mode and optional-kernel smoke checks run in installed-wheel qualification.
- Removed transient compiler caches and legacy debug traces; release measurements retain their numerical scope and cold-start limits.

## [1.4.6] - 2026-05-08

- Coerced UI string configuration values before sklearn NN fitting.
- Accepted scalar, JSON and comma-separated hidden-layer settings.
- Hardened sample-count, random-state and boolean CC options from UI payloads.

## [1.4.5] - 2026-05-07

### Changed

- Replaced runtime `ultralytics` flat-field plane detection with OpenCV DNN inference using the bundled ONNX model.
- Updated default neural-network color-correction hidden layers to `[64]`.
- Made color-correction 3-D LUT construction lazy and configurable.

### Added

- Added flat-field multiplier caching for repeated white-image/configuration pairs.
- Added unit coverage for ONNX detector parsing, crop handling, lazy LUT behavior, custom NN batch-norm safety, and FFC cache reuse.
- Added benchmark/repro scripts for comparing FFC and NN color-correction configurations on the bundled real images.

### Fixed

- Hardened custom PyTorch NN batch normalization for small training splits.
- Prevented custom PyTorch NN CUDA out-of-memory fallback from retrying forever at `batch_size=1`.
- Avoided duplicate color-chart detection when training CC with `n_samples > 1`.

## [1.4.4] - 2026-05-07

### Changed

- Added the associated Plant Phenome Journal manuscript citation to the README.
- Hardened fit-time/predict-time parity, model persistence, and config validation.
- Improved finite white-balance factors and CPU/GPU-safe custom NN training paths.

## [1.4.3]

- Unified neural-network configuration around `hidden_layers`, including compatibility with the earlier `nlayers` key.
- Updated sklearn MLP layer handling and pipeline forwarding of hidden-layer settings.

## [1.4.2]

- Removed the conflicting `opencv-python` dependency in favor of `opencv-contrib-python`.
- Added the MCC availability flag and version-compatible MCC factories.
- Added post-install module verification.

## [1.4.1]

- Promoted Numba to a core dependency for that release, enabling acceleration by default.

## [1.4.0]

- Added Numba/CUDA image kernels, precomputed color-conversion tables and 3-D color LUT interpolation.
- Added chart-detection caching, parallel batch prediction and floating-point FFC application.
- Fixed custom-NN memory-retry handling and uint8 clipping in FFC.
- Historical 1.4.0 measurements reported color-conversion kernel speedups of 22–54×, chart-cache hits up to 350×, and average end-to-end improvement of 1.41× over 1.3.4 (up to 1.50×). The original LUT note recorded MAE below 0.0001 on its qualification inputs.
