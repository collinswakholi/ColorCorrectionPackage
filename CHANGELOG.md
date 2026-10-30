# Changelog

## Unreleased (1.4.6)

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
