# Changelog

## [3.0.0] - Unreleased

3.0 keeps only the 2.x default fitting recipe (UI-compatible, native-v1 or
precision-f64-v1 runtime). Calibrations saved by 2.x and earlier still load and
apply unchanged; the 2.x default fit gives the same results. See
`docs/migration-3.md`.

### Removed

- Fitting with `fitting_policy="legacy-v1"`, `"reproducible-v1"` and
  `"ui-compatible-v1"` with the `staged-v1` runtime, and with `stage_policy` or
  `color_semantics` `legacy-v1`. These now raise a `ValueError` naming the
  replacement; models they saved still load.
- The API-1 configuration profile: `api_version=1`, `Config.from_dict` of an
  unversioned dictionary (pass `source_version=2`), and `CC_kwargs` with
  `mtd`/`cc_method`/`method`/`root_polynomial_expansion` but no `model`, which
  silently selected `legacy-v1`. All raise a `ValueError`.
- The `"ui-native-v2"` fitting-policy alias (use the default `precision="auto"`).
- The 1.x `ColorCorrectionPipeline.key_functions` module. Its pickles still load:
  `key_functions.Regressor_Model`/`CustomNN` are remapped on unpickling, and they
  now also apply through `predict_image` (2.3.0 failed on missing LUT attributes).
- Legacy fitters and their helpers: `core.fit_model`, `estimate_gamma_profile`,
  `wb_correction`, `color_correction`, `color_correction_1`,
  `extract_color_chart_ex`; the `core.metrics` and `core.color_spaces` modules;
  `fitting.fit_observations`/`fit_conventional_observations`;
  `FlatFieldCorrection.compute_multiplier`, `fit_model`, `apply_ffc` and its
  plotting helpers; `ColorCorrection.clear_ffc_cache`.
- The OpenCV MCC chart-detection backend (`backend="mcc"`, `allow_mcc_fallback`,
  `core.HAS_MCC` and its import-time `ImportWarning`).
- `io.read_image`, `io.read_batch` and `io.write_image`; use `read_image_data` and
  `write_image_data`.
- Unused helpers: `core.utils.compute_temperature` (always raised),
  `compute_diag`, `estimate_fit`, `poly_func_torch`, `free_memory`,
  `clear_chart_cache`; `core.accel.xyz_to_lab_fast`, `lab_to_xyz_fast`,
  `ffc_sample_bins`, `extrapolate_saturated_pixels`; `core.transforms`
  `fit_polynomial_surface`/`apply_polynomial_surface`; `batch.collect_batch`.
- `examples/legacy_training.py`, `setup.py`, `MANIFEST.in`, the `[segmentation]`
  extra, the import-time filter for torch `FutureWarning`s and the four `v1_*`
  golden scenarios.

### Changed

- Load-only compatibility code (saved-policy validation, the legacy polynomial
  gamma applier and the `CustomNN`/`DeterministicTorchRegressor` estimators)
  lives in `ColorCorrectionPipeline._compat`. Pickled class paths are unchanged.
- `ColorCorrection.get_reference_values` requires explicit-v1 color semantics.

## [2.3.0]

- Faster high-precision prediction: `precision="high"` runs its stages per pixel block on up to eight threads and no longer keeps a full-frame copy of every intermediate stage. A 3840×2160 prediction takes 1.7 s instead of 7.9 s with a 472 MB instead of 1.35 GB allocation peak, and the bundled 1152×768 sample takes 0.18 s instead of 0.82 s. Outputs are bit-identical; native byte prediction is unchanged.
- Opt-in linear-light flat-field correction: `FFC_kwargs={"domain": "linear"}` or `FFCOptions(apply={"domain": "linear"})` fits the white-field gain from luminance and multiplies decoded linear RGB by it instead of scaling Lab L* only, which removes most of the chroma error an L*-only gain leaves under vignetting. Available with UI-compatible (API 2) fitting; the default `"lightness"` is unchanged, and calibrations, bundles and pickles without the setting replay as before. Linear calibrations record `ffc_domain` and save a `spatial-linear-gain-v1` FFC stage, which earlier versions refuse to load.
- Faster startup: `from ColorCorrectionPipeline import ColorCorrection` takes about 1.1 s instead of 3.7 s, and loading a saved bundle and correcting an image in a new process takes about 2.5 s instead of 4.9 s. colour-science (with matplotlib), scikit-learn and pandas now load on first use; fitting still loads them, so a full calibration takes about the same time. `CustomNN` keeps its import path and existing pickles load unchanged.
- New two-call API: `calibrate(chart, white=..., save_to=...)` returns a `Calibration` (fitted pipeline, metrics, `delta_e`, `correct()`, `save()`), and `correct(image, calibration_or_bundle_path)` returns float RGB. Both are thin layers over `Config` and `ColorCorrection` and give identical results.
- `Config.from_preset("rpcc2" | "rpcc3" | "ccm" | "3x3" | "pls" | "high-precision", ...)` builds a v2 configuration with `frame_index=0`, flat-field on or off, and merged `input`/`chart` overrides.
- `ColorCorrection.fit(chart, white=None, *, name="", config=None)` and `ColorCorrection.predict(image, ...)` are lowercase aliases of `fit_or_raise` and `predict_image`; the original methods are unchanged.
- Clearer errors: multi-frame images (camera JPEGs with a preview frame) say how to choose `frame_index`, and `CC_kwargs={"mtd": "conv"}` points to `cc_method="conv"` instead of first logging that it is using that method.
- Make the legacy `custom` (CustomNN) color-correction method reproducible: `random_state` now seeds weight initialization, batch shuffling and dropout as well as the validation split, without changing the caller's global torch RNG. Fits with this method previously varied between runs, so their results will differ from earlier runs.
- Define `apply_srgb_to_linear_image` and `apply_linear_to_srgb_image`, which `ColorCorrectionPipeline.core.accel.__all__` has listed since 1.4.0, so `from ColorCorrectionPipeline.core.accel import *` works.
- Report a chart/white-image shape mismatch in legacy flat-field correction as a `ValueError`, including under `python -O`.
- Restore the OpenCV MCC chart-detection backend on opencv-contrib 4.11+/5.x, where it silently found no chart (or raised `AttributeError` in `key_functions`) after OpenCV changed the detector API.
- Legacy-path numeric fixes. These change legacy (`legacy-v1` / `mtd` / `cc_method`) outputs slightly; v2 defaults are unchanged:
  - `core.utils.to_uint8` rounds to the nearest code instead of truncating. On the sample, legacy FFC chart ΔE00 after goes from 10.27 to 10.11. `to_float64` accepts an explicit `scale=`.
  - `key_functions.to_uint8` clips and rounds instead of wrapping (1.01 used to become 1, −0.01 became 254), and `to_float64` scales uint16 by 65535.
  - The fit-time saturation adjustment is a continuous per-channel highlight knee. 1.0 still maps to 0.95, but 0.99 no longer jumps to 0.94 and unclipped channels are no longer darkened.
  - The legacy gamma curve keeps black at 0 and white at 100 and never decreases. The fit through the chart's neutrals is unchanged.
  - Lab conversion of sRGB values adapts to a non-D65 `REF_ILLUMINANT`, which removes a bias in Lab and ΔE (sRGB white read as b*=−19 under D50). D65 results are identical.
- Behaviour change, legacy image I/O (`io.write_image`, and `io.read_image` without `strict=True`). The strict path (`write_image_data`, strict reads) is unchanged, and pipeline outputs are unaffected. Files written by `write_image` may differ by one code from earlier versions:
  - `write_image` quantizes with the strict exporter's equation: floats round to the nearest code instead of truncating (half of arbitrary float values used to land one code low), uint16 is scaled by 255/65535 and rounded instead of `// 257`, and float16 is rounded instead of cast.
  - `write_image` raises `ValueError` for NaN or infinite values instead of writing them as 0 behind a cast warning, and clips other integer types to 0–255 instead of wrapping (int32 300 used to become 44).
  - `read_image(..., dtype="float32"/"float64", range_="0-255")` scales 16-bit files to 0–255 (it returned raw 0–65535 codes), and `range_="0-65535"` scales 8-bit files by 257. `range_="0-1"` results are bit-identical.
  - `read_image(..., dtype="uint8")` rounds 16-bit and float sources to the nearest code, matching `read_image(..., strict=True)`; `dtype="uint16"` rounds float sources.
- A few internal `try`/`except` blocks (CUDA cache release, chart plotting, flat-field ROI parsing, legacy gamma search) no longer swallow `KeyboardInterrupt` and `SystemExit`.
- New optional extra `tiff` (`pip install "ColorCorrectionPipeline[tiff]"`) installs `imagecodecs`, which tifffile needs for compressed (e.g. 16-bit LZW) TIFFs.
- Add a pytest suite with golden-output tests for eleven configurations (kept out of the sdist and wheel).
- Progress and diagnostic messages (`Info: Prediction done in ...`, `Detected white plane ...`, `FFC fitting using method ...`, `Info: Using custom CC method ...`, chart-detection retries, `read_batch`/`write_image` progress) now go to the `ColorCorrectionPipeline` logger hierarchy instead of stdout, which carries a `NullHandler`. Enable them with `logging.basicConfig(level=logging.INFO)`; see `docs/logging.md`. Fallbacks that change the result (no white plane detected, missing detector model, unknown FFC fit method, fewer charts than requested, unknown `write_image` color space) are raised as `UserWarning`s and stay visible by default. `CustomNN(verbose=True)` training output and the legacy `key_functions` reporting helpers still print.

## [2.2.0] - 2026-11-13

- Add reproducible random chart sampling with an exact sample count per patch and a recorded seed for UI-native fits. Existing grid recipes keep their sampling behavior. Incompatible fitting policies report an error when exact sampling is requested.
- Reduce full-frame processing during calibration and strict batch export while retaining verified PNG writes and bundle replay.
- Document the linear-light 3×3 CCM workflow and flag implicit legacy conventional fits, which still use their original encoded-sRGB objective.

## [2.1.0] - 2026-11-10

- Added `ColorCorrection.fit_or_raise()` for callers that want a calibration result on success and a `CalibrationError` on failure. Existing `run()` callers retain their return contract.
- Unified activation of fitted and reloaded calibrations so failed refits and bundle loads preserve the active calibration. Updated the examples to save a bundle, reload it and verify its prediction.
- Reduced first-prediction compilation time with the optional Numba backend. Across four models on the 1152 × 768 Windows sample, first prediction with strict PNG export fell from 12.9–21.7 seconds to 2.38–4.78 seconds.
- Bounded native batch execution by the requested tile size, including tiles that split image rows.
- Added a source-checkout tool and protocol for evaluating a calibration on independent chart captures and reporting held-out CIEDE2000 separately from fit residuals.

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
