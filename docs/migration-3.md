# Migrating to 3.0

3.0 removes the historical fitting recipes and the helpers that only served them.
**Calibrations saved by 2.x and earlier still load and apply unchanged**, and the
2.x default fit (`Config()`, presets, `calibrate`/`correct`) gives the same
results as in 2.3.0.

## Saved models keep working

Everything below loads through the same public APIs as before
(`MyModels.load`, `MyModels.load_bundle`, `ColorCorrection.load_bundle`,
`correct(image, "calibration.ccpb")`) and predicts the same pixels:

- pickles and bundles (schema/format version 1 and 2) fitted with
  `fitting_policy="legacy-v1"`, `"reproducible-v1"` or `"ui-compatible-v1"` with
  the `staged-v1` runtime, including conv and `ours` linear/PLS/NN/`custom` CC,
  legacy polynomial gamma, white balance and FFC gains;
- configurations fitted from API-1 (unversioned) dictionaries;
- metadata-free containers written before 2.0, read as `legacy-v1`;
- 1.x pickles whose classes live at `ColorCorrectionPipeline.key_functions`
  (`Regressor_Model`, `CustomNN`). Unpickling remaps them onto the current
  classes. 2.3.0 unpickled these but failed in `predict_image`; they now apply.

Pickle is still the default `model_format`, and pickles remain a trusted-file
format. The load-only code lives in `ColorCorrectionPipeline/_compat/`; nothing in
it fits a calibration. `tests/test_legacy_model_loading.py` checks fixtures written
by 2.3.0 for each removed recipe.

## Fitting requests that now raise

Each of these raises a `ValueError` that names the replacement. 3.0 does not
reinterpret them silently:

| 2.x request | 3.0 replacement |
| --- | --- |
| `Config(fitting_policy="legacy-v1")`, `"reproducible-v1"`, `"ui-compatible-v1"` (staged-v1 runtime) | `Config()` (RPCC2), or `Config(model="3x3"\|"rpcc3"\|"linear"\|"pls"\|"nn", basis=..., degree=...)` |
| `Config(fitting_policy="ui-native-v2")` | `Config()`: `precision="auto"` is the same auto-v2 request |
| `CC_kwargs={"mtd": ..., "cc_method": ..., "method": ..., "root_polynomial_expansion": ...}` without `model`/`basis` (selected legacy-v1) | `Config(model=...)`. Conventional encoded-sRGB `conv` → `Config(model="3x3")` fits a linear-light 3×3 CCM |
| `api_version=1` | API 2 only (the default) |
| `Config.from_dict(unversioned_dict)` (read as API 1) | `Config.from_dict(mapping, source_version=2)`, or save with `to_dict()`/`to_grouped_dict()`, which record `api_version` |
| `stage_policy="legacy-v1"`, `color_semantics` version `legacy-v1` | the defaults (`modern-v1`, `explicit-v1`) |
| `chart_detection={"backend": "mcc"}`, `allow_mcc_fallback` | `backend="auto"` (the bundled chart model) |

`fitting_policy="ui-native-v1"` (an explicit native-v1 runtime) and the
`cc_method="ours"|"conv"` vocabulary together with an explicit `model` are still
accepted. A removed recipe also cannot be refitted from a loaded calibration;
fit a new one with the current recipe and compare it on your own validation
images before replacing it.

## Removed APIs

| Removed | Use instead |
| --- | --- |
| `ColorCorrectionPipeline.key_functions` (1.x module) | `ColorCorrection`, `Config`; its pickles still load |
| `core.fit_model`, `estimate_gamma_profile`, `wb_correction`, `color_correction`, `color_correction_1`, `extract_color_chart_ex` | `ColorCorrection.fit(...)` |
| `core.metrics` (`Metrics`, `get_metrics`, `arrange_metrics`, `desc_stats`, `compute_mae`) | fit metrics (`delta_e00_*`) and `ColorSemantics.delta_e00` |
| `core.color_spaces` (`convert_to_lab`, `convert_to_LCHab`, `srgb_to_cielab_D50`, `adapt_chart`, `do_color_adaptation`, saturation extrapolation) | `ColorCorrectionPipeline.colorimetry` |
| `core.utils.compute_temperature` (always raised), `compute_diag`, `estimate_fit`, `poly_func_torch`, `free_memory`, `clear_chart_cache`, `core.HAS_MCC` | none needed |
| `core.accel.xyz_to_lab_fast`, `lab_to_xyz_fast` | `colorimetry.xyz_to_lab`, `lab_to_xyz` |
| `fitting.fit_observations`, `fit_conventional_observations` | `ColorCorrection.fit(...)` (`DeterministicTorchRegressor` stays load-only) |
| `FlatFieldCorrection.compute_multiplier`, `fit_model`, `apply_ffc`, plotting helpers | the pipeline FFC stage; `detect_and_crop` and safe-mode validation remain |
| `ColorCorrection.clear_ffc_cache` and the FFC multiplier cache | none needed (the UI FFC estimator never used it) |
| `ColorCorrection.get_reference_values` for `legacy-v1` semantics | explicit-v1 only |
| `io.read_image` (including `strict=True`), `io.read_batch`, `io.write_image` | `io.read_image_data(path, ...).pixels`, `io.write_image_data` |
| `examples/legacy_training.py` | `examples/customizable.py`, `examples/compare_bases.py` |
| the `[segmentation]` extra, `setup.py`, `MANIFEST.in` | Hatchling builds from `pyproject.toml` |

The package no longer installs an import-time filter for torch `FutureWarning`s,
and `core` no longer warns when OpenCV's `mcc` module is missing.
