# Numerical policies and grouped evaluation

Numerical profiles identify the training and application recipes of a calibration.
New fits use the UI-compatible RPCC2 recipe with modern-v1 stages and
explicit-v1 color semantics. Fitting with `legacy-v1` or `reproducible-v1` was
removed in 3.0; models saved with them still load and apply, and metadata-free
saved models restore legacy interpretation. See [Migrating to 3.0](migration-3.md).

See the [migration guide](migration-2.md) for solver/basis controls and
precision selection.

```python
from ColorCorrectionPipeline import ColorCorrection, Config

config = Config(
    do_ffc=False,  # enable only when a suitable white-field capture is supplied
    model="3x3",
    validation_options={"enabled": True, "n_splits": 4},
    image_options={"frame_index": 0},  # camera JPEGs often carry a preview frame
)
pipeline = ColorCorrection()
metrics, stages, failed = pipeline.run(image, config=config)
report = pipeline.last_run["numerical_report"]
```

`image` may use the existing supported path/array/ImageData contract. Configurations
are copied, and candidate calibration state is committed only after success. The
new policies require explicit-v1 color interpretation and model detector geometry.
CC uses the [UI-compatible policy](ui-compatible-fitting.md), with its own explicit
linear-light fitting, sampling and FFC semantics. A failed new calibration preserves the
previous active models. `show=True` for modern GC/WB/CC is rejected; inspect saved
observations and reports instead.

## Saturation evidence and selection

Modern saturation is an **identity image operation**, without hidden 0.95 scaling
or reference-assisted estimated measurements. Original source samples are captured
before FFC or any learned image operation. Reports include per-cell any-channel
and per-channel clipping fractions at the observation's encoded RGB threshold.
These are sampled encoded-image pixels, not sensor RAW clipping evidence.

The default `StagePolicy.saturation='exclude'` excludes a cell if its original or
current sampled clipping fraction exceeds `max_clipped_fraction` (default zero).
The alternative `'report'` deliberately keeps flagged cells and records that
choice. Source and current observations remain unchanged. Neither policy claims
to recover lost information.

In modern mode, `check_saturation` controls the standalone reporting stage. GC,
WB and CC still obey their declared StagePolicy selection when that switch is
false. Use `saturation='report'` to retain flagged training cells explicitly.
The UI-compatible fit that `Config` builds uses `saturation='report'` and requires
`check_saturation=False`.

## Neutral gamma and white balance

Gamma uses only `ReferenceProfile.neutral_ids`, full-precision fixed-region means,
canonical reference lightness at working D65, and a monotonic PCHIP in Lab L*.
It requires at least three usable, distinct measured neutral values spanning at
least 20 L* by default, **before** adding assumed 0→0 and 100→100 endpoints.
Clipped/unusable neutrals are excluded. Duplicate measured lightness values are
merged only when their target lightness agrees within the declared tolerance;
contradictory order or insufficient information fails calibration.

Endpoints are stored model assumptions, not measured observations. Input L* is
bounded to [0,100]; output is finite/monotonic. Fit and prediction use the same
tone application and explicit D65 conversion. Adjusting L* while preserving a*/b*
can produce out-of-gamut RGB; the pipeline's recorded clipping policy still applies.
This is not a radiometric or linear-light response calibration.

WB fits median per-channel reference/measured ratios from at least two usable
declared neutral cells. Minimum signal, positive finite gains and maximum gain are
checked. `wb_headroom=0.95` preserves historical headroom; setting 1.0 is an explicit
numerical choice. The fitted diagonal matrix stores the actual headroom operation.
No same-reference identity expectation should ignore the chosen headroom.

## Guarded flat-field estimation

Modern FFC retains the established uint8 Lab-lightness method and float image
application. It adds quality gates rather than claiming higher-precision radiometry:

- Explicit, in-bounds crop; adequate area and distinct sampling coordinates.
- Signal and saturation checks on raw crop pixels before smoothing or division.
- Valid degree, smoothing, seed, fitting controls, design rank and conditioning.
- Positive bounded observed gains, fitted grid and final interpolated gain map.
- Calibration dimensions required during application; implicit resizing is rejected.

By default no-plane detection fails with guidance. Specify an explicit `crop_rect`
or `missing_plane='full-image'` to make full-frame fallback intentional. Interactive
manual crop is unavailable in safe mode without an explicit rectangle. Defaults
include normalized L* signal ≥0.02, saturated fraction ≤0.01, crop fraction ≥0.1,
and gains within [0.5,1.8]. These configurable safety limits are operational choices,
not validated universal acquisition thresholds.

Example FFC settings:

```python
FFC_kwargs = {
    "crop_rect": [0, 0, width, height], "bins": 30, "smooth_window": 5,
    "degree": 3, "fit_method": "linear", "min_gain": 0.5, "max_gain": 1.8,
}
```

`numerical_report['FFC']` records actual plane selection, dimensions, signal,
saturation, design and gain limits. Cached gain reuse retains that quality record.
Saved modern gains cannot load with absent/invalid bounds or incompatible recorded
dimensions. Matching dimensions alone does not establish matching lens, crop,
orientation, illumination, exposure or camera conditions. Dark-frame subtraction
and linear/high-precision FFC remain later work.

## Reproducible fitting (removed in 3.0)

The `reproducible-v1` fitter (encoded-sRGB polynomial least squares, PLS, sklearn
MLP, deterministic CPU Torch NN and the Finlayson conventional route) was removed
in 3.0. Models saved with it still load and apply with their recorded estimators
and reports. See [Migrating to 3.0](migration-3.md).

## Fit objective and perceptual evaluation

New fits minimize RGB squared error in linear light, not CIEDE2000. PLS and
neural fits also retain their recorded objectives and stopping rules. A lower
RGB error can therefore accompany a higher mean Delta-E.

Canonical reference truth and fitting targets remain unclipped. A physical
reference outside the display gamut can have a negative encoded channel that
cannot be reproduced by a clipped sRGB export. Forcing a fit toward that target can move otherwise accurate colors. Clipping display
swatches, dropping difficult cells or substituting reference-assisted samples
would change the stated fitting problem; the package does not do so implicitly.

Saved `reproducible-v1` fit reports also record `fitting_domain`, `target_domain`,
`out_of_gamut_targets` and `objective.optimizes_delta_e00=False`. Per-stage
canonical Delta-E metrics remain the separate check on rendered results.

For a linear-light matrix matched to the UI, select `Config(model="3x3")`.
It also minimizes RGB error, so perceptual improvement still
requires measurement. The affine `linear` model and the 3x3 matrix are different
hypothesis classes; equal polynomial degree alone does not match their settings.
See [UI-compatible fitting](ui-compatible-fitting.md) for complete controls.

## What the evaluation means

Default new evaluation uses four deterministic grouped folds. Every canonical
cell identity stays entirely in training or testing; excluded cells appear in
neither, and synthetic anchors are training-only. Each fold fits a fresh CC model.
Held-out targets never enter that fit. Hyperparameters are caller-selected;
automatic method selection or search is not performed on these folds.

Reports label scope **CC-only, conditional on already fitted upstream stages**.
They score direct estimator responses to held-out measured region means against
unclipped canonical reference truth using common-white CIEDE2000. This is not
whole-pipeline validation, mean corrected-pixel scoring, LUT validation, an
independent capture test or physical color-accuracy certification. Training-chart
stage residuals still use actual transformed image samples and are separate.

Reports account for expected/scored/excluded/failed rows and groups, preserve fold
membership/reasons, and use equal group weights. Any failed/incomplete fold makes
the aggregate unavailable; successful partial fold diagnostics remain visible.
Insufficient groups produce unavailable with a reason. No synthetic endpoint or
reference-assisted estimate is counted as independent test evidence. Increasing
polynomial degree can make folds underdetermined even when the full fit is viable;
that is reported rather than hidden. Disable evaluation explicitly through
`validation_options.enabled=False` when needed.

The public tuple return remains. CC metrics include `fitting` and `validation`;
`last_run['numerical_report']` and model metadata also retain the complete records.
Metadata validation checks new policies and estimator versions before activation.
Prefer the non-executable [portable calibration bundle](calibration-bundles.md).
Trusted-file pickle persistence remains available for compatibility.

For whole-session validation on separate photographs, see the [independent capture study](capture-study.md).
