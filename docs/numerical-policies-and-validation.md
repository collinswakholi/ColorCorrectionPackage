# Opt-in numerical policies and grouped evaluation

Patches 5–6 introduce new calibration algorithms behind explicit versions. Existing
calls, default configurations, metadata-free models and earlier explicit-v1 models
retain their legacy numerical paths. A degree-one legacy gamma fit now works; its
previous degree-selection calculation required at least two candidates and failed.

```python
from ColorCorrectionPipeline import ColorCorrection, Config

config = Config(
    do_ffc=False,  # enable only when a suitable white-field capture is supplied
    color_semantics={"version": "explicit-v1"},
    stage_policy={"version": "modern-v1", "wb_headroom": 0.95},
    fitting_policy={"version": "reproducible-v1", "random_state": 42},
    CC_kwargs={"mtd": "linear", "degree": 1},
    validation_options={"enabled": True, "n_splits": 4},
)
pipeline = ColorCorrection()
metrics, stages, failed = pipeline.run(image, config=config)
report = pipeline.last_run["numerical_report"]
```

`image` may use the existing supported path/array/ImageData contract. Configurations
are copied, and candidate calibration state is committed only after success. The
new policies require explicit-v1 color interpretation and model detector geometry.
Modern CC requires reproducible fitting. A failed new calibration preserves the
previous active models. `show=True` for modern GC/WB/CC is rejected; inspect saved
observations and reports instead. Legacy plotting remains available.

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
Legacy mode retains its identified fit-only scaling for compatibility.

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

## Reproducible fitting

Modern CC contributes one fixed-region mean per canonical cell. Repeated pixels
or repeated means do not create additional independent colors or extra cell weight.
The lower-level helper collapses repeated cell identities with consistent targets.
Global NumPy randomness is not used. Legacy sampling controls (`n_samples`,
`augment_extremes`, `extreme_anchor_repeat`, `shuffle_fit_data`) are rejected for
reproducible-v1 rather than silently accepted.

Supported new fits are polynomial linear least squares, PLS, sklearn MLP and a
deterministic CPU Torch NN. `linearCC` aliases linear and `MLP` aliases nn. Linear
fits require full column rank and acceptable conditioning. PLS components are
bounded by centered design rank and independent-cell count. Neural learning rate,
batch size, patience and explicit positive iteration caps affect the actual fit.
Unsupported/irrelevant options and `param_search=True` fail clearly. The conventional
route supports Finlayson 2015 with explicit degree/root-polynomial choice and a
stable least-squares solver. Other conventional bases are not silently substituted.

Torch initialization, dropout, data ordering and grouped internal split are seeded
in a scoped CPU RNG context. Library fitting calls use a lock and restore caller
RNG/determinism settings afterward. Unrelated external Torch RNG calls occurring
concurrently in the same process are outside that isolation guarantee. GPU training
is not qualified by reproducible-v1. Fixed software/hardware CPU results are the
tested reproducibility scope; cross-version/platform bitwise identity is not promised.

There are no default black/white anchors. Positive `FittingPolicy.anchor_weight`
adds two named synthetic identity-endpoint assumptions with that weight for linear
or conventional least squares. Anchors are train-only, never reported as measured,
and do not count toward independent measured-cell sufficiency. Nonlinear estimators
reject nonzero anchor weighting. Their internal grouped holdout remains separate
from outer evaluation. sklearn MLP uses training-objective stopping, without a
row-level internal validation split that could leak repeated cells.

Direct estimator output remains the numerical reference. Reproducible pipeline
CC defaults to direct application; opt-in LUT controls retain the existing LUT
approximation. Effective parameters, feature order, group counts, weighting,
rank/condition and execution device are recorded with the fitted model.

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
