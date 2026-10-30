# UI-compatible fitting

`fitting_policy="ui-compatible-v1"` selects additive numerical implementations
of the UI engine's CCM, root-polynomial, polynomial, PLS and residual NN fits.
It does not change the default `legacy-v1` or existing `reproducible-v1` fits,
their feature domains, or previously saved calibrations.
Select it through the pipeline `Config`. The legacy `fit_model` adapter rejects
this policy because it has no reference or observation contract to interpret it.

```python
from ColorCorrectionPipeline import ColorCorrection, Config

config = Config(
    fitting_policy="ui-compatible-v1",
    do_ffc=False,
    check_saturation=False,
    CC_kwargs={"model": "rpcc2"},
    validation_options={"enabled": True, "n_splits": 4},
    save=True,
    model_format="bundle",
    save_path="calibration",
)
pipeline = ColorCorrection()
metrics, stages, failed = pipeline.run("chart.png", config=config)
if failed:
    raise RuntimeError(pipeline.last_run["errors"])
images = pipeline.predict_image("sample.png", stages="final")
```

When omitted, this policy selects explicit sRGB/D65 semantics with linear-RGB
fitting, modern neutral stages, saturation reporting, UI Classic24 reference
values, and UI sampling. Supplied conflicting policies fail validation. Input
normalization, output encodings and reference contracts still use the existing
explicit APIs. Fitting uses Bradford adaptation; another adaptation must use a
different fitting policy.

## Model choices

| `CC_kwargs` | Numerical operation |
| --- | --- |
| `{"model": "ccm"}` | Linear-light 3-channel matrix, no intercept |
| `{"model": "rpcc2"}` | Six-term degree-2 root-polynomial matrix |
| `{"model": "rpcc3"}` | Thirteen-term degree-3 root-polynomial matrix |
| `{"mtd": "linear", "degree": 2}` | Polynomial ridge with separate intercept |
| `{"mtd": "pls", "degree": 2, "ncomp": 6}` | Population-standardized NIPALS PLS |
| `{"mtd": "nn", "degree": 2, "hidden_layers": [100]}` | Float64 residual ReLU network |

The `model` choice determines CCM/root-polynomial features. `degree` controls
the polynomial basis for `linear`, `pls` and `nn`, with the UI's 96-feature
ceiling. `include_combinations=False` is supported for polynomial linear fits.
Existing `cc_method="conv"`, `method="Finlayson 2015"`, degree 1/2/3 and
`root_polynomial_expansion=True` map to CCM/RPCC2/RPCC3 in this policy. The
`custom` and `mlp` method aliases select the same UI residual NN.

CCM, RPCC and polynomial linear fits use ridge 1e-5. PLS uses population
standardization, a fixed 500-iteration inner cap, default tolerance 1e-6, and
the native component-selection rule. NN defaults are width 100, Adam, learning
rate .001, cap 10,000 epochs, patience 1,500, tolerance 1e-6, no dropout and no
batch normalization. Its fixed seed is 42. Supported NN controls also include
SGD/RMSprop, one to four widths in [4,512], dropout up to .8 and batch
normalization. Batch size follows the UI's `min(200, training rows)` rule; a
supplied `batch_size` must agree. An explicit cap of 10,000 does not promise
that early stopping will run all 10,000 epochs.

Unknown or unsupported controls fail instead of silently selecting a different
algorithm. Training is CPU float64. Chart detection retains its independent
GPU-first provider policy and CPU fallback.

## Image and stage semantics

Detection geometry uses the original capture. The opt-in sampling path keeps
homography coordinates in float64 and samples the inner 70% of each cropped
patch using the native per-axis grid. `chart_detection.sampling_inset` defaults
to .75 and `samples_per_axis` to 9, producing 81 samples per patch. Their product
is an effective 52.5% inner region for these defaults.

Measured encoded RGB means are decoded before feature expansion. Reference
targets are adapted to D65 and converted to unclipped linear sRGB. Prediction
clips linear RGB before encoding. A residual NN adds its predicted residual to
the linear-RGB identity input.

Working image stages deliberately round through float32, as the native UI does;
the public arrays remain float64 containers. The default CC LUT is 33 cubed,
with float32-rounded table values. `use_lut=False` selects direct evaluation.
The LUT applies to all image sizes by default (`lut_min_pixels=1`), avoiding a
change of rendering method when an image is split into small chunks.
Strict export verification and atomic bundle persistence retain their existing
guarantees. These explicit precision choices apply to the UI fitting policies.

For FFC, enable `do_ffc=True` and supply a suitable white image. This policy
qualifies `fit_method="linear"` (a plane, effective degree 1) and `"poly"`
(default degree 5, ridge 1e-8). It reproduces the native float32 lightness-window
sampling, extremity transform and endpoint-aligned cubic gain interpolation.
For example:

```python
FFC_kwargs = {
    "fit_method": "linear", "bins": 30, "smooth_window": 7,
    "crop_rect": [43, 42, 1107, 723],
}
```

The crop uses `[x1,y1,x2,y2]`. Quality checks still require adequate signal,
coverage, positive bounded gains and matching image dimensions. FFC PLS/NN and
other unqualified FFC controls fail clearly in this policy; they remain available
through the existing policies. This is a Lab-lightness correction, not a
radiometric calibration.

## Matching the native UI image pipeline

For new calibrations that must follow the native UI's normal image pipeline,
select `fitting_policy="ui-native-v1"`. This is an explicit runtime variant of
the UI-compatible fitters, recorded in the calibration and checked on reload.
The existing `"ui-compatible-v1"` selection keeps its staged image equations;
neither default selection nor an existing calibration is silently upgraded.

```python
config = Config(
    fitting_policy="ui-native-v1",
    check_saturation=False,
    CC_kwargs={"model": "rpcc3"},
    do_ffc=False,
)
```

This mode requires full-range uint8 sRGB captures, white images and prediction
inputs. It follows the native float32 byte normalization, detector letterbox,
ordered homography and sample reduction, and the native D50 coordinates. When
FFC and GC are enabled together, GC training samples the original capture
before applying the spatial gain to each sample. Rendering applies FFC and GC
in one Lab conversion, matching UI `fast_mode=False` without an FFC preview.
Requesting an FFC output also returns that standalone stage; the final GC/WB/CC
output retains the recorded fusion rule. This differs from the older staged
policy's separate FFC and GC image boundaries.

For a custom reference supplied as physical XYZ or Lab values, construct it with
`ReferenceProfile.from_values(..., retain_source=True)` to retain its original
coordinates alongside canonical Lab D50. The retained values are owned, hashed
and checked against canonical truth before use. A contradictory source fails.
Other references continue to use canonical Lab D50; CAT02-derived references
also retain canonical truth rather than being reinterpreted as Bradford sources.
The built-in native reference retains the original UI catalog coordinates.

For comparisons between APIs, shared physical XYZ D65 targets avoid a source
white ambiguity: the UI uses D50 coordinates `(0.34567, 0.35850)`, while the
package's existing `"D50"` shorthand retains `(0.3457, 0.3585)`. Supply explicit
source coordinates when retaining Lab references; equal white labels alone do
not establish identical reference conversion.

Use `predict_image` or `predict_batch` with this mode. `prepare_execution`
currently rejects it explicitly because prepared execution does not represent
these fusion rules. Direct CC and explicit chunking retain reference execution.
The modern neutral quality checks, strict exports and transactional bundle
persistence remain active. CPU NumPy is the reference implementation; optional
Numba executes the same equations. No UI checkout or native UI binary is needed
by the installed package.

Image kernels use bounded workers around Numba functions that release the GIL.
They avoid an OpenMP spin phase, respect the configured thread cap (up to eight),
and execute serially inside existing batch workers. The package does not change
global OpenMP environment settings. Worker failure joins outstanding jobs and
recomputes from the original input through the reference path. The uint8 decode
cache recognizes exact stored code values; all other values use the original
power calculation, with no approximate transfer-function lookup.

## Qualification and limits

The current high-level contract supports Classic24. Set `check_saturation=False`;
this policy requires `StagePolicy.saturation="report"`, no synthetic anchors,
seed 42 and the default `condition_limit`. It qualifies the UI configuration
with saturation recovery and saturated-cell skipping disabled. It does not
reproduce the UI's reference-assisted saturation recovery. Modern neutral-stage
quality gates can reject insufficient or contradictory neutrals that another
implementation would accept.

Grouped validation refits held-out colors, including the UI's automatic
per-fold batch size. Its result is conditional on the already fitted upstream
stages; it is not independent-capture or whole-pipeline validation. Failed folds
remain explicit and do not produce a misleading aggregate score.

Identical-observation solver checks and end-to-end image checks answer different
questions. In particular, small detector/sampler differences can change an NN's
optimization trajectory. Numerical compatibility does not promise byte identity
between independently detected calibrations, equal speed to the native engine,
or independent color accuracy from a training-chart residual.
NN training uses bias-first dense sums, row-ordered gradient reductions and
native scalar feature construction. This removes the long-run drift caused by
BLAS reduction order: the qualified Windows 100-unit, 10,000-epoch reference
matches every parameter exactly on identical binary design and target arrays,
with either NumPy or optional Numba training arithmetic. The fit report records
`native-ordered-f64-v1` and the actual arithmetic backend.

This does not promise identical independently fitted networks: observation
geometry, sampling and reference conversion must also agree, and platform math
libraries can differ. Existing saved networks retain their original inference
equations; training changes do not reinterpret their stored parameters. Each
saved model retains deterministic prediction and exact bundle reload semantics.

With the `performance` extra, ordinary unchunked `predict_image` calls can run
supported UI stages in one CPU pass, with the recorded CC LUT. The staged
runtime retains each float32 boundary; the native runtime retains its recorded
FFC/GC fusion rule. This adds no combined LUT approximation. Guarded optional
execution and kernel failures use the corresponding reference implementation.
The staged runtime also uses reference execution for custom hooks, direct CC
and batch calls requesting stage timing context. Native hook and prepared-mode
limits are described above. Initial JIT compilation is separate from warm
execution; see [measured performance](performance.md).

Safe bundles store the known UI feature-order identifier, effective parameters
and owned affine/network arrays. Loading validates their structure and matching
fitting/color metadata before activation. No UI executable, UI checkout or
external estimator deserializer is needed. Older package readers reject the new
estimator kind explicitly. Existing bundle kinds and trusted pickle APIs retain
their established behavior.

The numerical algorithms are ported from the MIT-licensed
[UI engine](https://github.com/collinswakholi/ColorCorrectionPackage_UI),
including `maps.rs`, `pls.rs`, `mlp.rs`, `sample.rs` and `ffc.rs`.
