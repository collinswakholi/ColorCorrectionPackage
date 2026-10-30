# UI-compatible fitting

`fitting_policy="ui-compatible-v1"` selects additive numerical implementations
of the UI engine's CCM, root-polynomial, polynomial, PLS and residual NN fits.
It does not change the default `legacy-v1` or existing `reproducible-v1` fits,
their feature domains, or previously saved calibrations.

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
guarantees. These explicit precision choices apply only to this new policy.

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

Safe bundles store the known UI feature-order identifier, effective parameters
and owned affine/network arrays. Loading validates their structure and matching
fitting/color metadata before activation. No UI executable, UI checkout or
external estimator deserializer is needed. Older package readers reject the new
estimator kind explicitly. Existing bundle kinds and trusted pickle APIs retain
their established behavior.

The numerical algorithms are ported from the MIT-licensed
[UI engine](https://github.com/collinswakholi/ColorCorrectionPackage_UI),
including `maps.rs`, `pls.rs`, `mlp.rs`, `sample.rs` and `ffc.rs`.
