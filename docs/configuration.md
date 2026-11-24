# Configuration guide

A calibration runs up to four stages, in this order:

| Stage | What it corrects | Needs |
| --- | --- | --- |
| **FFC**, flat-field correction | Uneven lighting across the frame (vignetting, hot spots) | A photo of a plain white target taken with the same setup |
| **GC**, gamma correction | Tone: how measured grey levels map to true lightness | The chart's six grey patches |
| **WB**, white balance | Color cast of the light | The chart's grey patches |
| **CC**, color correction | Remaining color error | All 24 chart patches |

All four are on by default. Everything below is optional: the defaults suit most
ColorChecker Classic 24 photographs.

## Three ways to configure

Pick the simplest one that does what you need.

**1. A preset name.** Good for most work.

```python
from ColorCorrectionPipeline import calibrate, correct

cal = calibrate("chart.jpg", white="white.jpg", model="rpcc2")
image = correct("scene.jpg", cal)          # float RGB, 0–1
```

**2. A preset plus a few changes.** `Config.from_preset` takes the same names and
accepts any setting from the reference tables below.

```python
from ColorCorrectionPipeline import Config, calibrate

config = Config.from_preset("rpcc3", white=True, steps={"cc": {"validation": {"enabled": False}}})
cal = calibrate("chart.jpg", white="white.jpg", config=config)
```

**3. A full `Config`.** Settings are grouped by the stage that uses them
(`steps.ffc`, `steps.gc`, `steps.wb`, `steps.cc`), plus a few shared groups
(`input`, `chart`, `numerics`, `persistence`, …). Plain dictionaries and the
typed option classes are interchangeable:

```python
from ColorCorrectionPipeline import Config, Steps, FFCOptions, CCOptions

config = Config(
    steps=Steps(
        ffc=FFCOptions(fit={"method": "linear"}),
        cc=CCOptions(fit={"model": "rpcc3"}),
    ),
    input={"frame_index": 0},
)

same = Config(
    steps={"ffc": {"fit": {"method": "linear"}}, "cc": {"fit": {"model": "rpcc3"}}},
    input={"frame_index": 0},
)
```

Anything you leave out keeps its default. Unknown names raise an error, so a
typo never silently does nothing.

`calibrate()` and `Config.from_preset()` read frame 0 of multi-frame files by
default, because many camera JPEGs embed a preview frame. A plain `Config()`
does not: add `input={"frame_index": 0}` if your JPEGs are rejected as
multi-frame.

## Presets

| Name | Color model | Use when |
| --- | --- | --- |
| `rpcc2` (default) | Root-polynomial, degree 2 | General use. Robust to exposure changes. |
| `rpcc3` | Root-polynomial, degree 3 | A well-exposed chart and you want a closer fit. |
| `ccm` (alias `3x3`) | 3×3 matrix in linear light | You want the simplest, most predictable transform. |
| `pls` | Partial least squares | Many correlated features. |
| `high-precision` | `rpcc2` computed in float64 | Scenes are 16-bit or float, even if the chart is an 8-bit JPEG. |

## Recipes

Each recipe is complete: pass the `config` to `calibrate(..., config=config)`
or `ColorCorrection().fit(..., config=config)`.

### No white image: skip flat-field correction

```python
from ColorCorrectionPipeline import calibrate

cal = calibrate("chart.jpg")                  # no white= means no FFC
```

With a full `Config`, turn FFC off explicitly and pass no white image:

```python
from ColorCorrectionPipeline import ColorCorrection, Config

config = Config(steps={"ffc": {"enabled": False}}, input={"frame_index": 0})
ColorCorrection().fit("chart.jpg", None, config=config)
```

### Run only some stages

```python
from ColorCorrectionPipeline import Config

wb_and_cc_only = Config(
    steps={"ffc": {"enabled": False}, "gc": {"enabled": False}},
    input={"frame_index": 0},
)
```

### Choose the color model and its features

`ccm`, `rpcc2` and `rpcc3` have fixed features. `linear`, `pls` and `nn` let
you choose the feature **basis** (`polynomial`, `root` or `hybrid`) and
**degree** (1–4):

```python
from ColorCorrectionPipeline import Config

root_pls = Config(steps={"cc": {"fit": {
    "model": "pls", "basis": "root", "degree": 3,
    "pls": {"components": 6},          # root or hybrid PLS needs a component count
}}})

small_nn = Config(steps={"cc": {"fit": {
    "model": "nn", "basis": "polynomial", "degree": 2,
    "nn": {"hidden_layers": [64], "max_iterations": 3000},
}}})
```

The top-level shortcuts `Config(model=..., basis=..., degree=...)` do the same.

### Flat field: smoother surface, or correct color as well as brightness

```python
from ColorCorrectionPipeline import Config

config = Config(steps={"ffc": {
    "fit": {"method": "linear"},       # a plane instead of the default degree-5 surface
    "apply": {"domain": "linear"},     # gain on linear RGB: also fixes vignetting color shifts
}})
```

The default `domain="lightness"` scales only Lab L\*. `"linear"` multiplies
linear-light RGB, which removes the color fringing an L\*-only gain leaves in
strongly vignetted corners.

The white plane is found automatically. To use a fixed region instead, give
its pixel box as `[x1, y1, x2, y2]` in the white image:

```python
from ColorCorrectionPipeline import Config

config = Config(steps={"ffc": {"white_plane": {"crop_rect": [40, 40, 1100, 720]}}})
```

### 16-bit or float scenes

```python
from ColorCorrectionPipeline import Config

config = Config(numerics={"precision": "high"})   # or Config.from_preset("high-precision")
```

`auto` (the default) picks byte or float64 arithmetic from the calibration
images. Choose `high` when an 8-bit chart will later correct 16-bit or float
scenes.

### Faster calibration: skip cross-validation

By default the CC stage also runs a 4-fold held-out check on the chart patches
and reports it in the metrics. It doesn't change the fitted model. Turn it off
to save time:

```python
from ColorCorrectionPipeline import Config

config = Config(steps={"cc": {"validation": {"enabled": False}}})
```

### Force CPU chart detection

Detection tries a GPU first and falls back to CPU. To always use the CPU:

```python
from ColorCorrectionPipeline import Config, calibrate

config = Config(chart={"device": "cpu"}, input={"frame_index": 0})
cal = calibrate("chart.jpg", device="cpu")          # same, via the shortcut
```

### Reproducible random patch sampling

By default each patch is read on a fixed 9 × 9 grid (81 points). For an exact
number of random points per patch instead:

```python
from ColorCorrectionPipeline import Config

config = Config(chart={"samples_per_patch": 20, "sampling_seed": 42})
```

`samples_per_patch` is 1–4096; the seed defaults to 0. Don't combine it with
`samples_per_axis`.

### Save the calibration automatically

```python
from ColorCorrectionPipeline import Config

config = Config(persistence={"enabled": True, "directory": "results", "format": "bundle"})
```

The fit then writes `results/<name>_models.ccpb` (the name defaults to
`chart` with `calibrate()`). Always set `format="bundle"`: the default
`"pickle"` is kept only for compatibility and should be loaded only from
trusted sources. You can also skip this and call `cal.save("calibration.ccpb")`
yourself.

### Your own chart reference values

The built-in reference is the X-Rite ColorChecker Classic 24. If you measured
your own chart, supply its 24 values in row-major order (top-left to
bottom-right):

```python
from ColorCorrectionPipeline import Config, ReferenceProfile

reference = ReferenceProfile.from_values(
    measured_lab,                      # 24 × 3 Lab values
    color_space="lab", white="D50", scale=1.0,
    profile_id="my-chart-2026", revision=1,
    names=patch_names,                 # 24 labels
    neutral_ids=["r4c1", "r4c2", "r4c3", "r4c4", "r4c5", "r4c6"],
)
config = Config(reference=reference)
```

`neutral_ids` names the grey patches GC and WB use (`r4c1` is row 4, column 1).

### Save settings to a file and load them back

```python
import json
from ColorCorrectionPipeline import Config

config = Config(model="rpcc3", steps={"ffc": {"enabled": False}})
with open("settings.json", "w", encoding="utf8") as f:
    json.dump(config.to_grouped_dict(), f, indent=2)

with open("settings.json", encoding="utf8") as f:
    restored = Config.from_dict(json.load(f))
```

The saved file records `api_version: 2`. A dictionary without it is rejected,
because unversioned dictionaries used the configuration format removed in 3.0.

## Reference

Defaults are what you get when a setting is left out.

### `steps.ffc`: flat-field correction

| Setting | Default | Allowed | Meaning |
| --- | --- | --- | --- |
| `enabled` | `True` | bool | Run FFC. Needs a white image the same size as the chart and scenes. |
| `fit.method` | `"poly"` | `"poly"`, `"linear"` | Shape of the illumination surface. `linear` is a plane. |
| `fit.degree` | `5` | 1–8 | Polynomial degree for `poly`. Ignored for `linear`. |
| `fit.bins` | `50` | 2–512 | Grid cells per axis used to sample the white image. |
| `fit.smooth_window` | `5` | odd integer | Smoothing window over those cells. |
| `apply.domain` | `"lightness"` | `"lightness"`, `"linear"` | Apply the gain to Lab L\* only, or to linear RGB. |
| `white_plane.crop_rect` | detected | `[x1, y1, x2, y2]` | Fixed white region instead of automatic detection. |
| `white_plane.model_path` | bundled | path | Custom white-plane detector model. |
| `quality.min_signal` | `0.02` | 0–1 | Reject a white image darker than this. |
| `quality.max_saturated_fraction` | `0.01` | 0–1 | Reject a white image with more clipped pixels than this. |
| `quality.min_crop_fraction` | `0.1` | 0–1 | Reject a white region smaller than this share of the frame. |
| `quality.min_gain`, `quality.max_gain` | `0.5`, `1.8` | > 0 | Reject corrections outside this gain range. |

### `steps.gc`: gamma correction

GC fits a smooth, monotonic curve through the grey patches. There is no gamma
exponent to set; these settings are safety checks.

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Run GC. |
| `quality.min_neutrals` | `3` | Fewest usable grey patches. |
| `quality.min_span` | `20.0` | Smallest lightness range (L\*) the grey patches must cover. |
| `quality.duplicate_tolerance` | `1e-06` | Grey patches closer than this count as duplicates. |
| `quality.target_tolerance` | `1.0` | Duplicates whose reference values disagree by more than this fail the fit. |

### `steps.wb`: white balance

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Run WB. |
| `fit.headroom` | `0.95` | Scales the channel gains down slightly to avoid clipping highlights. |
| `quality.min_signal` | `0.01` | Ignore grey patches darker than this. |
| `quality.max_gain` | `8.0` | Fail if a channel would need more gain than this. |

### `steps.cc`: color correction

| Setting | Default | Allowed | Meaning |
| --- | --- | --- | --- |
| `enabled` | `True` | bool | Run CC. |
| `fit.model` | `"rpcc2"` | `ccm`/`3x3`, `rpcc2`, `rpcc3`, `linear`, `pls`, `nn` | Color model. |
| `fit.basis` | per model | `polynomial`, `root`, `hybrid` | Feature type for `linear`, `pls`, `nn`. |
| `fit.degree` | per model | 1–4 | Feature degree for `linear`, `pls`, `nn`. |
| `fit.include_combinations` | `True` | bool | Cross terms such as R·G. `False` only for a polynomial `linear` model. |
| `fit.pls.components` | `0` (automatic) | 0–10000 | PLS components. Root or hybrid PLS needs a positive value. |
| `fit.pls.tol` | `1e-06` | > 0 | PLS convergence tolerance. |
| `fit.nn.hidden_layers` | `[100]` | 1–4 layers of 4–512 | Network shape. |
| `fit.nn.max_iterations` | `10000` | ≥ 1 | Training iterations. |
| `fit.nn.patience` | `1500` | ≥ 1 | Stop after this many iterations without improvement. |
| `fit.nn.learning_rate` | `0.001` | 1e-8–100 | Optimizer step size. |
| `fit.nn.optim_type` | `"Adam"` | `Adam`, `SGD`, `RMSprop` | Optimizer. |
| `fit.nn.dropout_rate` | `0.0` | 0–0.8 | Dropout between layers. |
| `fit.nn.use_batch_norm` | `False` | bool | Batch normalization. |
| `fit.nn.batch_size` | automatic | 1–200 | Training batch size. |
| `fit.nn.tol` | `1e-06` | > 0 | Improvement threshold for early stopping. |
| `apply.use_lut` | `True` (`False` for high precision) | bool | Apply CC through a 3-D lookup table, which is much faster on large images. |
| `apply.grid_size` | `33` | 2–129 | Lookup table points per axis. |
| `apply.lazy_lut` | `True` | bool | Build the table on first use rather than at fit time. |
| `apply.min_pixels` | `1` | ≥ 1 | Use the table only for images at least this large. |
| `validation.enabled` | `True` | bool | Report 4-fold held-out error on the chart. |
| `validation.n_splits` | `4` | ≥ 2 | Number of folds. |
| `validation.random_state` | `42` | 0–2³²−1 | Fold shuffling seed. |

CC training always runs on the CPU with a fixed seed, so results are
reproducible. `fit.device` and `fit.random_state` exist only for compatibility
and accept only `"cpu"` and `42`.

### Shared groups

| Group | Settings | Default | Meaning |
| --- | --- | --- | --- |
| `input` | `frame_index` | none (single-frame files only) | Which frame of a multi-frame file to read. |
| | `apply_orientation` | `True` | Rotate by the EXIF orientation tag. |
| | `icc_policy` | `"reject"` | `"reject"`, `"assume-declared"` or `"convert-srgb"` for images with an embedded color profile. |
| | `alpha_policy` | `"reject"` | `"reject"`, `"drop"` or `"composite"` (onto `alpha_background`) for images with transparency. |
| | `grayscale_policy` | `"reject"` | `"reject"` or `"replicate"` for single-channel images. |
| | `max_pixels`, `max_decoded_bytes`, `max_file_bytes` | built-in limits | Refuse images larger than this. |
| `chart` | `device` | `"auto"` | `"auto"` (GPU, then CPU) or `"cpu"` for chart detection. |
| | `samples_per_axis` | `9` | Grid points per axis inside each patch. |
| | `samples_per_patch`, `sampling_seed` | not set, `0` | Exact random points per patch instead of the grid. |
| | `confidence`, `iou`, `max_detections` | `0.15`, `0.7`, `300` | Detector thresholds. See [chart detection](chart-detection.md). |
| | `sampling_inset` | `0.75` | Share of each patch that is sampled, avoiding its edges. |
| `numerics` | `precision` | `"auto"` | `"auto"`, `"native-byte"` (8-bit sRGB only) or `"high"` (float64). |
| `color` | `input_encoding`, `output_encoding` | `"srgb"` | `"srgb"` or `"linear-srgb"`. |
| | `output_policy` | `"clip"` | `"clip"`, `"preserve"` or `"raise"` for values outside 0–1. |
| `reference` | a `ReferenceProfile` | ColorChecker Classic 24 | Target values for the chart patches. |
| `observations` | `saturation` | `"report"` | Clipped chart patches are reported in the metrics, not repaired. |
| | `max_clipped_fraction` | `0.0` | Clipped-sample share above which a patch is flagged. |
| `persistence` | `enabled` | `False` | Save the fitted models during calibration. |
| | `directory` | none | Where to save. |
| | `format` | `"pickle"` | Use `"bundle"` (`.ccpb`): portable and safe to load. |

## Rules worth knowing

- **Settings apply to new calibrations only.** A loaded `.ccpb` bundle keeps
  the settings it was fitted with, so its output never changes.
- **Contradictions fail.** For example
  `Config(do_ffc=True, steps={"ffc": {"enabled": False}})` raises instead of
  picking one.
- **FFC needs matching sizes.** The chart, white and scene images must have the
  same dimensions when FFC is on.
- **Older argument styles still work**, for example
  `Config(model="rpcc3", do_ffc=False, CC_kwargs={"lut_grid_size": 33})`. New
  code should prefer the grouped form above.
- To see what a fit actually used, read `pipeline.numerical_report` after
  fitting.

See also: [calibration bundles](calibration-bundles.md),
[batch processing](batch-and-reporting.md),
[image reading and export](image-io-policies.md), and the
[fully customizable example](../ColorCorrectionPipeline/examples/customizable.py).
