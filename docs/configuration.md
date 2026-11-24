# Configuration guide

A calibration runs up to four stages, always in this order:

| Stage | Fixes | Learns from |
| --- | --- | --- |
| **FFC**, flat-field correction | Uneven lighting across the frame: vignetting, hot spots | A photo of a plain white target, same camera and lights |
| **GC**, gamma correction | Tone: how measured greys map to true lightness | The chart's six grey patches |
| **WB**, white balance | The color cast of the light | The chart's grey patches |
| **CC**, color correction | All remaining color error | All 24 chart patches |

All four are on by default, and the defaults suit most ColorChecker Classic 24
photographs. **You only write the settings you want to change**; everything
else keeps the default shown below.

**Contents:** [the settings map](#the-settings-map) ·
[three ways to configure](#three-ways-to-configure) ·
[FFC](#ffc-flat-field-correction) · [GC](#gc-gamma-correction) ·
[WB](#wb-white-balance) · [CC](#cc-color-correction) ·
[shared settings](#shared-settings) · [chart reference values](#chart-reference-values) ·
[recipes](#recipes) ·
[rules](#rules-worth-knowing) · [older argument names](#older-argument-names)

## The settings map

Every setting, with its **default value** and, in the comment, its **choices**.
This is a working configuration: running it gives exactly the default behaviour.

```python
from ColorCorrectionPipeline import Config

config = Config(
    steps={
        "ffc": {                                  # Flat-field correction
            "enabled": True,                      # True | False
            "fit": {
                "method": "poly",                 # "poly" (smooth surface) | "linear" (tilted plane)
                "degree": 5,                      # 1–8, used by "poly" only
                "bins": 50,                       # 2–512 sampling cells per axis
                "smooth_window": 5,               # odd number ≥ 1, in cells
            },
            "apply": {"domain": "lightness"},     # "lightness" (L* only) | "linear" (RGB, fixes color too)
            "white_plane": {
                # "crop_rect": [x1, y1, x2, y2],  # fixed white region; default: detect it
            },
            "quality": {
                "missing_plane": "error",         # "error" | "full-image" when no white target is found
                "min_signal": 0.02,               # 0–1: reject a white image darker than this
                "max_saturated_fraction": 0.01,   # 0–1: reject if more pixels than this are clipped
                "min_crop_fraction": 0.1,         # 0–1: reject a white region smaller than this
                "min_gain": 0.5,                  # reject corrections below this gain…
                "max_gain": 1.8,                  # …or above this one
            },
        },
        "gc": {                                   # Gamma correction
            "enabled": True,
            "quality": {
                "min_neutrals": 3,                # 3–6: fewest usable grey patches
                "min_span": 20.0,                 # 0–100: smallest L* range the greys must cover
                "duplicate_tolerance": 1e-06,     # greys closer than this are merged
                "target_tolerance": 1.0,          # merged greys whose targets differ more fail
            },
        },
        "wb": {                                   # White balance
            "enabled": True,
            "fit": {"headroom": 0.95},            # 0–1: scale gains down to protect highlights
            "quality": {
                "min_signal": 0.01,               # ignore grey patches darker than this
                "max_gain": 8.0,                  # up to 100: fail if a channel needs more gain
            },
        },
        "cc": {                                   # Color correction
            "enabled": True,
            "fit": {"model": "rpcc2"},            # "ccm" | "rpcc2" | "rpcc3" | "linear" | "pls" | "nn"
                                                  # (+ "basis"/"degree" and "pls"/"nn" options, see CC)
            "apply": {
                "use_lut": True,                  # True: 3-D lookup table (fast) | False: exact model
                "grid_size": 33,                  # 2–129 table points per axis
                "lazy_lut": True,                 # build the table on first use
                "min_pixels": 1,                  # use the table for images at least this large
            },
            "validation": {
                "enabled": True,                  # held-out check on the chart (report only)
                "n_splits": 4,                    # ≥ 2 folds
                "random_state": 42,               # fold shuffling seed
            },
        },
    },
    input={
        "frame_index": None,                      # None: single-frame files only | 0, 1, …: pick a frame
        "apply_orientation": True,                # honour the EXIF rotation tag
        "icc_policy": "reject",                   # "reject" | "assume-declared" | "convert-srgb"
        "alpha_policy": "reject",                 # "reject" | "drop" | "composite"
        "grayscale_policy": "reject",             # "reject" | "replicate"
    },
    chart={
        "type": "auto",                           # "auto" | "classic" | "spyder" | "digital-sg"
        "device": "auto",                         # "auto" (GPU, else CPU) | "cpu"
        "samples_per_axis": 9,                    # grid points per axis in each patch (9 × 9 = 81)
        # "samples_per_patch": 20,                # OR: exact random points per patch (1–4096)…
        # "sampling_seed": 0,                     # …with this seed
    },
    numerics={"precision": "auto"},               # "auto" | "native-byte" | "high"
    color={
        "input_encoding": "srgb",                 # "srgb" | "linear-srgb"
        "output_encoding": "srgb",                # "srgb" | "linear-srgb"
        "output_policy": "clip",                  # "clip" | "preserve" | "raise" for values outside 0–1
    },
    persistence={
        "enabled": False,                         # save models while fitting
        "directory": None,                        # where to save
        "format": "pickle",                       # "bundle" (.ccpb, recommended) | "pickle"
    },
    # reference=ReferenceProfile(...)             # default: ColorChecker Classic 24 values
)
```

## Three ways to configure

**1. A preset name.** Enough for most work.

```python
from ColorCorrectionPipeline import calibrate, correct

cal = calibrate("chart.jpg", white="white.jpg", model="rpcc2")
image = correct("scene.jpg", cal)          # float RGB, 0–1
```

| Preset | Same as | Use when |
| --- | --- | --- |
| `rpcc2` (default) | `cc.fit.model = "rpcc2"` | General use. Robust to exposure changes. |
| `rpcc3` | `cc.fit.model = "rpcc3"` | A well-exposed chart and you want a closer fit. |
| `ccm` or `3x3` | `cc.fit.model = "ccm"` | You want a simple 3×3 matrix. |
| `pls` | `cc.fit.model = "pls"` | Many correlated features. |
| `high-precision` | `rpcc2` + `numerics.precision = "high"` | Scenes are 16-bit or float. |

**2. A preset plus a few changes.** Any setting from the map can be added:

```python
from ColorCorrectionPipeline import Config, calibrate

config = Config.from_preset("rpcc3", steps={"cc": {"validation": {"enabled": False}}})
cal = calibrate("chart.jpg", white="white.jpg", config=config)
```

**3. A `Config` with only what you change.** Dictionaries and the typed classes
(`Steps`, `FFCOptions`, `GCOptions`, `WBOptions`, `CCOptions`) are interchangeable:

```python
from ColorCorrectionPipeline import CCOptions, Config, FFCOptions, Steps

config = Config(
    steps=Steps(ffc=FFCOptions(fit={"method": "linear"}), cc=CCOptions(fit={"model": "rpcc3"})),
    input={"frame_index": 0},
)
same = Config(
    steps={"ffc": {"fit": {"method": "linear"}}, "cc": {"fit": {"model": "rpcc3"}}},
    input={"frame_index": 0},
)
```

A misspelt setting name raises an error rather than being ignored.

> **Camera JPEGs and `frame_index`.** Many cameras store a small preview as a
> second frame inside the JPEG. `calibrate()` and `Config.from_preset()` read
> frame 0 for you. A plain `Config()` keeps the strict default (`None`), so add
> `input={"frame_index": 0}` if your photos are rejected as multi-frame.

## FFC: flat-field correction

**What it does.** Measures how brightness falls off across a photo of a plain
white target, then divides that falloff out of every image.
**Turn it off** when you have no white photo: `calibrate()` does this
automatically when you leave out `white=`. With a `Config`, set
`steps={"ffc": {"enabled": False}}`.

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `fit.method` | `"poly"`: a smooth polynomial surface. Follows curved falloff such as lens vignetting. <br>`"linear"`: a tilted plane. Use for a simple left-to-right or top-to-bottom gradient. | `"poly"` | Shape of the lighting model. |
| `fit.degree` | 1–8. Higher bends more. Only `"poly"` uses it. | `5` | Flexibility of the surface. Lower it if the correction looks wavy. |
| `fit.bins` | 2–512, and at least `degree + 1` | `50` | How many cells per axis the white image is averaged into before fitting. |
| `fit.smooth_window` | odd numbers: 1, 3, 5, … | `5` | Smooths those cells before fitting. Raise it for a noisy or textured white target. |
| `apply.domain` | `"lightness"`: scales Lab L\* only. <br>`"linear"`: multiplies linear RGB, so it also removes the color shift in dark corners. | `"lightness"` | Where the correction is applied. `"linear"` is usually better for strong vignetting. |
| `white_plane.crop_rect` | `[x1, y1, x2, y2]` in white-image pixels | detected | Use a fixed region instead of finding the white target automatically. |
| `quality.missing_plane` | `"error"`: stop if no white target is found. <br>`"full-image"`: use the whole frame. | `"error"` | What happens when detection finds nothing. |
| `white_plane.model_path` | path to an ONNX model | bundled | A custom white-target detector. |
| `quality.min_signal` | 0–1 | `0.02` | Rejects a white photo that is too dark. |
| `quality.max_saturated_fraction` | 0–1 | `0.01` | Rejects a white photo with more than 1% clipped pixels. |
| `quality.min_crop_fraction` | 0–1 | `0.1` | Rejects a detected white region smaller than 10% of the frame. |
| `quality.min_gain`, `quality.max_gain` | positive numbers | `0.5`, `1.8` | Rejects a correction that would darken or brighten any spot beyond this. |

```python
from ColorCorrectionPipeline import Config

gentle_ffc = Config(steps={"ffc": {
    "fit": {"method": "linear"},          # mild, even gradient
    "apply": {"domain": "linear"},        # correct color in the corners too
}})
fixed_region = Config(steps={"ffc": {"white_plane": {"crop_rect": [40, 40, 1100, 720]}}})
```

The white photo, the chart photo and every scene must have **the same pixel
dimensions** while FFC is on.

## GC: gamma correction

**What it does.** Fits a smooth curve through the six grey patches so that
measured lightness matches the reference. There is **no gamma exponent to set**:
the curve is learned. The settings below are safety checks that stop a bad fit.

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `enabled` | `True`, `False` | `True` | Run GC. |
| `quality.min_neutrals` | 3–6 | `3` | Fails if fewer grey patches are usable, for example when the darkest are clipped. |
| `quality.min_span` | 0–100 L\* units | `20.0` | Fails if the usable greys cover too narrow a lightness range. |
| `quality.duplicate_tolerance` | small number | `1e-06` | Greys measuring closer than this are treated as one. |
| `quality.target_tolerance` | L\* units | `1.0` | Fails if such merged greys have reference values further apart than this. |

## WB: white balance

**What it does.** Scales the red, green and blue channels so the grey patches
come out neutral.

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `enabled` | `True`, `False` | `True` | Run WB. |
| `fit.headroom` | above 0, up to 1 | `0.95` | Scales all gains down slightly so bright areas don't clip. Use `1.0` for no margin. |
| `quality.min_signal` | 0–1 | `0.01` | Ignores grey patches darker than this. |
| `quality.max_gain` | up to 100 | `8.0` | Fails if a channel would need more than 8× gain, which usually means a bad capture. |

## CC: color correction

**What it does.** Learns a mapping from the camera's colors to the chart's true
colors, using all 24 patches.

### Choosing the model (`fit.model`)

| Model | What it is | Features | Use when |
| --- | --- | --- | --- |
| `"ccm"` (alias `"3x3"`) | A 3×3 matrix | 3 | You want the simplest, most predictable correction. |
| `"rpcc2"` **(default)** | Root-polynomial, degree 2 | 6 | General use. Stays correct when exposure changes. |
| `"rpcc3"` | Root-polynomial, degree 3 | 13 | A clean, well-exposed chart and you want a closer fit. |
| `"linear"` | Least squares on the features you choose | see below | You want to pick the basis and degree yourself. |
| `"pls"` | Partial least squares on those features | see below | Many features that are correlated with each other. |
| `"nn"` | A small neural network | see below | Strongly non-linear cameras or lighting, as an experiment. |

`"linear"`, `"pls"` and `"nn"` also take a **basis** and a **degree**. They
decide how many terms the model fits from the 24 patches. Fewer terms generalise
better, and more than about 20 overfit the chart:

| `fit.basis` | What the terms are | Terms at degree 1 / 2 / 3 / 4 |
| --- | --- | --- |
| `"polynomial"` (default) | Products of R, G, B, such as R·G or B² | 3 / 9 / 19 / 34 |
| `"root"` | Roots of those products, such as √(R·G). Exposure-invariant. | 4 / 7 / 14 / 23 |
| `"hybrid"` | Both together | 4 / 13 / 30 / 54 |

`fit.degree` is 1–4 (default 2). `fit.include_combinations=False` drops the
cross terms (R·G and so on) from a polynomial `"linear"` model.

```python
from ColorCorrectionPipeline import Config

root_pls = Config(steps={"cc": {"fit": {
    "model": "pls", "basis": "root", "degree": 3,
    "pls": {"components": 6},       # root or hybrid PLS needs an explicit count
}}})
small_nn = Config(steps={"cc": {"fit": {
    "model": "nn", "degree": 2,
    "nn": {"hidden_layers": [64], "max_iterations": 3000},
}}})
```

### PLS options (`fit.pls`, only with `model="pls"`)

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `components` | 0–10000; `0` = choose automatically | `0` | Number of latent components. Must be > 0 for `"root"` or `"hybrid"`. |
| `tol` | > 0 | `1e-06` | Convergence tolerance. |

### Neural-network options (`fit.nn`, only with `model="nn"`)

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `hidden_layers` | 1–4 layers, each 4–512 wide | `[100]` | Network size. Smaller is less likely to overfit 24 patches. |
| `max_iterations` | ≥ 1 | `10000` | Upper limit on training steps. |
| `patience` | ≥ 1 | `1500` | Stops early after this many steps without improvement. |
| `learning_rate` | 1e-8–100 | `0.001` | Optimizer step size. |
| `optim_type` | `"Adam"`, `"SGD"`, `"RMSprop"` | `"Adam"` | Optimizer. |
| `dropout_rate` | 0–0.8 | `0.0` | Randomly drops units while training, to reduce overfitting. |
| `use_batch_norm` | `True`, `False` | `False` | Batch normalization between layers. |
| `batch_size` | 1–200 | automatic | Training batch size. |
| `tol` | > 0 | `1e-06` | Smallest change that counts as improvement. |

### Applying the model (`apply`) and checking it (`validation`)

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `apply.use_lut` | `True`: a 3-D lookup table, much faster on large images. <br>`False`: evaluate the model on every pixel. | `True` (`False` with `precision="high"`) | Speed against exactness. The table differs from the exact model by a tiny interpolation error. |
| `apply.grid_size` | 2–129 | `33` | Lookup-table points per axis. Larger is closer to exact but slower to build. |
| `apply.lazy_lut` | `True`, `False` | `True` | Build the table on first use rather than straight after fitting. |
| `apply.min_pixels` | ≥ 1 | `1` | Images smaller than this skip the table. |
| `validation.enabled` | `True`, `False` | `True` | Refits on part of the chart and reports the error on the held-out patches. It **does not change the model**. Turn it off to calibrate faster. |
| `validation.n_splits` | ≥ 2 | `4` | Number of folds. |
| `validation.random_state` | 0 – 2³²−1 | `42` | Seed for splitting the patches into folds. |

## Shared settings

### `input`: reading image files

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `frame_index` | `None`, or `0`, `1`, … | `None` | `None` accepts single-frame files only; a number picks that frame. Use `0` for camera JPEGs. |
| `apply_orientation` | `True`, `False` | `True` | Rotates images by their EXIF orientation tag. |
| `icc_policy` | `"reject"`: refuse images with an embedded color profile. <br>`"assume-declared"`: trust the declared encoding. <br>`"convert-srgb"`: convert to sRGB first. | `"reject"` | Images with an embedded ICC color profile. |
| `alpha_policy` | `"reject"`, `"drop"` (ignore transparency), `"composite"` (blend onto `alpha_background`) | `"reject"` | Images with an alpha channel. |
| `alpha_background` | three numbers 0–1 | none | Background color for `"composite"`. |
| `grayscale_policy` | `"reject"`, `"replicate"` (copy the grey channel to R, G and B) | `"reject"` | Single-channel images. |
| `max_pixels`, `max_decoded_bytes`, `max_file_bytes` | positive integers | built-in limits | Refuses images above these sizes. |

### `chart`: finding and reading the chart

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `type` | `"auto"`: use the chart in the photo; a Classic wins if there are several. <br>`"classic"` (or `"passport"`), `"spyder"`, `"digital-sg"`: use that chart. | `"auto"` | Which chart to calibrate from. A custom `reference` decides this for you. |
| `device` | `"auto"`: try a GPU, fall back to CPU. <br>`"cpu"`: CPU only. | `"auto"` | Where chart detection runs. Fitting always runs on the CPU. |
| `samples_per_axis` | positive integer | `9` | Reads each patch on a fixed grid of N × N points. |
| `samples_per_patch` | 1–4096 | not set | Reads exactly this many random points per patch instead of the grid. Don't combine with `samples_per_axis`. |
| `sampling_seed` | 0 – 2³²−1 | `0` | Seed for `samples_per_patch`. |
| `confidence` | 0–1 | `0.15` | Minimum detection confidence. Lower it if a dim chart is missed. |
| `iou`, `max_detections` | 0–1, ≥ 1 | `0.7`, `300` | Detector overlap threshold and candidate limit. |
| `sampling_inset` | 0–1 | `0.75` | Share of each patch that is sampled, which keeps away from patch edges. |
| `chart_index` | 0, 1, … | `0` | Which chart to use when the photo contains several. |

See [chart detection](chart-detection.md) and [GPU selection](gpu-detection.md)
for models and devices.

### `numerics`, `color`, `reference`, `persistence`

| Setting | Choices | Default | What it changes |
| --- | --- | --- | --- |
| `numerics.precision` | `"auto"`: decide from the calibration images. <br>`"native-byte"`: 8-bit sRGB arithmetic, fastest. <br>`"high"`: float64 throughout. | `"auto"` | Arithmetic precision. Use `"high"` when an 8-bit chart will later correct 16-bit or float scenes. |
| `color.input_encoding` | `"srgb"`, `"linear-srgb"` | `"srgb"` | How the pixels of your input images are encoded. |
| `color.output_encoding` | `"srgb"`, `"linear-srgb"` | `"srgb"` | The encoding of the corrected output. |
| `color.output_policy` | `"clip"` to 0–1, `"preserve"` out-of-range values, `"raise"` an error | `"clip"` | What happens to corrected values outside 0–1. |
| `reference` | a `ReferenceProfile` | built-in values for the detected chart | Target values for the chart's patches. See [chart reference values](#chart-reference-values). |
| `persistence.enabled` | `True`, `False` | `False` | Save the fitted models while calibrating. |
| `persistence.directory` | a folder | none | Where to save them. |
| `persistence.format` | `"bundle"`: portable `.ccpb`, safe to load. <br>`"pickle"`: Python pickle; load only files you trust. | `"pickle"` | File format. **Choose `"bundle"`**; `"pickle"` remains the default only for compatibility. |

### Settings that exist but are fixed in 3.0

These appear in saved settings files. Changing them raises an error that
explains why:

| Setting | Fixed value | Why |
| --- | --- | --- |
| `steps.cc.fit.device`, `steps.cc.fit.random_state` | `"cpu"`, `42` | Training is deterministic CPU code, so results are reproducible. |
| `steps.ffc.quality.safe_mode` | `True` | The white-image checks above always run. |
| `observations.saturation` | `"report"` | Clipped chart patches are reported in the metrics, not repaired or dropped. `observations.max_clipped_fraction` (default `0.0`) only sets when a patch is flagged. |
| `steps.*.reporting.show` | `False` | Interactive plots during fitting are not supported. |

## Chart reference values

A calibration compares each measured patch with its **reference value**, the
patch's true colour. GC and WB also need to know which patches are **grey**.

### Built-in charts

| Chart | `chart` name | Cells | Grey patches used by GC and WB | Source of values |
| --- | --- | --- | --- | --- |
| ColorChecker Classic 24 | `"classic"` (or `"passport"`) | 4 × 6 | Bottom row, white to black (`r4c1`–`r4c6`) | Post-November-2014 values |
| SpyderCHECKR 24 | `"spyder"` | 4 × 6 | Datacolor's grey row A1–A6, the bottom row (`r4c1`–`r4c6`) | darktable "after 2018" V2 table |
| ColorChecker Digital SG | `"digital-sg"` | 10 × 14 | The central grey ramps E5–J5 and E6–J6 (`r5c5`–`r5c10`, `r6c5`–`r6c10`) | BabelColor "After Nov 2014" |

All values are CIE Lab D50 and match ColorCorrect Studio's audited catalog. They
are published measurements, not manufacturer-certified ones. Digital SG's many
white, grey and black **border** patches are evenness checks, so GC and WB never
use them; the colour fit uses all 140 cells.

`ReferenceProfile.builtin("spyder")` returns a built-in reference if you want to
inspect it.

> **Spyder and Digital SG support is new** and has been verified on synthetic
> charts photographed through a simulated camera, not yet on real photographs.
> Each fit records `last_run["chart"]` and logs a notice. For the most conservative
> result on these charts, turn GC and WB off:
> `steps={"gc": {"enabled": False}, "wb": {"enabled": False}}`.

### Your own measured values

If you measured your chart with a spectrophotometer, or have a supplier's batch
values, use them instead. The chart's layout, cell ids and grey patches are filled
in for you:

```python
from ColorCorrectionPipeline import Config, ReferenceProfile

reference = ReferenceProfile.for_chart("spyder", measured_lab)     # 24 Lab D50 triplets
config = Config(reference=reference)
```

A custom reference decides which chart the calibration looks for, so
`chart={"type": ...}` isn't needed. Options:

| Option | Choices | Default | Meaning |
| --- | --- | --- | --- |
| `chart` | `"classic"`, `"passport"`, `"spyder"`, `"digital-sg"` | `"classic"` | Which chart the values belong to: 24 values, or 140 for Digital SG. |
| `color_space` | `"lab"`, `"xyz"`, `"srgb"`, `"linear-srgb"` | `"lab"` | What the numbers are. |
| `white` | `"D50"`, `"D55"`, `"D60"`, `"D65"`, `"D75"`, `"A"`, `"C"`, `"E"`, or an `(x, y)` pair | `"D50"` | The values' white point. sRGB values use `"D65"`. |
| `scale` | Lab `1`; XYZ `1` or `100`; RGB `1`, `255` or `65535` | `1` | The numeric range. |
| `order` | `"row-major"` (left to right, then down) or `"column-major"` (top to bottom, then right) | `"row-major"` | How a plain list is ordered. |
| `neutral_ids` | cell ids such as `"r4c1"`, or patch names such as `"white"` | the chart's grey patches | Override the grey patches GC and WB use. |
| `name` | text | `"Custom reference"` (file name for CSVs) | A label stored with the calibration. |

**From a spreadsheet or CSV.** `ReferenceProfile.from_csv(path, chart=...)` and
`ReferenceProfile.from_text(text, chart=...)` accept comma-, tab- or
semicolon-separated rows, with one optional header row, in any of three forms:

```text
L,a,b                 cell,L,a,b             row,column,L,a,b
70.19,-31.85,1.98     A1,70.19,-31.85,1.98   1,1,70.19,-31.85,1.98
54.38,8.84,-25.71     B1,54.38,8.84,-25.71   1,2,54.38,8.84,-25.71
...                  ...                    ...
```

- **Plain triplets** are read in `order`.
- **`cell` labels** such as `A1` are grid positions: the letter is the column and
  the number the row, counted from the top-left of the chart as the detector
  sees it.
- **`row,column`** pairs are 1-based.
- Labelled rows may come in any order, but every cell must appear exactly once.

> **SpyderCHECKR labels.** Datacolor prints row A at the *bottom* of the chart, so
> its "A1" (white) is grid cell `A4`, not `A1`. Use plain triplets in grid order,
> or `row,column` pairs, when copying Datacolor's printed table.

The same formats work in ColorCorrect Studio's reference editor and Python SDK.

## Recipes

### Skip flat-field correction

```python
from ColorCorrectionPipeline import ColorCorrection, Config, calibrate

cal = calibrate("chart.jpg")                       # no white= means no FFC

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

### 16-bit or float scenes

```python
from ColorCorrectionPipeline import Config

config = Config(numerics={"precision": "high"})    # or Config.from_preset("high-precision")
```

### Faster calibration

```python
from ColorCorrectionPipeline import Config

config = Config(steps={"cc": {"validation": {"enabled": False}}})
```

### CPU-only chart detection

```python
from ColorCorrectionPipeline import Config, calibrate

config = Config(chart={"device": "cpu"}, input={"frame_index": 0})
cal = calibrate("chart.jpg", device="cpu")         # same, via the shortcut
```

### Reproducible random patch sampling

```python
from ColorCorrectionPipeline import Config

config = Config(chart={"samples_per_patch": 20, "sampling_seed": 42})
```

### Save the calibration automatically

```python
from ColorCorrectionPipeline import Config

config = Config(persistence={"enabled": True, "directory": "results", "format": "bundle"})
```

This writes `results/<name>_models.ccpb`. The name is `chart` with
`calibrate()`, or `name_=` with `ColorCorrection.fit()`. You can also skip this
and call `cal.save("calibration.ccpb")` afterwards.

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

The file records `"api_version": 2`. A dictionary without it is rejected,
because unversioned dictionaries used the format removed in 3.0.

## Rules worth knowing

- **Settings apply to new calibrations.** A loaded `.ccpb` bundle keeps the
  settings it was fitted with, so its output never changes.
- **Contradictions fail.** `Config(do_ffc=True, steps={"ffc": {"enabled": False}})`
  raises instead of picking one.
- **FFC needs matching sizes.** With FFC on, the chart, white and scene images
  must all have the same pixel dimensions.
- **To see what a fit actually used**, read `pipeline.numerical_report` after
  fitting. It records the resolved FFC surface, the greys GC used, the CC
  features and more.

## Older argument names

Code written for 2.x keeps working. Each older name maps to one grouped setting:

| Older form | Grouped setting |
| --- | --- |
| `Config(do_ffc=…)`, `do_gc`, `do_wb`, `do_cc` | `steps.<stage>.enabled` |
| `Config(model=…, basis=…, degree=…)` | `steps.cc.fit.model`, `.basis`, `.degree` |
| `Config(precision=…)` | `numerics.precision` |
| `FFC_kwargs={"fit_method": …, "degree": …, "bins": …, "smooth_window": …}` | `steps.ffc.fit.method`, `.degree`, `.bins`, `.smooth_window` |
| `FFC_kwargs={"crop_rect": …, "domain": …, "min_gain": …}` | `steps.ffc.white_plane.crop_rect`, `steps.ffc.apply.domain`, `steps.ffc.quality.min_gain` |
| `CC_kwargs={"lut_grid_size": …, "use_lut": …, "ncomp": …}` | `steps.cc.apply.grid_size`, `steps.cc.apply.use_lut`, `steps.cc.fit.pls.components` |
| `stage_policy={"wb_headroom": …, "gc_min_neutrals": …}` | `steps.wb.fit.headroom`, `steps.gc.quality.min_neutrals` |
| `validation_options={…}` | `steps.cc.validation` |
| `image_options={…}`, `chart_detection={…}` | `input`, `chart` |
| `save=…, save_path=…, model_format=…` | `persistence.enabled`, `.directory`, `.format` |

See also: [calibration bundles](calibration-bundles.md),
[batch processing](batch-and-reporting.md),
[reading and saving images](image-io-policies.md), and the
[fully customizable example](../ColorCorrectionPipeline/examples/customizable.py).
