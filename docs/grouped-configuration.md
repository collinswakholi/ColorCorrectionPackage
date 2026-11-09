# Grouped configuration

The grouped API organizes a new calibration by the step that owns each setting.
It is additive: existing `Config` constructors, kwargs dictionaries and saved
calibrations keep their numerical behavior. Grouping does not select a different
solver, change a reference, or create a new bundle format.

This API requires ColorCorrectionPipeline 2.x. The installed distribution and
Python import are both `ColorCorrectionPipeline`. Use `--upgrade` when updating
an existing 1.x installation:

```shell
python -m pip install --upgrade "ColorCorrectionPipeline>=2,<3"
python -m pip install --upgrade "ColorCorrectionPipeline[performance]>=2,<3"
```

The optional performance extra enables qualified CPU kernels; it does not
select the chart detector's GPU provider or change the fitting recipe.

## Configure the steps

```python
from ColorCorrectionPipeline import (
    Config, Steps, FFCOptions, GCOptions, WBOptions, CCOptions,
)

config = Config(
    input={"frame_index": 0},
    chart={"device": "auto", "allow_cpu_fallback": True},
    numerics={"precision": "auto"},
    steps=Steps(
        ffc=FFCOptions(
            enabled=True,
            fit={"method": "linear", "degree": 1, "bins": 30, "smooth_window": 7},
            # Omit white_plane.crop_rect to detect the white plane automatically.
        ),
        gc=GCOptions(
            enabled=True,
            quality={"min_neutrals": 3, "min_span": 20.0},
        ),
        wb=WBOptions(
            enabled=True,
            fit={"headroom": 0.95},
            quality={"min_signal": 0.01, "max_gain": 8.0},
        ),
        cc=CCOptions(
            enabled=True,
            fit={"model": "rpcc3"},
            apply={"grid_size": 33},
            validation={"enabled": True, "n_splits": 4, "random_state": 42},
        ),
    ),
    persistence={"enabled": True, "directory": "results", "format": "bundle"},
)
config.validate()
```

The option containers are frozen and own their supplied mappings. The named
containers expose the four step names; their `fit`, `quality`, `reporting`,
`apply` and `validation` members are mappings. Read a leaf with indexing, such as
`config.steps.wb.fit["headroom"]`. A plain mapping is also accepted at each
grouped boundary:

```python
config = Config(
    steps={
        "ffc": {"enabled": False},
        "cc": {"fit": {"model": "3x3"}},
    },
)
```

Omitted settings inherit the same modern recipe defaults as an equivalent flat
configuration. In particular, new construction defaults to UI-native-family
RPCC2, report-only saturation and automatic precision. FFC remains enabled by
default and needs a matching white image. Disable it explicitly when none is
available. Enabled FFC still requires compatible chart, white and scene spatial
dimensions.

These groups configure the next calibration. They do not refit a loaded model or
override its stored curve, matrix, feature order, precision or execution recipe.
The grouped snapshot describes the requested configuration. Inspect
`numerical_report` and the fitted model's effective settings for defaults and
runtime decisions resolved during calibration.

## Step ownership and control names

| Group | Accepted controls | Meaning |
| --- | --- | --- |
| `steps.ffc.enabled` | Boolean | Fit/use the white-field correction. |
| `steps.ffc.fit` | `method`, `degree`, `bins`, `smooth_window` | White-field surface estimator and sampling. Native UI fitting supports linear and polynomial surfaces; linear has effective degree one. FFC degree is unrelated to CC feature degree. |
| `steps.ffc.white_plane` | `model_path`, `manual_crop`, `crop_rect` | White-plane detection/cropping, separate from chart detection. A crop is `[x1, y1, x2, y2]` in white-image pixels. |
| `steps.ffc.quality` | `safe_mode`, `missing_plane`, `min_signal`, `max_saturated_fraction`, `min_crop_fraction`, `min_gain`, `max_gain` | Existing white-plane and gain safeguards. Modern fitting requires safe mode. |
| `steps.ffc.reporting` | `show`, `get_deltaE` | Existing plotting/reporting controls, subject to the active backend's restrictions. |
| `steps.gc.enabled` | Boolean | Fit/apply the neutral-lightness correction. |
| `steps.gc.quality` | `min_neutrals`, `min_span`, `duplicate_tolerance`, `target_tolerance` | Neutral-count/span, duplicate grouping and contradiction guards. These are not a gamma exponent or a polynomial-degree control. |
| `steps.gc.reporting` | `show`, `get_deltaE` | Existing reporting controls. Modern interactive `show=True` is unsupported. |
| `steps.wb.enabled` | Boolean | Fit/apply white balance. |
| `steps.wb.fit` | `headroom` | The multiplier used in fitted neutral channel gains; default 0.95. |
| `steps.wb.quality` | `min_signal`, `max_gain` | Neutral eligibility and maximum positive-gain safeguards. |
| `steps.wb.reporting` | `show`, `get_deltaE` | Existing reporting controls. Modern interactive `show=True` is unsupported. |
| `steps.cc.enabled` | Boolean | Fit/apply color correction. |
| `steps.cc.fit` | `model`, `basis`, `degree`, `include_combinations`, `device`, `random_state`, `nn`, `pls` | Model/features and solver-specific controls. Native-family training remains deterministic CPU fitting with its existing seed constraints. |
| `steps.cc.apply` | `use_lut`, `lazy_lut`, `grid_size`, `min_pixels` | Stored CC evaluator settings, separate from training. High precision retains its existing direct-evaluation/LUT qualification rules. |
| `steps.cc.validation` | `enabled`, `n_splits`, `random_state` | Grouped held-out **CC** validation conditional on already fitted upstream stages. It is not independent-capture or whole-pipeline validation. |
| `steps.cc.reporting` | `show`, `get_deltaE` | Existing reporting controls. Modern interactive `show=True` is unsupported. |

GC uses the existing monotonic neutral-lightness model with fixed endpoint
behavior. WB uses the existing diagonal neutral-ratio algorithm. Grouping
exposes their working guards and headroom without adding manual gamma,
temperature or channel-gain fitting algorithms. See [numerical policies and
validation](numerical-policies-and-validation.md) for their interpretation.

### Select CC features and a solver

Fixed `3x3`/`ccm`, `rpcc2` and `rpcc3` presets keep their fixed feature bases.
For `linear`, `pls` and `nn`, select `polynomial`, `root` or `hybrid` features
through degree four. The [migration guide](migration-2.md#model-and-basis-selection)
describes feature counts, intercepts and historical limits.

```python
root_pls = Config(
    steps=Steps(cc=CCOptions(fit={
        "model": "pls", "basis": "root", "degree": 3,
        "pls": {"components": 6, "tol": 1e-6},
    })),
)

hybrid_nn = Config(
    steps=Steps(cc=CCOptions(fit={
        "model": "nn", "basis": "hybrid", "degree": 4,
        "nn": {
            "hidden_layers": [100], "max_iterations": 10000,
            "patience": 1500, "learning_rate": 0.001, "tol": 1e-6,
            "optim_type": "Adam", "dropout_rate": 0.0,
            "use_batch_norm": False,
        },
    })),
)
```

`fit.nn` accepts `hidden_layers`, `max_iterations`, `patience`, `learning_rate`,
`tol`, `batch_size`, `optim_type`, `dropout_rate`, `use_batch_norm` and `verbose`.
`fit.pls` accepts `components`, `tol` and `max_iterations`. These remain
method-specific: NN controls must not be passed to a matrix/PLS fit, and PLS
controls must not be passed to NN. Existing native limits and fixed effective
settings still apply. New root/hybrid PLS fits require a positive component
budget. An iteration cap does not guarantee that early stopping will exhaust it.

The grouped form rejects unknown group/control names. Existing validation remains
responsible for numerical bounds and cross-step invariants, including reference
layout, native sampling, enabled FFC, precision and LUT compatibility. The API
does not make an unsupported backend or solver combination valid.

## Shared settings stay outside the steps

| `Config` group | Meaning / existing equivalent |
| --- | --- |
| `input` | Decode/frame/orientation/ICC/alpha/grayscale/allocation settings; aliases `image_options`. Uses those existing field names. |
| `chart` | Chart model, selection, sampling and provider settings; aliases `chart_detection`. Names include `sampling_inset`, `samples_per_axis`, `device`, `allow_cpu_fallback` and `cpu_threads`. |
| `reference` | A validated `ReferenceProfile` or its canonical dictionary; aliases `reference_profile`. Physical reference construction still uses `ReferenceProfile.from_values`. |
| `color` | Partial `ColorSemantics` controls, merged with the selected modern recipe; aliases `color_semantics`. |
| `numerics` | `precision`: `auto`, `native-byte` or `high`. Precision affects original measurements, FFC and stage boundaries, so it is not only a CC setting. |
| `observations` | `saturation`, `max_clipped_fraction`. UI-family fits retain report-only saturation; the clipping threshold affects diagnostics rather than inventing recovered measurements. |
| `persistence` | `enabled`, `directory`, `format`; aliases `save`, `save_path`, `model_format`. This saves models/metrics during calibration, not scene image files. |

Chart inference device, color-fitting device and optional image kernels are
different capabilities. In particular, an eight-thread OpenMP/BLAS environment
does not override the chart detector's explicit `chart.cpu_threads` setting.
See [chart detection](chart-detection.md), [image I/O](image-io-policies.md), and
[color semantics](color-semantics.md) for the unchanged underlying contracts.

Image export remains an explicit `write_image_data` call or a batch operation.
Declare dtype, transfer conversion, precision-loss consent and overwrite there.
The existing prepared `ExecutionOptions` API also remains separate and retains
its runtime restrictions. There are no inactive `Config.export` or
`Config.execution` wrappers.

The model format still defaults to `pickle` for compatibility. New examples
explicitly select `bundle`; `save_bundle`/`load_bundle` remain the portable,
validated inference path. Grouping does not weaken strict image readback,
atomic publication, failed-fit rollback or bundle validation.

## Aliases, ownership and mutation

Old forms remain supported:

```python
flat = Config(
    model="rpcc3",
    do_ffc=False,
    stage_policy={"version": "modern-v1", "saturation": "report", "wb_headroom": 0.95},
    CC_kwargs={"use_lut": True, "lut_grid_size": 33},
)
```

Equal duplicate aliases are accepted. Contradictions fail instead of silently
choosing a winner, including when an explicitly supplied flat flag equals its
historical default. Numeric duplicates must match both value and type: `True`
does not equal integer `1`, and integer `2` does not equal float `2.0` for this
check. Equivalent `Path` and string paths are accepted. The original explicit
flat value keeps its type for normal validation:

```python
# Raises: two explicit requests disagree about the same FFC switch.
Config(do_ffc=True, steps=Steps(ffc=FFCOptions(enabled=False)))
```

The existing mutable flat attributes remain the internal owner. `config.steps`
and the shared mapping accessors return fresh immutable views of their current
state; `config.reference` exposes the owned immutable `ReferenceProfile`.
The object does not keep a second mutable grouped copy. For example,
changing `config.CC_kwargs["model"]` is visible in the next grouped snapshot.
An already obtained snapshot does not change. Build another option object or
modify the existing supported flat attributes for another fit; mutating a
returned read-only mapping is not supported.

## Serialize without reinterpreting older settings

`to_dict()` keeps its existing flat result. It is still used by runtime and
saved-calibration compatibility paths. For an explicitly grouped JSON document:

```python
import json
from pathlib import Path

document = config.to_grouped_dict()
assert document["layout"] == "grouped-v1"
assert document["api_version"] == 2
Path("calibration.json").write_text(json.dumps(document, indent=2), encoding="utf8")
restored = Config.from_dict(json.loads(Path("calibration.json").read_text(encoding="utf8")))
```

`layout="grouped-v1"` versions the **settings layout**. `api_version=2` versions
the **configuration interpretation**. Neither changes the recorded numerical
runtime (`native-v1` for the byte profile) or bundle schema. New mappings can
also select API 2 deliberately with `Config.from_dict(mapping, source_version=2)`.
Recorded policy identities and supported flat compatibility fields without a
grouped alias remain explicit in that document; presentation does not discard
them merely to make the JSON tree look simpler.

An old unversioned flat dictionary still takes the API-1 compatibility route.
Do not add a new marker merely to relabel an old experiment. Historical ML and
conventional trainers remain available through their original flat settings and
explicit policies:

```python
historical = Config(
    fitting_policy="legacy-v1",
    CC_kwargs={"mtd": "nn", "hidden_layers": [64], "max_iterations": 1000},
)
historical_document = historical.to_dict()
```

The grouped API is for modern API-2 UI-family configurations. API-1/historical
configurations must retain their flat form; `to_grouped_dict()` rejects them
rather than silently translating a historical sklearn MLP into the native
residual NN. Loading an existing calibration remains the way to reproduce its
stored predictions without retraining.

## Run the complete example

The installed [customizable example](../ColorCorrectionPipeline/examples/customizable.py)
provides grouped calibration settings, optional physical reference input,
multiple scenes, safe bundle replay, a PNG preview and float64 TIFF export:

```shell
python -m ColorCorrectionPipeline.examples.customizable --write-settings settings.json
python -m ColorCorrectionPipeline.examples.customizable --settings settings.json --output my-run
```

Top-level workflow settings override the example defaults. A supplied
`calibration` dictionary or `exports` list replaces that entire section, without
a recursive merge. Existing flat calibration sections are still accepted.
The sample starter uses an explicit `grouped-v1` marker.

Capture paths, `output_directory` and the calibration persistence directory are
relative to the JSON file. `--output` is relative to the working directory.
Nested detector/model paths keep their normal working-directory semantics.
Each output directory must be new. See the module's `--help` and settings
dictionary before choosing capture-specific controls.
