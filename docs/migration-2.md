# Migrating to 2.0

New calibrations use the UI-native workflow with RPCC2 as the default. Select
the solver, feature basis and degree directly. The historical ours/conv
terminology is retired from the primary model menu. Existing ML and
conventional training can still run with their original numerical policies,
and supported calibration files retain their recorded predictions.

Grouped configuration is an additive way to express that same recipe. It places
step switches beside their controls without changing fitted numbers, existing
kwargs forms or saved-calibration interpretation. See the
[grouped configuration reference](grouped-configuration.md) for all categories.

## Install and run

The distribution is installed with `pip install ColorCorrectionPipeline`. The
APIs in this guide require a 2.x build; until publication, install the local
candidate wheel instead. An unqualified public install cannot retrieve an
unpublished candidate.

```shell
pip install ColorCorrectionPipeline
pip install "ColorCorrectionPipeline[performance]"
python -m ColorCorrectionPipeline.examples.quickstart --output results-rpcc2
```

The performance extra is optional. Its CPU kernels are qualified against the
reference arithmetic. Chart detection has a separate GPU provider policy and
CPU recovery. The sample resources are included with the wheel and their
hashes are checked before each tutorial. No UI checkout or UI executable is
needed.

To reproduce a released 1.x environment, pin the
original package and dependencies, for example `pip install ColorCorrectionPipeline==1.5.0`.

## Group existing controls by step

```python
from ColorCorrectionPipeline import Config, Steps, FFCOptions, WBOptions, CCOptions

config = Config(
    input={"frame_index": 0},
    chart={"device": "auto", "allow_cpu_fallback": True},
    numerics={"precision": "auto"},
    steps=Steps(
        ffc=FFCOptions(enabled=False),
        wb=WBOptions(fit={"headroom": 0.95}),
        cc=CCOptions(
            fit={"model": "nn", "basis": "root", "degree": 3,
                 "nn": {"hidden_layers": [100], "max_iterations": 10000}},
            validation={"enabled": True, "n_splits": 4, "random_state": 42},
        ),
    ),
)
```

`Steps`, `FFCOptions`, `GCOptions`, `WBOptions` and `CCOptions` are frozen
containers with owned immutable subgroup mappings. The same structure can be
provided as dictionaries. Omitted controls retain their existing defaults.

| Existing control | Grouped form |
| --- | --- |
| `do_ffc` / `FFC_kwargs` | `steps.ffc.enabled`, `.fit`, `.white_plane`, `.quality`, `.reporting` |
| `do_gc` / modern GC fields in `stage_policy` | `steps.gc.enabled`, `.quality`, `.reporting` |
| `do_wb` / modern WB fields in `stage_policy` | `steps.wb.enabled`, `.fit`, `.quality`, `.reporting` |
| `do_cc`, `model`, `basis`, `degree`, solver kwargs | `steps.cc.enabled`, `.fit`, including `.fit.nn` or `.fit.pls` |
| CC LUT kwargs | `steps.cc.apply` (`grid_size` / `min_pixels` replace `lut_grid_size` / `lut_min_pixels` there) |
| `validation_options` | `steps.cc.validation` |
| `image_options`, `chart_detection` | Shared `input`, `chart` mappings with existing field names |
| `reference_profile`, `color_semantics`, `precision` | Shared `reference`, `color`, `numerics.precision` |
| `save`, `save_path`, `model_format` | `persistence.enabled`, `.directory`, `.format` |

These are aliases into one authoritative configuration. Existing mutable flat
attributes continue to work. A grouped accessor returns a fresh read-only
snapshot, rather than another mutable copy to synchronize. Equal duplicates are
accepted and conflicting values fail, including explicitly supplied default
flags such as `do_ffc=True` versus `steps.ffc.enabled=False`. Numeric aliases
must have the same type as well as value (`True` versus `1`, or `2` versus
`2.0`, conflict); equivalent `Path` and string paths are accepted.

`to_dict()` retains its existing flat representation. `to_grouped_dict()` is the
explicit nested JSON form with `layout="grouped-v1"` and `api_version=2`;
`Config.from_dict` reads that marker. An unversioned flat dictionary still means
API 1. To deliberately interpret a new unversioned mapping as API 2, supply
`source_version=2`. Configuration interpretation, grouped layout version,
numerical runtime and bundle schema are separate version domains.

Grouped controls target modern API-2 UI-family configurations. Historical
training stays in its explicit flat form; its configuration cannot be silently
converted to a native residual NN by grouping a similarly named method. The
grouped API does not add export or prepared-execution wrappers to `Config`:
use `write_image_data` and the existing runtime-qualified `ExecutionOptions`
API for those operations.

## Model and basis selection

```python
from ColorCorrectionPipeline import Config

# These presets specify their fixed bases.
default = Config()
ccm = Config(model="3x3")
rpcc2 = Config(model="rpcc2")
rpcc3 = Config(model="rpcc3")

# Choose a solver and its feature basis independently.
ordinary = Config(model="linear", basis="polynomial", degree=4)
root_pls = Config(model="pls", basis="root", degree=3, CC_kwargs={"ncomp": 6})
hybrid_nn = Config(model="nn", basis="hybrid", degree=4,
                   CC_kwargs={"hidden_layers": [100], "max_iterations": 10000})
```

| Basis | Degree 1 | Degree 2 | Degree 3 | Degree 4 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary polynomial | 3 | 9 | 19 | 34 |
| Root polynomial | 3 | 6 | 13 | 22 |
| Hybrid | 3 | 12 | 29 | 53 |

Counts exclude constants and intercepts. Hybrid includes the ordinary features
and additional distinct root features. Fixed matrix presets use a through-origin
ridge fit. Root-linear has the same convention. Ordinary and hybrid linear
have a separate intercept. PLS centers/scales its inputs, and NN adds a learned
residual to the original linear RGB input. `NN100` means `nn` with one
hidden layer of width 100.

The degree-four limit applies to new generalized CC fitting. Existing saved
higher-degree calibrations retain their feature order and coefficients. FFC
uses its own degree and fitting settings.

New root/hybrid PLS fits require a positive `ncomp`. It is an explicit budget,
and the fit report records the number actually used. Pick it using
training-only validation or a separately measured capture. The Classic24 chart
provides 24 measured cells, while a degree-four hybrid model has 53 nonconstant
features. Regularization can provide a numerical fit, but a low training-chart
error does not establish performance on other colors or captures.

## Preserve a historical training configuration

Explicit historical policies retain their respective trainers, domains,
sampling and effective defaults.

```python
historical = Config(
    fitting_policy="legacy-v1",
    CC_kwargs={
        "mtd": "custom", "degree": 2, "hidden_layers": [64],
        "optim_type": "Adam", "learning_rate": 0.001,
        "max_iterations": 1000, "batch_size": 16,
        "patience": 10, "dropout_rate": 0.2,
    },
)

# Serialized v1 configurations are interpreted through their recorded policy.
restored_config = Config.from_dict(previous_config_dictionary)
```

| Previous usage | 2.0 use |
| --- | --- |
| A new `Config()` | UI-native RPCC2. Refitting intentionally uses the new workflow. |
| Explicit `legacy-v1` or `reproducible-v1` | Refit the original sklearn/Torch or conventional recipe with its working settings. |
| Explicit `ui-native-v1` with `CC_kwargs={"model": "rpcc3"}` | Keep that config for exact workflow reproduction, or select `model="rpcc3"` for a new v2 calibration with otherwise matched settings. |
| A safe `.ccpb` calibration | Call `ColorCorrection().load_bundle(path)` and apply it without retraining. |
| A legacy pickle | Use only the explicitly trusted-file loading API. |
| Old settings above degree four | Retain the historical policy or apply the stored calibration. Never clamp a saved degree. |

Historical sklearn MLP and Torch custom NN are distinct from the UI residual
network. Similar method names, hidden widths and seeds do not reconstruct their
equations. Preserve the original policy with the settings when reproducing an
old result. Historical higher-degree limits, aliases, stopping settings and
effective controls are handled by their versioned training path.

## Precision and input contracts

`precision="auto"` chooses a runtime once during calibration. Full-range
uint8 sRGB chart and enabled white-field images select the byte-native
workflow. Supported higher-precision inputs select the float64 runtime.
Use `precision="high"` explicitly when a byte chart will calibrate later
uint16 or floating scenes.

The saved runtime governs application to every scene and batch item. An
incompatible input raises an actionable error. Declare channel order,
transfer encoding and the physical sample range through `ImageData` when
the source is ambiguous. A float dtype alone does not imply linear light.

Detector previews locate geometry. Measurements and high-precision FFC
estimation use original-precision samples. The byte and high-precision
profiles record their arithmetic and fusion behavior separately. A strict
file readback checks stored samples. Qualification also checks the samples
before export so earlier precision loss cannot hide behind a correct codec.

```python
from ColorCorrectionPipeline import ImageData

# Declare a 12-bit sensor's stored range without stretching it from observed extrema.
chart = ImageData.from_array(raw_rgb_uint16, channel_order="RGB",
                             color_space="srgb", value_range=(0, 4095))
config = Config(model="rpcc2", precision="high", do_ffc=False)
```

Use an encoding that describes the capture. RAW demosaicing, tone mapping and
unsupported ICC conversions require a qualified preprocessing step. Preserve
the selected range in the calibration report. A uint16 container created by
expanding byte JPEG codes demonstrates the interface and adds no capture
information.

## Executable examples

All tutorials support `--help`. Each output directory must be new. The focused
calibration tutorials below accept
`--chart`, `--white`, `--scene` for your own captures. With a caller-supplied chart,
omitting `--white` disables FFC. `--no-white` disables it for bundled samples.

```shell
python -m ColorCorrectionPipeline.examples.quickstart --output run-default
python -m ColorCorrectionPipeline.examples.quickstart --no-white --output run-no-white
python -m ColorCorrectionPipeline.examples.compare_bases --output comparison --models linear,pls,nn --bases polynomial,root,hybrid --degrees 2,3,4 --pls-components 3
python -m ColorCorrectionPipeline.examples.legacy_training --method custom --iterations 100 --output old-torch
python -m ColorCorrectionPipeline.examples.high_precision --output precision-demo
python -m ColorCorrectionPipeline.examples.apply_bundle --bundle run-default/calibration.ccpb --image scene.png --output replay
python -m ColorCorrectionPipeline.examples.customizable --write-settings settings.json
python -m ColorCorrectionPipeline.examples.customizable --settings settings.json --output customized
```

`quickstart` saves `calibration.ccpb`, `corrected.png` and `results.json`.
It checks exact replay through a freshly loaded calibration and reports the
calibration, apply and verified-export times separately. `compare_bases` adds
effective settings and grouped CC validation. NN uses its full 10,000-epoch cap
unless `--nn-epochs` is supplied. A shortened cap is a smoke test configuration.

`high_precision` writes float64 and uint16 TIFF files plus an explicit 8-bit PNG
preview. With the built-in byte JPEGs it expands the original codes to uint16
for demonstration. Pass genuine uint16 or float TIFF captures to assess your
measurement workflow.

[`customizable.py`](../ColorCorrectionPipeline/examples/customizable.py) is a
complete editable workflow using `SETTINGS` or a JSON settings file. Its starter
uses an explicit `grouped-v1` calibration document. Change
`calibration.steps.cc.fit` to select the module's 3×3, polynomial linear, root
PLS or hybrid NN recipes. A supplied `calibration` dictionary replaces that
entire section, so retain any overrides you still need. Existing flat
calibration sections remain accepted. The example selects API version 2
deliberately for new flat mappings and restores array-valued options.

Set `use_sample_images=false` and fill in `chart`, `white` and a nonempty `scenes`
list for your captures. Set `calibration.steps.ffc.enabled=false` to omit the
white in the grouped starter. Flat settings can still use `do_ffc=False`. Capture,
output-directory and resolved calibration-save paths resolve relative to the JSON file;
nested detector/model paths retain their usual working-directory semantics.
`--output` resolves relative to the working directory. Optional physical
`reference` data uses the `ReferenceProfile.from_values` contract. Exports accept
`write_image_data` options and record readback verification, precision consent
and timing. Each scene receives its own directory. The example refuses existing
output directories and compares predictions from the in-memory calibration and
a freshly loaded safe bundle exactly.

The reported training-chart ΔE00 compares fixed patch means on the fitted
capture. Grouped validation excludes held-out CC cells while upstream FFC/GC/WB
remain fitted. Scene images have no independent measured truth. An independent
capture with a measured reference is required for that accuracy claim.

## Verification

The standalone `scripts/verify_v2.py` checks the 36 solver/basis/degree requests,
feature counts, independent Finlayson root values, exposure scaling of the
root basis, zero-residual NN identity, legacy trainer operation, invalid
configurations and serialization. Add `--photos` for real calibration,
safe-bundle replay, strict export and adjacent uint16/float64 checks.

```shell
python scripts/verify_v2.py --photos --output evidence/v2.json
```

For a release candidate, run it from a neutral directory against the built
wheel and source distribution with `--forbid-source /path/to/checkout`.
Use the existing UI comparator for the 15 shared methods. New root/hybrid
PLS and NN combinations require their own reference and held-out validation.
