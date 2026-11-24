# ColorCorrectionPipeline

Calibrate **flat-field correction (FFC) → gamma correction (GC) → white balance
(WB) → color correction (CC)**, then apply the fitted calibration to images
captured under the same conditions.

**Version 3.0 fits only the UI-native workflow (RPCC2 by default).** The 2.x
historical fitters (`legacy-v1`, `reproducible-v1`, `staged-v1`, API-1
configurations, the OpenCV MCC backend and the 1.x `key_functions` module) were
removed; requesting one raises an error that names the replacement. For a
linear-light 3×3 CCM, use `Config(model="3x3")`. Calibrations saved by 2.x and
earlier still load and keep their original predictions. See
[migrating to 3.0](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/migration-3.md).

## Install

```shell
pip install --upgrade ColorCorrectionPipeline
```

To install the version used in the example below, run `pip install ColorCorrectionPipeline==3.0.0`.
The distribution and Python import are both named `ColorCorrectionPipeline`;
the repository is named ColorCorrectionPackage.

Python **3.8–3.14** is qualified across the CI matrix. Prefer maintained CPython
3.13 or 3.14 for a new environment; see [Python compatibility](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/python-compatibility.md).
Optional CPU acceleration is available with
`pip install --upgrade "ColorCorrectionPipeline[performance]"`. Reading compressed
16-bit TIFFs (LZW, Deflate) needs the `imagecodecs` codecs:
`pip install --upgrade "ColorCorrectionPipeline[tiff]"`. Chart detection separately
automatically tries supported GPU providers and falls back to CPU if none completes a verified model forward. No device setting is needed. Color fitting and exact bundle replay use the qualified CPU kernels; model loading does not require the GPU used for detection. See [device selection and diagnostics](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/gpu-detection.md).

## Quick start

> Added in 2.3.0.

```python
from ColorCorrectionPipeline import calibrate, correct

cal = calibrate("chart.jpg", white="white.jpg", save_to="calibration.ccpb")
print("Chart Delta E00 after correction:", cal.delta_e)

corrected = correct("scene.jpg", cal)                  # float RGB in 0–1
corrected = correct("scene.jpg", "calibration.ccpb")   # later, from the saved bundle
```

`calibrate` fits FFC → GC → WB → CC on a ColorChecker Classic 24 photograph and raises
`CalibrationError` if a stage fails (for example, when no chart is found). Leave out
`white=` to skip flat-field correction. It reads frame 0 of multi-frame files, which
many camera JPEGs need because they embed a preview frame.

Choose a model with `calibrate(..., model=...)` or build the same configuration with
`Config.from_preset(...)` to adjust it further:

| Preset | Model | Use when |
| --- | --- | --- |
| `rpcc2` (default) | Root-polynomial, degree 2 | General use; robust to exposure changes. |
| `rpcc3` | Root-polynomial, degree 3 | More flexibility with a well-exposed chart. |
| `ccm` / `3x3` | Linear-light 3×3 matrix | Simplest, most predictable transform. |
| `pls` | Partial least squares | Many correlated features. |
| `high-precision` | RPCC2 in float64 | 16-bit or float scenes from a byte chart. |

```python
from ColorCorrectionPipeline import Config, ColorCorrection

config = Config.from_preset("rpcc2", white=False, chart={"samples_per_patch": 20, "sampling_seed": 42})
pipeline = ColorCorrection()
fit = pipeline.fit("chart.jpg", config=config)          # same as fit_or_raise
outputs = pipeline.predict("scene.jpg")                 # same as predict_image
```

The rest of this page uses the full API, which these shortcuts call.

## Run the sample photographs

```shell
python -m ColorCorrectionPipeline.examples.quickstart --output demo
```

The quickstart fits the included chart and white field, explicitly saves
`demo/calibration.ccpb`, reloads it and writes `demo/corrected.png` and
`demo/results.json`. To reuse that calibration on your own image in a fresh process:

```shell
python -m ColorCorrectionPipeline.examples.apply_bundle --bundle demo/calibration.ccpb --image scene.jpg --output replay
```

## One complete test example

Save this as `test_calibration.py` and run `python test_calibration.py`.
It uses the sample photographs installed with the wheel (or, in a source checkout,
the copies in `Images/`), fits all four steps, saves and reloads
a safe calibration bundle, checks exact replay, and writes a verified PNG.
Choose a new output directory for each run.

```python
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

import numpy as np

from ColorCorrectionPipeline import (
    ColorCorrection, Config, Steps, FFCOptions, GCOptions, WBOptions, CCOptions,
)
from ColorCorrectionPipeline.io import write_image_data

output = Path("sample-results")
config = Config(
    steps=Steps(
        ffc=FFCOptions(enabled=True, fit={"method": "linear", "degree": 1}),
        gc=GCOptions(enabled=True, quality={"min_neutrals": 3}),
        wb=WBOptions(enabled=True, fit={"headroom": 0.95}),
        cc=CCOptions(
            enabled=True,
            fit={"model": "rpcc2"},
            validation={"enabled": False},
        ),
    ),
    input={"frame_index": 0},
    chart={"device": "auto", "allow_cpu_fallback": True},
    color={"input_encoding": "srgb", "output_encoding": "srgb"},
    numerics={"precision": "auto"},
    observations={"saturation": "report"},
    persistence={"enabled": True, "directory": str(output), "format": "bundle"},
)

output.mkdir(exist_ok=False)

with ExitStack() as stack:
    def sample(name):
        # Keep packaged files alive until calibration and prediction finish.
        package = "ColorCorrectionPipeline.examples.data"
        try:
            if hasattr(resources, "files"):
                source = resources.as_file(resources.files(package).joinpath(name))
            else:  # Python 3.8
                source = resources.path(package, name)
            path = stack.enter_context(source)
            if Path(path).is_file():
                return path
        except (ModuleNotFoundError, FileNotFoundError):
            pass
        # Editable installs and source checkouts keep the samples in Images/.
        return Path("Images") / name

    chart, white, scene = map(sample, ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))
    pipeline = ColorCorrection()
    fit = pipeline.fit_or_raise(chart, white, name_="chart", config=config)

    bundle = output / "chart_models.ccpb"
    restored = ColorCorrection().load_bundle(bundle)
    corrected = restored.predict_image(scene, stages="final")["CC"]
    direct = pipeline.predict_image(scene, stages="final")["CC"]
    np.testing.assert_array_equal(corrected, direct)

    record = write_image_data(
        output / "corrected.png", corrected,
        dtype="uint8", input_encoding="srgb", allow_precision_loss=True,
        png_profile="fast-lossless",
    )
    assert record.maximum_codec_error == 0

    print("Training-chart mean Delta E00:", fit.metrics["chart_CC"]["delta_e00_after_mean"])
    print("Bundle replay: exact; PNG codec error:", record.maximum_codec_error)
    print("Outputs:", output.resolve())
```

With automatic persistence, set the format explicitly to `bundle` and
use the fit name to locate the file. Here it is `sample-results/chart_models.ccpb`.
The result's `artifacts` lists the paths written by that fit. With saving enabled,
omitting the format selects the trusted-file pickle format. With saving disabled,
call `pipeline.save_bundle(path)` after a successful fit to choose the filename,
as the quickstart does.

`fit_or_raise()` raises `CalibrationError` for a failed stage and propagates
configuration and filesystem exceptions. After a failed refit, the pipeline still
holds the last successful calibration. Existing callers can continue using
`metrics, images, failed = pipeline.run(...)`.

The PNG is an explicitly quantized preview; zero codec error means its stored
samples survived readback unchanged. Use a suitable TIFF export when retaining
floating-point precision. Chart ΔE00 measures the calibration chart, not
independent scene accuracy.

For your own captures, replace the three sample paths with your chart, matching
white-field image and scene. Enabled FFC requires compatible image dimensions.
Without a matching white image, disable `steps.ffc.enabled` and pass `None`
as the white-image argument.

## Configuration breakdown

| Category | Purpose in the example |
| --- | --- |
| `steps.ffc` | Enables white-field correction and selects a linear illumination surface. Optional `apply={"domain": "linear"}` applies the gain to linear-light RGB instead of Lab L* (default `"lightness"`). |
| `steps.gc` | Fits the neutral-lightness curve; requires at least three usable neutral patches. |
| `steps.wb` | Fits neutral white balance with 0.95 headroom. |
| `steps.cc` | Fits RPCC2. `fit`, `apply`/LUT options and CC-only `validation` have separate subgroups; validation is disabled for this smoke test. |
| `input` | Selects frame zero explicitly for the bundled JPEGs. |
| `chart` | Selects chart detection, independently of CPU color fitting. `auto` allows GPU detection with CPU fallback. |
| `reference` | Omitted here to use the built-in UI-native Classic24 reference. Supply a `ReferenceProfile` for measured targets. |
| `color` | Declares input and output encoding; the example uses sRGB. |
| `numerics` | Resolves byte or high-precision execution from calibration inputs. Use `high` if a byte chart will later serve uint16/float scenes. |
| `observations` | Reports saturation evidence without synthesizing replacement measurements. |
| `persistence` | Saves models and metrics during calibration, using a safe bundle in the output directory. Image export is a separate writer call. |

CC supports `3x3`/`ccm`, `rpcc2`, `rpcc3`, `linear`, `pls` and `nn`.
Linear, PLS and NN fits support polynomial, root or hybrid features through
degree four. See the [grouped configuration reference](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/grouped-configuration.md)
for solver-specific controls, defaults, constraints and JSON.

Grouped options are immutable snapshots over the existing flat configuration.
Old constructor arguments and kwargs remain supported; contradictory aliases
fail explicitly. `Config.from_dict` needs a recorded `api_version` (or
`source_version=2`): unversioned dictionaries were API-1 requests, which 3.0
removed. A loaded calibration owns its recorded numerical recipe.

### Reproducible random chart sampling

```python
Config(model="3x3", chart={"samples_per_patch": 20, "sampling_seed": 42})
```

For a UI-native fit, omit both settings for the default 9 × 9 grid (81 points per patch). Set `samples_per_patch` (1–4096) for an exact count inside the inner 70% of each detected patch. The unsigned 32-bit seed defaults to zero. The same positions are reused across stages; the fitter still receives one mean RGB per patch. Do not combine a grid size with a random count. These settings affect calibration, not application of a saved model.

## Further usage

- [Fully customizable example](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/ColorCorrectionPipeline/examples/customizable.py):
  editable grouped JSON, custom references, multiple scenes, TIFF/PNG exports and replay checks.
- [Migrating to 3.0](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/migration-3.md): removed fitters and APIs, their replacements, and loading saved models.
- [Migration and model selection](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/migration-2.md): 2.x defaults and feature bases.
- [Images and exports](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/image-io-policies.md): encoding, range, precision consent and strict readback.
- [Calibration bundles](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/calibration-bundles.md): portable validated inference parameters.
- [Numerical policies](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/numerical-policies-and-validation.md): fit objectives, quality guards and validation scope.
- [Batch processing](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/batch-and-reporting.md) and [prepared execution](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/acceleration.md).
- [Logging and warnings](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/logging.md): progress and diagnostic messages use `logging`; enable them with `logging.basicConfig(level=logging.INFO)`.
- [Architecture](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/architecture.md) and [distribution qualification](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/distribution-qualification.md): source ownership and installed-package checks.

A desktop interface is available in
[ColorCorrectionPackage_UI](https://github.com/collinswakholi/ColorCorrectionPackage_UI).
This package builds on [ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool).

## Development

```shell
pip install -e ".[dev,performance]"
pytest tests                       # unit, regression and golden-output tests
python tests/golden/capture.py     # compare against the stored 2.2.0 baseline only
black --check ColorCorrectionPipeline tests scripts
ruff check ColorCorrectionPipeline tests scripts
mypy ColorCorrectionPipeline       # untyped modules excluded in pyproject.toml [tool.mypy]
```

CI (`.github/workflows/lint-and-test.yml`) runs the same formatting, lint and type-check
gates, the fast tests on Ubuntu and Windows, and the golden tests on Windows, where the
baseline was recorded.

To see the package's progress and diagnostic messages while developing, enable logging
(`logging.basicConfig(level=logging.INFO)`, or `DEBUG` for chart-detection retries); see
[Logging and warnings](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/docs/logging.md).

`tests/golden` stores metrics and thumbnails for eleven configurations captured from
2.2.0. Regenerate them with `--write` only for an intentional, documented change. The
`tests/` folder is not shipped in the sdist or wheel.

## License and citation

Python source is [MIT licensed](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/LICENSE). Bundled detector models declare
AGPL-3.0; their [notices and provenance](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v3.0.0/ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md)
are retained, including the recorded model-ancestry and training-data rights limitations.

Wakholi, C., Hardigan, M. A., Lee, J., Lukas, S. B., Feldman, M. J., Altendorf,
K. R., Neyhart, J. L., & Rippner, D. A. (2026). A systematic color correction
pipeline for controlled-environment imaging. *The Plant Phenome Journal*, 9(1),
e70067. [doi:10.1002/ppj2.70067](https://doi.org/10.1002/ppj2.70067).

Thanks to Devin A. Rippner for technical guidance, ORISE for fellowship support,
and USDA-ARS for research support.
[Report an issue](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
