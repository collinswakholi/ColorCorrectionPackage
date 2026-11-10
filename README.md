# ColorCorrectionPipeline

Calibrate **flat-field correction (FFC) → gamma correction (GC) → white balance
(WB) → color correction (CC)**, then apply the fitted calibration to images
captured under the same conditions.

**Version 2.0 defaults to the UI-native RPCC2 workflow.** Legacy methodology is
retired as the default, along with `ours`/`conv` naming in the primary API.
Historical training configurations and saved calibrations retain their original
numerical behavior. See the [migration guide](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/migration-2.md).

## Install

```shell
pip install --upgrade ColorCorrectionPipeline
```

For the published 2.0.0 release, use `pip install ColorCorrectionPipeline==2.0.0`.
The checkout adds `fit_or_raise()`. To run the Python example below, install this
revision with `python -m pip install .` from the repository root.
The distribution and Python import are both named `ColorCorrectionPipeline`;
the repository is named ColorCorrectionPackage.

Python **3.8–3.14** is qualified across the CI matrix. Prefer maintained CPython
3.13 or 3.14 for a new environment; see [Python compatibility](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/python-compatibility.md).
Optional CPU acceleration is available with
`pip install --upgrade "ColorCorrectionPipeline[performance]"`. Chart detection separately
prefers a supported GPU provider and falls back to CPU; see [GPU setup](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/gpu-detection.md).

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

Save this as `test_calibration.py` outside the source checkout and run
`python test_calibration.py` so it imports the installed package.
It uses the installed sample photographs, fits all four steps, saves and reloads
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
        if hasattr(resources, "files"):
            source = resources.as_file(resources.files(package).joinpath(name))
        else:  # Python 3.8
            source = resources.path(package, name)
        return stack.enter_context(source)

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
omitting the format selects the historical pickle format. With saving disabled,
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
| `steps.ffc` | Enables white-field correction and selects a linear illumination surface. |
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
degree four. See the [grouped configuration reference](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/grouped-configuration.md)
for solver-specific controls, defaults, constraints and JSON.

Grouped options are immutable snapshots over the existing flat configuration.
Old constructor arguments and kwargs remain supported; contradictory aliases
fail explicitly. Unversioned flat dictionaries retain their historical API-1
interpretation. A loaded calibration owns its recorded numerical recipe.

## Further usage

- [Fully customizable example](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/ColorCorrectionPipeline/examples/customizable.py):
  editable grouped JSON, custom references, multiple scenes, TIFF/PNG exports and replay checks.
- [Migration and model selection](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/migration-2.md): retired defaults, historical training and feature bases.
- [Images and exports](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/image-io-policies.md): encoding, range, precision consent and strict readback.
- [Calibration bundles](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/calibration-bundles.md): portable validated inference parameters.
- [Numerical policies](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/numerical-policies-and-validation.md): fit objectives, quality guards and validation scope.
- [Batch processing](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/batch-and-reporting.md) and [prepared execution](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/acceleration.md).
- [Architecture](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/architecture.md) and [distribution qualification](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/docs/distribution-qualification.md): source ownership and installed-package checks.

A desktop interface is available in
[ColorCorrectionPackage_UI](https://github.com/collinswakholi/ColorCorrectionPackage_UI).
This package builds on [ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool).

## License and citation

Python source is [MIT licensed](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/LICENSE). Bundled detector models declare
AGPL-3.0; their [notices and provenance](https://github.com/collinswakholi/ColorCorrectionPackage/blob/v2.0.0/ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md)
are retained, including the recorded model-ancestry and training-data rights limitations.

Wakholi, C., Hardigan, M. A., Lee, J., Lukas, S. B., Feldman, M. J., Altendorf,
K. R., Neyhart, J. L., & Rippner, D. A. (2026). A systematic color correction
pipeline for controlled-environment imaging. *The Plant Phenome Journal*, 9(1),
e70067. [doi:10.1002/ppj2.70067](https://doi.org/10.1002/ppj2.70067).

Thanks to Devin A. Rippner for technical guidance, ORISE for fellowship support,
and USDA-ARS for research support.
[Report an issue](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
