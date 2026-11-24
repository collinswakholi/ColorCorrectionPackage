# ColorCorrectionPipeline

Calibrate a camera setup from one photo of a **ColorChecker Classic 24** chart,
then apply that calibration to every image taken under the same conditions.

The pipeline corrects, in order: uneven lighting (**flat-field correction**,
from an optional photo of a white target), tone (**gamma**), color cast
(**white balance**), and remaining color error (**color correction**).

## Install

```shell
pip install ColorCorrectionPipeline
```

Python 3.8–3.14 on Windows, Linux and macOS. Optional extras:
`ColorCorrectionPipeline[performance]` adds faster CPU kernels (Numba), and
`ColorCorrectionPipeline[tiff]` reads compressed 16-bit TIFFs. Chart detection
uses a GPU when one is available and falls back to the CPU on its own.

## Quick start

You need a photo of the chart and, optionally, a photo of a plain white target
taken with the same lighting and camera position.

```python
from ColorCorrectionPipeline import calibrate, correct

cal = calibrate("chart.jpg", white="white.jpg", save_to="calibration.ccpb")
print("Chart error after correction (mean Delta E00):", cal.delta_e)

image = correct("scene.jpg", cal)                 # RGB float array, values 0–1
image = correct("scene.jpg", "calibration.ccpb")  # later, from the saved file
```

- Leave out `white=` if you have no white photo; flat-field correction is then skipped.
- `calibrate` raises `CalibrationError` if a stage fails, for example when no chart is found.
- The `.ccpb` file is a portable calibration you can share and reload safely.

To save the result as an image file:

```python
from ColorCorrectionPipeline.io import write_image_data

write_image_data("scene_corrected.png", image, dtype="uint8", allow_precision_loss=True)
```

## Correct many images

```python
from ColorCorrectionPipeline import ColorCorrection

pipeline = ColorCorrection().load_bundle("calibration.ccpb")
results = pipeline.predict_images(["a.jpg", "b.jpg", "c.jpg"], stages="final")
corrected = [r["CC"] for r in results]            # same order as the input
```

`stages="final"` returns only the finished image, which is faster than keeping
every intermediate stage. See [batch processing](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/batch-and-reporting.md)
for progress callbacks, worker counts and writing results to disk.

## Configure

The defaults suit most photographs. To change the color model, pass a preset name:

```python
cal = calibrate("chart.jpg", white="white.jpg", model="rpcc3")
```

| Preset | Use when |
| --- | --- |
| `rpcc2` (default) | General use. Robust to exposure changes. |
| `rpcc3` | A well-exposed chart and you want a closer fit. |
| `ccm` (or `3x3`) | You want a simple 3×3 matrix. |
| `pls` | Many correlated features. |
| `high-precision` | Your scenes are 16-bit or float. |

For anything else, such as turning stages off, flat-field options, neural-network
or PLS models, CPU-only detection, or your own chart reference values, build a
`Config`:

```python
from ColorCorrectionPipeline import Config, calibrate

config = Config.from_preset("rpcc2", steps={"ffc": {"apply": {"domain": "linear"}}})
cal = calibrate("chart.jpg", white="white.jpg", config=config)
```

The **[configuration guide](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/configuration.md)**
has a recipe for each common task and a table of every setting with its default.

## Try it on the sample photos

The package ships a sample chart, white target and scene:

```shell
python -m ColorCorrectionPipeline.examples.quickstart --output demo
python -m ColorCorrectionPipeline.examples.apply_bundle --bundle demo/calibration.ccpb --image my_photo.jpg --output replay
```

The first command calibrates and writes `demo/calibration.ccpb`, a corrected
image and the metrics. The second applies that calibration to your own photo.

## Upgrading from 2.x

3.0 removed the old fitting recipes (`legacy-v1`, `reproducible-v1`, API-1
configuration dictionaries, MCC detection and `key_functions`). Requesting one
raises an error that names its replacement. Calibrations saved by 2.x still load
and give the same output. See [migrating to 3.0](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/migration-3.md),
or pin `ColorCorrectionPipeline==2.3.0` to keep the old recipes.

## Documentation

- [Configuration guide](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/configuration.md): recipes and every setting.
- [Batch processing](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/batch-and-reporting.md): many images, progress, reports.
- [Calibration bundles](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/calibration-bundles.md): the `.ccpb` format.
- [Reading and saving images](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/image-io-policies.md): bit depth, color profiles, TIFF and PNG export.
- [Chart detection](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/chart-detection.md) and [GPU selection](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/gpu-detection.md).
- [Logging](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/logging.md): enable progress messages with `logging.basicConfig(level=logging.INFO)`.
- [Fully customizable example](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/ColorCorrectionPipeline/examples/customizable.py): settings in a JSON file, several scenes, TIFF export.

A desktop app with the same workflow is available as
[ColorCorrect Studio](https://github.com/collinswakholi/ColorCorrectionPackage_UI).

## Development

```shell
pip install -e ".[dev,performance]"
pytest tests
black --check ColorCorrectionPipeline tests scripts
ruff check ColorCorrectionPipeline tests scripts
mypy ColorCorrectionPipeline
```

See [architecture](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/docs/architecture.md)
for how the code is organized.

## License and citation

The Python code is [MIT licensed](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/LICENSE).
The bundled chart-detection models are AGPL-3.0; see their
[notices](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md).

If you use this package in research, please cite:

> Wakholi, C., Hardigan, M. A., Lee, J., Lukas, S. B., Feldman, M. J., Altendorf,
> K. R., Neyhart, J. L., & Rippner, D. A. (2026). A systematic color correction
> pipeline for controlled-environment imaging. *The Plant Phenome Journal*, 9(1),
> e70067. [doi:10.1002/ppj2.70067](https://doi.org/10.1002/ppj2.70067)

Thanks to Devin A. Rippner for technical guidance, ORISE for fellowship support,
and USDA-ARS for research support. [Report an issue](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
