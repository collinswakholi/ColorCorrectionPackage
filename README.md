# ColorCorrectionPipeline

A stepwise color correction package for scientific images. Fit flat-field
correction (FFC), gamma correction (GC), white balance (WB) and color correction
(CC) from a chart photograph, then apply the calibration to images captured
under the same conditions.

The package builds on [ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool).
A desktop interface is available in [ColorCorrectionPackage_UI](https://github.com/collinswakholi/ColorCorrectionPackage_UI).

Starting in 2.0, new calibrations use the UI-native workflow and default to
RPCC2. Select a color model, feature basis and degree directly. The old
`ours`/`conv` terminology is retired from the primary training API. Historical
ML and conventional training configurations remain available through their
original numerical policy, and saved calibrations retain their recorded rules.
See the [2.0 migration guide](docs/migration-2.md) for compatibility, polynomial
and root-polynomial fitting, and high-precision image handling.

## Installation

Python 3.8 or later is required. The qualification matrix covers Python
3.8–3.12 on Windows, Linux and macOS; newer interpreters are not in that matrix.

```shell
pip install ColorCorrectionPipeline
```

The `performance` extra enables optional Numba CPU kernels. Install it with
`pip install "ColorCorrectionPipeline[performance]"`. Scientific, Torch,
plotting and ONNX Runtime dependencies are included. The repository is named
ColorCorrectionPackage. Both the installed distribution and Python import are
ColorCorrectionPipeline. Features described for 2.0 require that version.

Chart detection defaults to the bundled Classic24 ONNX model. It prefers a
supported GPU provider and falls back to CPU. See [GPU configuration](docs/gpu-detection.md)
for CUDA, DirectML, WebGPU and CoreML setup and execution diagnostics.
Chart detection and color-transform acceleration have separate backend policies.

## Calibrate and apply

Run the installed tutorial with the bundled sample photographs. It calibrates
RPCC2, applies the result, saves a portable bundle, checks exact replay and
writes a verified PNG. Choose a new output directory for each run.

```shell
python -m ColorCorrectionPipeline.examples.quickstart --output results-rpcc2
```

With your own chart and scene, omit `--white` to disable flat-field correction.
Each source must declare a supported encoding and range. The sample JPEGs expose
multiple frames, so the tutorials explicitly select frame zero.

To use the Python API, supply paths to your captures:

```python
from pathlib import Path
from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import write_image_data

output = Path("results")
output.mkdir(exist_ok=False)
pipeline = ColorCorrection()
config = Config(
    model="rpcc2",
    image_options={"frame_index": 0},
    save=True, save_path=str(output), model_format="bundle",
)
metrics, images, failed = pipeline.run(
    "chart.png", "white.png", name_="sample", config=config
)
if failed:
    raise RuntimeError(pipeline.last_run["errors"])

# Loading a safe bundle restores the numerical policies and reference context.
restored = ColorCorrection().load_bundle(output / "sample_models.ccpb")
result = restored.predict_image("scene.png", stages="final")
record = write_image_data(
    output / "corrected.png", result["CC"],
    dtype="uint8", allow_precision_loss=True,
)
```

Set `do_ffc=False` and omit the white image when no matching white-field
photograph is available. Enabled stages are required: `last_run` records stage
outcomes, errors, detector execution and persistence diagnostics. A failed fit
or save preserves the previous successful calibration.

The existing `legacy-v1` and `reproducible-v1` encoded-sRGB fits remain available
for training with their respective configurations. Their linear
solver minimizes RGB squared error against unclipped reference targets; lower
RGB error can increase CIEDE2000, particularly for out-of-gamut references.
The fit report records this objective and affected target cells. See
[fit objectives and perceptual evaluation](docs/numerical-policies-and-validation.md#fit-objective-and-perceptual-evaluation).
Linear-light CCM also optimizes RGB error; evaluate its measured results rather
than assuming every correction reduces perceptual error.

## Working with the package

| New fit | Configuration |
| --- | --- |
| Default root-polynomial matrix | `Config(model="rpcc2")` |
| 3×3 matrix | `Config(model="3x3")` |
| Degree-four polynomial linear fit | `Config(model="linear", basis="polynomial", degree=4)` |
| PLS on degree-three roots | `Config(model="pls", basis="root", degree=3, CC_kwargs={"ncomp": 6})` |
| Width-100 NN on hybrid features | `Config(model="nn", basis="hybrid", degree=4, CC_kwargs={"hidden_layers": [100]})` |

`precision="auto"` selects the recorded byte or high-precision runtime once at
calibration. Choose `precision="high"` when a byte chart will calibrate future
uint16/float scenes. New root/hybrid PLS fits require an explicit component
budget. Training-chart ΔE00 and held-out color error answer different questions;
the [comparison example](docs/migration-2.md#executable-examples) reports both.

- [Contracts](docs/contracts.md): immutable images, canonical references and chart observations.
- [Color semantics](docs/color-semantics.md) and [pipeline integration](docs/contracts-and-color-integration.md): declared encoding, illuminants and orientation.
- [Numerical policies](docs/numerical-policies-and-validation.md): clipping evidence, monotonic neutral gamma, guarded FFC, reproducible fitting and grouped evaluation.
- [Chart detection](docs/chart-detection.md): model settings, supported layouts and explicit OpenCV MCC compatibility.
- [Persistence and compatibility](docs/local-reliability.md): transactional state and honest failure behavior.
- [Calibration bundles](docs/calibration-bundles.md): portable validated parameters; legacy pickle loading remains trusted-file-only.
- [Image I/O](docs/image-io-policies.md): precision consent, strict readback verification and atomic publication. Optional `png_profile="fast-lossless"` preserves PNG samples with larger files.
- [Batch processing and reporting](docs/batch-and-reporting.md): bounded work, cancellation, per-file outcomes and stage retention.
- [Prepared execution](docs/acceleration.md): direct CPU execution, explicitly budgeted combined LUTs, cache ownership and cube export.
- [Performance measurements](docs/performance.md) and [distribution qualification](docs/distribution-qualification.md).

Legacy `run`, `predict_image`, `predict_images`, array/image helpers, saved-model
formats and import entry points remain available. New contracts and policies
do not reinterpret older calibrations. Chart model geometry differs from MCC,
so fitting a new calibration can produce different measurements.

The runtime is organized around `pipeline.py` and `config.py`; `contracts.py`,
`colorimetry.py` and `observations.py` define the interpretation boundaries.
`detection/` supplies model geometry and provider policy. `core/`, `modern.py`,
`stage_policies.py` and `fitting.py` implement stage execution and fitting.
`ui_fitting.py` and `ui_nn.py` contain UI numerical fits; `ui_native.py` records
native image semantics and `ui_execution.py` supplies optional CPU execution.
`bundle.py` handles safe inference artifacts; `acceleration.py` prepares execution;
`batch.py`, `prediction.py`, `reporting.py` and `io/` handle application and output.
`key_functions.py` retains historical utility imports for compatibility.

## Example output

The illustrations show an example workflow. Chart residuals and appearance do
not establish physical color accuracy on independent photographs; results
depend on the reference, capture conditions, fit and validation protocol.

| Before | After |
| --- | --- |
| ![Before correction](ReadMe_Images/before.svg) | ![After correction](ReadMe_Images/After.svg) |

## Development and qualification

Install developer tools with `python -m pip install -e ".[dev,performance]"`.
Build and inspect distributions with:

```shell
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
python scripts/verify_distribution.py --output evidence/artifacts.json
```

The release workflow installs the built wheel into isolated base and performance
environments and runs `scripts/verify_installed.py`, including a real-photo fit,
safe-bundle reload, precision I/O and bounded batch processing. See the
[qualification guide](docs/distribution-qualification.md) for neutral-directory
commands. Generated evidence belongs outside the source tree. Submit changes
through a pull request with the relevant numerical and compatibility evidence.
Run `python scripts/verify_v2.py --photos --output evidence/v2.json` from a
neutral directory to qualify the new selectors, bases, historical trainers,
real-photo bundle replay and high-precision scene handling. The installed
distribution supplies the example samples.

## License

Python source is MIT licensed; see [LICENSE](LICENSE). Bundled detector models
declare AGPL-3.0. Their original license, notices and provenance are retained
[beside the chart model](ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md).
Source licensing does not replace the model terms; the provenance record also
states the unresolved model ancestry and training-data rights information.

## Citation

Wakholi, C., Hardigan, M. A., Lee, J., Lukas, S. B., Feldman, M. J., Altendorf,
K. R., Neyhart, J. L., & Rippner, D. A. (2026). A systematic color correction
pipeline for controlled-environment imaging. *The Plant Phenome Journal*, 9(1),
e70067. [doi:10.1002/ppj2.70067](https://doi.org/10.1002/ppj2.70067).

```bibtex
@software{colorcorrectionpipeline,
  author = {Wakholi, Collins and Rippner, Devin A.},
  title = {ColorCorrectionPipeline: A stepwise color correction pipeline},
  url = {https://github.com/collinswakholi/ColorCorrectionPackage},
  version = {1.5.0},
  year = {2026}
}
```

Thanks to Devin A. Rippner for technical guidance, ORISE for fellowship support,
and USDA-ARS for research support. Report issues and feature requests on
[GitHub](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
