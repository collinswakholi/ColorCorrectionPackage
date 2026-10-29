# ColorCorrectionPipeline

A stepwise color correction package for scientific images. Fit flat-field
correction (FFC), gamma correction (GC), white balance (WB) and color correction
(CC) from a chart photograph, then apply the calibration to images captured
under the same conditions.

The package builds on [ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool).
A desktop interface is available in [ColorCorrectionPackage_UI](https://github.com/collinswakholi/ColorCorrectionPackage_UI).

## Installation

Python 3.8 or later is required. The qualification matrix covers Python
3.8–3.12 on Windows, Linux and macOS; newer interpreters are not in that matrix.

```shell
git clone https://github.com/collinswakholi/ColorCorrectionPackage.git
cd ColorCorrectionPackage
python -m pip install -e ".[performance]"
```

The `performance` extra enables optional Numba CPU kernels. Install with
`python -m pip install -e .` for the NumPy fallback. Scientific, Torch, plotting
and ONNX Runtime dependencies are included. The separately published PyPI
package can be installed with `python -m pip install ColorCorrectionPipeline`;
its features depend on the published version.

Chart detection defaults to the bundled Classic24 ONNX model. It prefers a
supported GPU provider and falls back to CPU. See [GPU configuration](docs/gpu-detection.md)
for CUDA, DirectML, WebGPU and CoreML setup and execution diagnostics.
Chart detection and color-transform acceleration have separate backend policies.

## Calibrate and apply

This example uses the bundled sample photographs and explicitly selects their
first frame. Modern numerical policies are opt-in; existing callers retain the
legacy encoded-sRGB stage mathematics and return conventions.

```python
from pathlib import Path
from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import write_image_data

output = Path("results")
output.mkdir(exist_ok=True)
pipeline = ColorCorrection()
config = Config(
    do_ffc=True, do_gc=True, do_wb=True, do_cc=True,
    check_saturation=False,
    color_semantics={"version": "explicit-v1"},
    stage_policy="modern-v1",
    fitting_policy="reproducible-v1",
    image_options={"frame_index": 0},
    FFC_kwargs={"fit_method": "linear", "degree": 2, "bins": 30},
    CC_kwargs={"degree": 1, "use_lut": False},
    save=True, save_path=str(output), model_format="bundle",
)
metrics, images, failed = pipeline.run(
    "Images/Image_1.JPG", "Images/white.JPG", name_="sample", config=config
)
if failed:
    raise RuntimeError(pipeline.last_run)

# Loading a safe bundle restores the numerical policies and reference context.
restored = ColorCorrection().load_bundle(output / "sample_models.ccpb")
with restored.prepare_execution({"mode": "direct", "backend": "auto"}) as execution:
    result = execution.predict_image("Images/Sample_1.JPG")
    record = write_image_data(
        output / "corrected.png", result.images["CC"],
        dtype="uint8", allow_precision_loss=True,
    )
```

Set `do_ffc=False` and omit the white image when no matching white-field
photograph is available. Enabled stages are required: `last_run` records stage
outcomes, errors, detector execution and persistence diagnostics. A failed fit
or save preserves the previous successful calibration.

## Working with the package

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
  version = {1.4.6},
  year = {2026}
}
```

Thanks to Devin A. Rippner for technical guidance, ORISE for fellowship support,
and USDA-ARS for research support. Report issues and feature requests on
[GitHub](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
