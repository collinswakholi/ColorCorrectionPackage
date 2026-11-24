# Logging and warnings

ColorCorrectionPipeline does not write progress or diagnostic text to standard output.
Since 2.3.0, messages that earlier versions printed (for example `Info: Prediction done
in 0.12s` or `Detected white plane: ...`) go to the standard-library `logging` module.

Every module logs to a logger named after itself, under the package root logger
`ColorCorrectionPipeline`. The root logger carries a `NullHandler`, so nothing is shown
until the application configures logging.

## Show the messages

```python
import logging

logging.basicConfig(level=logging.INFO)          # everything, including other libraries

# or only this package, at debug detail:
log = logging.getLogger("ColorCorrectionPipeline")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
```

| Level | Examples |
| --- | --- |
| `DEBUG` | optional image normalization/quantization kernels falling back to NumPy |
| `INFO` | prediction and batch timings, detected white plane, skipped stages |
| `WARNING` | no chart found in an image, several white-plane candidates (largest used), optional kernel failures that fall back to the reference path, LUT build failures |
| `ERROR` | a fit stage failed (also recorded in the fit result) |

Narrow the output with logger names, for example
`logging.getLogger("ColorCorrectionPipeline.flat_field").setLevel(logging.WARNING)`.

## Warnings that stay visible

Situations where the package silently substitutes a fallback that changes the result are
reported with `warnings.warn` (a `UserWarning`), which Python shows by default:

- the flat-field white plane is not detected, or no manual ROI is selected, so the full
  image is used;
- the flat-field detection model is missing or cannot be loaded.

Filter them with the `warnings` module as usual, e.g.
`warnings.filterwarnings("ignore", message="no white plane detected")`.

## Output that still prints

- Manual flat-field ROI selection prints its interactive instruction before opening the
  OpenCV window.
- The tutorial scripts under `ColorCorrectionPipeline.examples` print their results.
