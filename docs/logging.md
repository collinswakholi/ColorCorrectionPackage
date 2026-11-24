# Logging and warnings

ColorCorrectionPipeline does not write progress or diagnostic text to standard output.
Since 2.3.0, messages that earlier versions printed (for example `Info: Prediction done
in 0.12s`, `Detected white plane: ...`, `FFC fitting using method 'linear'...` or
`Info: Using custom CC method: nn`) go to the standard-library `logging` module.

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
| `DEBUG` | chart-detection retries with adjusted parameters |
| `INFO` | prediction and batch timings, selected CC method, detected white plane, FFC fit method, skipped stages, `read_batch(verbose=True)` progress, `write_image` destinations |
| `WARNING` | no chart found in an image, several white-plane candidates (largest used), `read_batch` failures, FFC metrics unavailable |
| `ERROR` | a fit stage failed (also recorded in the fit result), `write_image` failures |

Narrow the output with logger names, for example
`logging.getLogger("ColorCorrectionPipeline.flat_field").setLevel(logging.WARNING)`.

## Warnings that stay visible

Situations where the package silently substitutes a fallback that changes the result are
reported with `warnings.warn` (a `UserWarning`), which Python shows by default:

- the flat-field white plane is not detected, or no manual ROI is selected, so the full
  image is used;
- the flat-field detection model is missing or cannot be loaded;
- an unknown flat-field fit method falls back to `linear`;
- fewer charts are detected than requested;
- `write_image` receives an unknown `color_space` and assumes BGR.

Filter them with the `warnings` module as usual, e.g.
`warnings.filterwarnings("ignore", message="no white plane detected")`.

## Output that still prints

- `CustomNN(verbose=True)` (the legacy `custom` CC method) prints its device, architecture
  and training progress. `verbose` defaults to `False`; the output is an explicit request.
- Manual flat-field ROI selection prints its interactive instruction before opening the
  OpenCV window.
- The tutorial scripts under `ColorCorrectionPipeline.examples` print their results.
