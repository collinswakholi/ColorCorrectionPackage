"""Chart detection with the bundled ONNX model.

The supplied Opt2 ONNX contract uses 24 *anchor classes*, not 24 color cells.
Classic 24, SpyderCHECKR 24 and ColorChecker Digital SG charts are recognised; the
reference in use decides which one a calibration samples.
"""

from .charts import (
    DEFAULT_MODEL_PATH,
    ChartDetectionError,
    DetectionOptions,
    ModelUnavailableError,
    NoChartDetectedError,
    UnsupportedChartError,
    bind_detection_source,
    detection_context,
    get_detection_diagnostics,
    model_chart_samples,
    model_extract_color_charts,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "ChartDetectionError",
    "DetectionOptions",
    "ModelUnavailableError",
    "NoChartDetectedError",
    "UnsupportedChartError",
    "detection_context",
    "get_detection_diagnostics",
    "model_chart_samples",
    "model_extract_color_charts",
    "bind_detection_source",
]
