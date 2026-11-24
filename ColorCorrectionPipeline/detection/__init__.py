"""Classic24 chart detection with the bundled ONNX model.

The supplied Opt2 ONNX contract uses 24 *anchor classes*, not 24 color cells.
Only Classic 4x6 reference charts are supported by the Python pipeline today.
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
