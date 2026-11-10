"""
Core module for color_correc_optim
===================================

This module contains core color correction algorithms, utilities, and
transformations used throughout the pipeline.

Submodules:
    - color_spaces: Color space conversions and utilities
    - correction: Color correction algorithms (conventional and custom)
    - transforms: Polynomial features and transformations
    - metrics: Color difference metrics (Delta-E, MAE, etc.)
    - utils: General utility functions
"""

import warnings

from .color_spaces import (
    adapt_chart,
    convert_to_lab,
    convert_to_LCHab,
    do_color_adaptation,
    extrapolate_if_sat_image,
    srgb_to_cielab_D50,
)
from .correction import (
    CustomNN,
    Regressor_Model,
    color_correction,
    color_correction_1,
    estimate_gamma_profile,
    extract_color_chart_ex,
    fit_model,
    predict_,
    predict_image,
    wb_correction,
)
from .metrics import (
    Metrics,
    arrange_metrics,
    compute_mae,
    desc_stats,
    get_metrics,
)
from .utils import (
    HAS_MCC,
    compute_diag,
    compute_temperature,
    estimate_fit,
    extract_color_chart,
    extract_color_charts,
    extract_neutral_patches,
    free_memory,
    get_attr,
    poly_func,
    poly_func_torch,
    to_float64,
    to_uint8,
)

# MCC is now an explicitly selected legacy backend; model detection is independent.
if not HAS_MCC:
    warnings.warn(
        "cv2.mcc module not found \u2014 the legacy MCC backend is unavailable. "
        "Model-based detection remains available with the bundled ONNX model and onnxruntime.",
        ImportWarning,
    )


def __getattr__(name):
    if name == "get_poly_features":
        from .transforms import get_poly_features
        return get_poly_features
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()).union(__all__))


__all__ = [
    # Color spaces
    "adapt_chart",
    "convert_to_lab",
    "convert_to_LCHab",
    "do_color_adaptation",
    "extrapolate_if_sat_image",
    "srgb_to_cielab_D50",
    # Correction
    "CustomNN",
    "Regressor_Model",
    "color_correction",
    "color_correction_1",
    "estimate_gamma_profile",
    "extract_color_chart_ex",
    "fit_model",
    "predict_",
    "predict_image",
    "wb_correction",
    # Metrics
    "Metrics",
    "arrange_metrics",
    "compute_mae",
    "desc_stats",
    "get_metrics",
    # Transforms
    "get_poly_features",
    # Utils
    "HAS_MCC",
    "compute_diag",
    "compute_temperature",
    "estimate_fit",
    "extract_color_chart",
    "extract_color_charts",
    "extract_neutral_patches",
    "free_memory",
    "get_attr",
    "poly_func",
    "poly_func_torch",
    "to_float64",
    "to_uint8",
]
