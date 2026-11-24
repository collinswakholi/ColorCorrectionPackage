"""
Core module for ColorCorrectionPipeline
=======================================

Low-level inference classes, transforms and utilities used by the pipeline.

Submodules:
    - correction: color-correction inference (Regressor_Model, predict_) and the
      load-only appliers for 2.x models (CustomNN, legacy polynomial predict_image)
    - transforms: polynomial features
    - utils: chart extraction, format conversions and small numeric helpers
    - accel: vectorised/Numba color conversion and LUT kernels
"""

from .correction import Regressor_Model, predict_
from .utils import (
    extract_color_chart,
    extract_color_charts,
    extract_neutral_patches,
    get_attr,
    poly_func,
    to_float64,
    to_uint8,
)


def __getattr__(name):
    if name == "get_poly_features":
        from .transforms import get_poly_features

        return get_poly_features
    if name in ("CustomNN", "predict_image"):
        # Load-only 2.x appliers, deferred so importing core stays cheap.
        from . import correction

        value = getattr(correction, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()).union(__all__))


__all__ = [
    # Correction (CustomNN and predict_image apply models saved by 2.x)
    "CustomNN",
    "Regressor_Model",
    "predict_",
    "predict_image",
    # Transforms
    "get_poly_features",
    # Utils
    "extract_color_chart",
    "extract_color_charts",
    "extract_neutral_patches",
    "get_attr",
    "poly_func",
    "to_float64",
    "to_uint8",
]
