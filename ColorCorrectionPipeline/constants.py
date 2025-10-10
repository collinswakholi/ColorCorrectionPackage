"""
Constants module for color_correc_optim
========================================

This module contains package-wide constants to avoid circular imports.
All paths are computed relative to this file for portability.
"""

import os
from pathlib import Path

import numpy as np

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Define the absolute path to the YOLO model for flat-field correction
MODEL_PATH = str(PACKAGE_ROOT / "flat_field" / "models" / "plane_det_model_YOLO_512_n.pt")

# Ensure the model path exists (for development)
if not os.path.exists(MODEL_PATH):
    import warnings
    
    warnings.warn(
        f"YOLO model not found at {MODEL_PATH}. "
        "Flat-field correction will require manual cropping.",
        RuntimeWarning,
    )

# Color science constants
WP_DEFAULT = np.array([0.31271, 0.32902])  # D65 white point (xy coordinates)
CMFS_DEFAULT = "CIE 1931 2 Degree Standard Observer"  # Default color matching functions
EPSILON = 1e-15  # Small value to avoid division by zero

__all__ = [
    "MODEL_PATH",
    "PACKAGE_ROOT",
    "WP_DEFAULT",
    "CMFS_DEFAULT",
    "EPSILON",
]
