"""
ColorCorrectionPipeline - Step-wise Color Correction Pipeline
==============================================================

A step-wise, end-to-end color correction pipeline for digital images combining
flat-field correction, gamma correction, white-balance, and color-correction into
a single, easy-to-use workflow.

Main Components:
    - ColorCorrection: Main pipeline class for color correction
    - Config: Configuration container for pipeline steps  
    - MyModels: Model storage and persistence
    - FlatFieldCorrection: Flat-field correction utilities

Example:
    >>> from ColorCorrectionPipeline import ColorCorrection, Config
    >>> from ColorCorrectionPipeline.core.utils import to_float64
    >>> import cv2
    >>> 
    >>> # Load image
    >>> img_bgr = cv2.imread("sample.jpg")
    >>> img_rgb = to_float64(img_bgr[:, :, ::-1])
    >>> 
    >>> # Configure and run
    >>> config = Config(do_ffc=True, do_gc=True, do_wb=True, do_cc=True)
    >>> cc = ColorCorrection()
    >>> metrics, images, errors = cc.run(Image=img_rgb, config=config)
"""

import sys
import warnings
from typing import List

# Suppress known warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Version information
from .__version__ import __version__, __version_info__

# Import constants first to avoid circular imports
from .constants import MODEL_PATH
from .contracts import ImageData, ReferenceProfile, ChartObservation, ObservationSet
from .colorimetry import ColorSemantics
from .stage_policies import StagePolicy, MonotonicToneModel
from .fitting import FittingPolicy
from .bundle import BundleLimits, BundleError
from .results import BatchItemResult, BatchResult
from .reporting import save_report
from .acceleration import ExecutionOptions, PreparedExecution, AcceleratedResult, LutCache, AccuracyBudgetError

# Import main classes
try:
    from .config import Config
    from .flat_field import FlatFieldCorrection
    from .models import MyModels
    from .pipeline import ColorCorrection
except ImportError as e:
    # Fallback for development/installation issues
    warnings.warn(
        f"Failed to import main components: {e}. "
        "Package may not be properly installed.",
        ImportWarning,
    )
    raise

# Import submodules for convenience
from . import core
from . import io

# MCC is now an explicitly selected legacy backend; model detection is independent.
if not core.HAS_MCC:
    warnings.warn(
        "cv2.mcc module not found \u2014 the legacy MCC backend is unavailable. "
        "Model-based detection remains available with the bundled ONNX model and onnxruntime.",
        ImportWarning,
    )

# Public API
__all__: List[str] = [
    "__version__",
    "__version_info__",
    "ColorCorrection",
    "Config",
    "MyModels",
    "FlatFieldCorrection",
    "MODEL_PATH",
    "ImageData", "ReferenceProfile", "ChartObservation", "ObservationSet", "ColorSemantics",
    "StagePolicy", "MonotonicToneModel", "FittingPolicy",
    "BundleLimits", "BundleError", "BatchItemResult", "BatchResult", "save_report",
    "ExecutionOptions", "PreparedExecution", "AcceleratedResult", "LutCache", "AccuracyBudgetError",
    "core",
    "io",
]

# Package metadata
__author__ = "Collins Wakholi"
__email__ = "collinswakholi@example.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Collins Wakholi"

# For pdoc documentation
if "pdoc" in sys.modules:
    try:
        with open("README.txt", "r", encoding="utf-8") as fh:
            __doc__ = fh.read()
    except FileNotFoundError:
        pass
