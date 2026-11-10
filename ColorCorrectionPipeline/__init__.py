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
    >>> config = Config(model="rpcc2", image_options={"frame_index": 0})
    >>> cc = ColorCorrection()
    >>> fit = cc.fit_or_raise("chart.jpg", "white.jpg", config=config)

The example expects a chart capture and its matching white field. Run
``python -m ColorCorrectionPipeline.examples.quickstart --output demo``
to use the included sample images after installing the package.
"""

from importlib import import_module
import warnings

from .__version__ import __version__, __version_info__

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Resolve public names to their original objects and modules on first access.
# Asking for the package version should not initialize training or image backends.
_EXPORTS = {
    "FitResult": "pipeline", "CalibrationError": "pipeline",
    "ColorCorrection": "pipeline", "Config": "config", "MyModels": "models",
    "FlatFieldCorrection": "flat_field", "MODEL_PATH": "constants",
    "ImageData": "contracts", "ReferenceProfile": "contracts",
    "ChartObservation": "contracts", "ObservationSet": "contracts",
    "ColorSemantics": "colorimetry", "StagePolicy": "stage_policies",
    "MonotonicToneModel": "stage_policies", "FittingPolicy": "fitting",
    "Steps": "config_groups", "FFCOptions": "config_groups", "GCOptions": "config_groups",
    "WBOptions": "config_groups", "CCOptions": "config_groups",
    "BundleLimits": "bundle", "BundleError": "bundle", "BatchItemResult": "results",
    "BatchResult": "results", "save_report": "reporting", "ExecutionOptions": "acceleration",
    "PreparedExecution": "acceleration", "AcceleratedResult": "acceleration",
    "LutCache": "acceleration", "AccuracyBudgetError": "acceleration",
    "core": "core", "io": "io",
}
__all__ = ["__version__", "__version_info__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module = import_module("." + _EXPORTS[name], __name__)
    value = module if name == _EXPORTS[name] else getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__))


# Package metadata
__author__ = "Collins Wakholi"
__email__ = "49397327+collinswakholi@users.noreply.github.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Collins Wakholi"
