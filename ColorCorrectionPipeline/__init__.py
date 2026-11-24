"""
ColorCorrectionPipeline - Step-wise Color Correction Pipeline
==============================================================

A step-wise, end-to-end color correction pipeline for digital images combining
flat-field correction, gamma correction, white-balance, and color-correction into
a single, easy-to-use workflow.

Quick start (two calls):
    >>> from ColorCorrectionPipeline import calibrate, correct
    >>> cal = calibrate("chart.jpg", white="white.jpg", save_to="calibration.ccpb")
    >>> corrected = correct("scene.jpg", cal)        # float RGB, 0-1

Main components:
    - calibrate / correct / Calibration: two-call API for the common case
    - ColorCorrection: the pipeline (fit, predict, bundles, batch prediction)
    - Config (+ Steps, FFCOptions, GCOptions, WBOptions, CCOptions): full configuration
    - Config.from_preset: named presets (rpcc2, rpcc3, ccm, pls, high-precision)
    - ImageData, ReferenceProfile, ColorSemantics, FittingPolicy, StagePolicy: contracts
    - io / core: image readers and writers, low-level color and fitting functions

Full control:
    >>> from ColorCorrectionPipeline import ColorCorrection, Config
    >>> config = Config(model="rpcc2", input={"frame_index": 0})
    >>> cc = ColorCorrection()
    >>> fit = cc.fit("chart.jpg", "white.jpg", config=config)

The example expects a chart capture and its matching white field. Run
``python -m ColorCorrectionPipeline.examples.quickstart --output demo``
to use the included sample images after installing the package.
"""

import logging
from importlib import import_module

from .__version__ import __version__, __version_info__

# Library convention: never configure output for the application. Progress and
# diagnostic messages go to the "ColorCorrectionPipeline" logger hierarchy; enable
# them with e.g. logging.basicConfig(level=logging.INFO). See docs/logging.md.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Resolve public names to their original objects and modules on first access.
# Asking for the package version should not initialize training or image backends.
_EXPORTS = {
    "FitResult": "pipeline",
    "CalibrationError": "pipeline",
    "ColorCorrection": "pipeline",
    "Config": "config",
    "MyModels": "models",
    "FlatFieldCorrection": "flat_field",
    "MODEL_PATH": "constants",
    "ImageData": "contracts",
    "ReferenceProfile": "contracts",
    "ChartObservation": "contracts",
    "ObservationSet": "contracts",
    "ColorSemantics": "colorimetry",
    "StagePolicy": "stage_policies",
    "MonotonicToneModel": "stage_policies",
    "FittingPolicy": "fitting",
    "Steps": "config_groups",
    "FFCOptions": "config_groups",
    "GCOptions": "config_groups",
    "WBOptions": "config_groups",
    "CCOptions": "config_groups",
    "BundleLimits": "bundle",
    "BundleError": "bundle",
    "BatchItemResult": "results",
    "BatchResult": "results",
    "save_report": "reporting",
    "ExecutionOptions": "acceleration",
    "PreparedExecution": "acceleration",
    "AcceleratedResult": "acceleration",
    "LutCache": "acceleration",
    "AccuracyBudgetError": "acceleration",
    "core": "core",
    "io": "io",
    "calibrate": "simple",
    "correct": "simple",
    "Calibration": "simple",
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
__copyright__ = "Copyright (c) 2025-2026 Collins Wakholi"
