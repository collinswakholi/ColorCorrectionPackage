"""Version information for ColorCorrectionPipeline package."""

__version__ = "1.4.4"
__version_info__ = tuple(int(i) for i in __version__.split("."))

__all__ = ["__version__", "__version_info__"]

# Release 1.4.4: Publication citation and pipeline hardening
# - Added the associated Plant Phenome Journal manuscript citation to README
# - Hardened fit-time/predict-time parity, model persistence, and config validation
# - Improved finite white-balance factors and CPU/GPU-safe custom NN training paths
#
# Release 1.4.3: Unified hidden_layers API for neural-network CC
# - Replaced nlayers attribute with hidden_layers list in Regressor_Model
# - MLPRegressor now uses tuple(M.hidden_layers) for hidden_layer_sizes
# - Backwards compatible: old nlayers key auto-converts to [int(nlayers)]
# - Updated pipeline.py to pass hidden_layers instead of nlayers
#
# Release 1.4.2: Fixed opencv-python / opencv-contrib-python conflict
# - Removed opencv-python from dependencies (contrib is a strict superset)
# - Added HAS_MCC runtime flag in core/utils.py (follows accel.py pattern)
# - Import-time warning when cv2.mcc is missing
# - Consolidated all cv2.mcc API access through version-safe factories
# - Added tests/test_dependencies.py for post-install module verification
#
# Release 1.4.1: numba promoted to default (core) dependency
# - numba>=0.55.0 moved from optional [accel] extra to core dependencies
# - Hardware acceleration now active out-of-the-box on all supported platforms
#
# Release 1.4.0: Numba/CUDA acceleration + batch prediction API
# - core/accel.py: hardware-aware kernels (Numba CPU parallel / CUDA)
# - Fast sRGB<->Lab via pre-computed LUTs + Numba parallel (22-54x faster)
# - 3D LUT trilinear interpolation for CC prediction (MAE < 0.0001)
# - chart detection cache (350x+ faster on cache hit)
# - predict_images(): parallel batch prediction with progress callback
# - apply_ffc_float(): FFC in float64 (no uint8 round-trip)
# - Fixed OOM retry direction in CustomNN.predict() (batch_size //= 2)
# - Fixed uint8 clipping bug in FlatFieldCorrection.apply_ffc()
# - avg 1.41x end-to-end speedup over 1.3.4 (up to 1.50x on NN configs)
