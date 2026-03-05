"""Version information for ColorCorrectionPipeline package."""

__version__ = "1.4.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))

__all__ = ["__version__", "__version_info__"]

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
