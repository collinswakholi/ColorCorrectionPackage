"""Version information for ColorCorrectionPipeline package."""

__version__ = "1.3.4"
__version_info__ = tuple(int(i) for i in __version__.split("."))

__all__ = ["__version__", "__version_info__"]

# Release 1.3.4: Optimized chart detection with production-ready extract_color_chart_ex
# - Fixed negative dimension bug with adaptive sampling
# - Added colour-checker-detection dependency
# - Comprehensive error handling and validation
# - Vectorized operations for performance
# - 100% test success rate
