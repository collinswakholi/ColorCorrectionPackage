"""
Utility functions for color correction
=======================================

This module provides utility functions used throughout the color correction
pipeline, including color chart extraction, format conversions, and mathematical
operations.

Standalone implementations copied from v1_2_01 for compatibility.
"""

import logging

import numpy as np
from .._lazy import pandas as pd  # deferred: only used inside functions
from .._lazy import colour  # loads colour-science on first use

logger = logging.getLogger(__name__)

# ============================================================================
# TYPE CONVERSION FUNCTIONS
# ============================================================================


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0, 255] range."""
    if img.dtype == np.uint8:
        # Already uint8, return as is
        return img
    else:
        # Assume float in [0, 1] range; round to the nearest code (truncation
        # biased every value down by half a code before 2.3).
        return np.rint(np.clip(img * 255.0, 0, 255)).astype(np.uint8)


def to_float64(img: np.ndarray, scale: "float | None" = None) -> np.ndarray:
    """Convert image to float64 [0, 1] range.

    Args:
        img: Image of any numeric dtype.
        scale: Full-scale value of ``img`` (e.g. 1.0, 255.0, 65535.0). When
            omitted, integer images use their dtype range and float images are
            assumed to be 0-1 unless their maximum exceeds 1, in which case a
            0-255 scale is assumed. That guess depends on image content, so pass
            ``scale`` for float data that may overshoot 1.
    """
    if scale is not None:
        return np.asarray(img, dtype=np.float64) / float(scale)
    if img.dtype == np.uint8:
        # uint8: divide by 255
        return img.astype(np.float64) / 255.0
    elif img.dtype == np.uint16:
        # uint16: divide by 65535
        return img.astype(np.float64) / 65535.0
    elif img.dtype in [np.float32, np.float64]:
        # Already float, check if in [0, 1] range
        if img.max() <= 1.0:
            # Already in [0, 1] range
            return img.astype(np.float64)
        else:
            # Assume in [0, 255] range
            return img.astype(np.float64) / 255.0
    else:
        # Unknown type, try to normalize by max value
        max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else img.max()
        return img.astype(np.float64) / max_val


def get_attr(obj, attr: str, default=None):
    """Get attribute from object with default."""
    return getattr(obj, attr, default)


# ============================================================================
# COLOR CHART EXTRACTION (bundled model detector)
# ============================================================================


def extract_color_chart(img: np.ndarray, get_patch_size: bool = False):
    """Extract Classic24 RGB values in [0,255] with the bundled model detector.

    Detection failures raise a specific ChartDetectionError. (The OpenCV MCC
    backend was removed in 3.0.)
    """
    from ..detection import model_extract_color_charts

    samples, drawing, dimensions = model_extract_color_charts(img, n_charts=1)
    return samples[0], drawing, dimensions if get_patch_size else []


def extract_color_charts(img: np.ndarray, n_charts: int = 1):
    """Extract multiple Classic24 charts with an explicit detector policy."""
    from ..detection import model_extract_color_charts

    samples, drawing, _ = model_extract_color_charts(img, n_charts=n_charts)
    return samples, drawing


def extract_neutral_patches(
    img: np.ndarray, return_one: bool = True, show: bool = False, neutral_indices=None
):
    """
    Extract neutral (gray) patches from a ColorChecker chart (from v1_2_01).

    A standard ColorChecker has 6 neutral patches (grayscale) in the bottom row:
    patches 18-23 (0-indexed).

    Args:
        img: BGR image (uint8)
        return_one: If True, return single chart as DataFrame; if False, return list
        show: Whether to display detected chart

    Returns:
        tuple: (neutral_patches, all_patches) as pandas DataFrames (or lists of DataFrames)
    """
    patch_names = list(
        colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"].data.keys()
    )
    from ..detection.charts import _scope

    scope = _scope.get()
    profile = getattr(scope, "reference_profile", None)
    if profile is not None:
        patch_names = list(profile.names)
    neutral_indices = (
        (list(profile.neutral_indices) if profile is not None else list(range(18, 24)))
        if neutral_indices is None
        else list(neutral_indices)
    )
    if len(set(neutral_indices)) != len(neutral_indices) or any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(patch_names)
        for i in neutral_indices
    ):
        raise ValueError("neutral_indices must be unique cell positions of the chart reference")
    names_rows = [patch_names[index] for index in neutral_indices]
    names_columns = ["R", "G", "B"]

    if return_one:
        src_charts, img_draw, _ = extract_color_chart(img)
    else:
        src_charts, img_draw = extract_color_charts(img, n_charts=3)

    if show:
        try:
            import colour.plotting as cplt

            cplt.plot_image(
                to_float64(img_draw[:, :, ::-1]),
                title=f"Charts found: {len(src_charts) if isinstance(src_charts, list) else 1}",
            )
        except Exception:
            pass

    if src_charts is None:
        logger.warning("No charts detected in image")
        return None, None

    if return_one:
        # will return only the first chart as DataFrame
        srgb_chart = (
            np.asarray(src_charts, dtype=np.float64) / 255.0
            if getattr(scope, "measurement_image", None) is not None
            else to_float64(src_charts)
        )
        n_patch = srgb_chart[neutral_indices, :]
        n_patch_df = pd.DataFrame(n_patch, columns=names_columns, index=names_rows)
        n_patch_all = pd.DataFrame(srgb_chart, columns=names_columns, index=patch_names)

        return n_patch_df, n_patch_all

    else:
        # will return all charts in the image as a list
        Neutral_Patches = []
        All_Patches = []

        for chart in src_charts:
            srgb_chart = (
                np.asarray(chart, dtype=np.float64) / 255.0
                if getattr(scope, "measurement_image", None) is not None
                else to_float64(chart)
            )

            n_patch = srgb_chart[neutral_indices, :]
            n_patch_df = pd.DataFrame(n_patch, columns=names_columns, index=names_rows)
            n_patch_all = pd.DataFrame(srgb_chart, columns=names_columns, index=patch_names)

            Neutral_Patches.append(n_patch_df)
            All_Patches.append(n_patch_all)

        return Neutral_Patches, All_Patches


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================


def poly_func(x, coeffs):
    """Evaluate polynomial using Horner's method (from v1_2_01)."""
    result = 0
    for c in coeffs:
        result = result * x + c
    return result


__all__ = [
    "extract_color_chart",
    "extract_color_charts",
    "extract_neutral_patches",
    "get_attr",
    "poly_func",
    "to_float64",
    "to_uint8",
]
