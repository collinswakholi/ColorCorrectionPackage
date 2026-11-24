"""
Utility functions for color correction
=======================================

This module provides utility functions used throughout the color correction
pipeline, including color chart extraction, format conversions, and mathematical
operations.

Standalone implementations copied from v1_2_01 for compatibility.
"""

import hashlib
import logging
import warnings

import numpy as np
from .._lazy import pandas as pd  # deferred: only used inside functions
import cv2
from .._lazy import colour  # loads colour-science on first use

logger = logging.getLogger(__name__)

# ── MCC module availability (follows accel.py HAS_NUMBA pattern) ───────────
HAS_MCC = False
try:
    _mcc = cv2.mcc
    HAS_MCC = hasattr(_mcc, "CCheckerDetector") or hasattr(_mcc, "CCheckerDetector_create")
except AttributeError:
    pass


# ============================================================================
# CHART DETECTION CACHE
# ============================================================================


class _ChartCache:
    """Cache chart detection results keyed by image content hash.

    MCC detection is the most frequently called expensive operation
    (6-10 times per pipeline run, ~1-2s each). Caching the detection
    result saves 5-15s per pipeline run.
    """

    def __init__(self, max_entries: int = 8):
        self._cache = {}  # hash -> (best_checker_data, img_draw, src_patches)
        self._max = max_entries

    @staticmethod
    def _hash(img: np.ndarray) -> str:
        """Identify all pixels, shape and dtype; unsampled changes matter."""
        digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
        return "%s|%s|%s" % (img.shape, img.dtype, digest)

    def get(self, img: np.ndarray):
        """Return cached (src_patches, img_draw, dims) or None."""
        h = self._hash(img)
        return self._cache.get(h)

    def put(self, img: np.ndarray, result: tuple):
        """Store result for this image."""
        if len(self._cache) >= self._max:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        h = self._hash(img)
        self._cache[h] = result

    def clear(self):
        self._cache.clear()


# Module-level chart cache instance
_chart_cache = _ChartCache()

# Standalone implementations (no dependency on old package)

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
# COLOR CHART DETECTION (OpenCV MCC)
# ============================================================================


def _create_detector_params():
    """Create MCC DetectorParameters with version-safe fallback."""
    for factory in [
        lambda: cv2.mcc.DetectorParameters.create(),
        lambda: cv2.mcc.DetectorParameters_create(),
        lambda: cv2.mcc.DetectorParameters(),
        lambda: cv2.mcc.DetectorParametersMCC(),  # OpenCV >= 4.11 / 5.x
    ]:
        try:
            return factory()
        except Exception:
            continue
    return None


def _create_detector():
    """Create MCC CCheckerDetector with version-safe fallback."""
    for factory in [
        lambda: cv2.mcc.CCheckerDetector.create(),
        lambda: cv2.mcc.CCheckerDetector_create(),
    ]:
        try:
            return factory()
        except Exception:
            continue
    raise RuntimeError(
        "Cannot create cv2.mcc.CCheckerDetector — check opencv-contrib-python install"
    )


def _create_cdraw(checker):
    """Create MCC CCheckerDraw with version-safe fallback."""
    for factory in [
        lambda: cv2.mcc.CCheckerDraw.create(checker),
        lambda: cv2.mcc.CCheckerDraw_create(checker),
    ]:
        try:
            return factory()
        except Exception:
            continue
    if hasattr(cv2.mcc, "CCheckerDetector"):
        # OpenCV >= 4.11 / 5.x removed CCheckerDraw; drawing moved to the detector.
        return _DetectorDraw(checker)
    raise RuntimeError("Cannot create cv2.mcc.CCheckerDraw")


class _DetectorDraw:
    """CCheckerDraw stand-in for OpenCV versions that draw through the detector."""

    def __init__(self, checker):
        self._checker = checker

    def draw(self, img):
        out = _create_detector().draw([self._checker], img)
        if out is not None and out is not img:
            img[...] = out


def _mcc_process(detector, img, nc=1, params=None):
    """Run MCC24 chart detection across OpenCV API generations.

    OpenCV <= 4.10 takes ``process(img, chartType, nc=..., params=...)``.
    OpenCV >= 4.11 / 5.x takes ``process(img[, nc])`` with the chart type and
    parameters set on the detector; passing ``MCC24`` positionally there is
    read as ``nc=0`` and silently finds nothing.
    """
    if hasattr(detector, "setColorChartType"):
        detector.setColorChartType(cv2.mcc.MCC24)
        if params is not None and hasattr(detector, "setDetectionParams"):
            detector.setDetectionParams(params)
        return detector.process(img, int(nc))
    if params is not None:
        return detector.process(img, cv2.mcc.MCC24, nc=nc, params=params)
    return detector.process(img, cv2.mcc.MCC24, nc=nc)


def _set_param_safe(params, name, value):
    """Set a DetectorParameters property, silently ignoring failures."""
    try:
        setattr(params, name, value)
    except Exception:
        pass


def _make_adjusted_params():
    """Build a DetectorParameters with relaxed thresholds (version-safe)."""
    dp = _create_detector_params()
    if dp is not None:
        _set_param_safe(dp, "adaptiveThreshWinSizeMin", 3)
        _set_param_safe(dp, "adaptiveThreshWinSizeStep", 8)
        _set_param_safe(dp, "confidenceThreshold", 0.50)
        _set_param_safe(dp, "maxError", 0.35)
        _set_param_safe(dp, "minGroupSize", 4)
    return dp


# Module-level default params (created once)
CDP = _create_detector_params()


def detect_refine_charts(detector, img, n_charts, params=None):
    """Detect and refine color charts using OpenCV's mcc module (from v1_2_01)."""
    if params is None:
        params = CDP
    n = n_charts + 1  # increase n for extra chart detections

    def run_detection(current_params):
        # Try with params first, fall back to without params
        detected = False
        for attempt_params in [current_params, None]:
            try:
                _mcc_process(detector, img, n, attempt_params)
                detected = True
                break
            except Exception:
                continue
        if not detected:
            # Last resort: minimal call
            _mcc_process(detector, img, n)
        list_ = detector.getListColorChecker()
        if len(list_) < 1:
            logger.debug("No color chart found in image on this detection attempt")
            return None
        return detector

    try:
        det_ = run_detection(params)
    except Exception:
        logger.debug("Retrying chart detection with adjusted parameters")
        adjusted = _make_adjusted_params()
        det_ = run_detection(adjusted)

    if det_ is None:
        logger.debug("Retrying chart detection with adjusted parameters")
        det_ = run_detection(_make_adjusted_params())
    if det_ is None:
        return []

    if n_charts == 1:
        return [det_.getBestColorChecker()]

    elif n_charts > 1:
        charts = det_.getListColorChecker()
        charts = sorted(charts, key=lambda c: c.getCost())
        len_ = len(charts)
        if (len_ - 1) < n_charts:
            warnings.warn(f"Detected {len(charts)-1} charts instead of {n_charts}.", stacklevel=2)
        elif (len_ - 1) >= n_charts:
            charts = sorted(charts, key=lambda c: c.getCost())[:n_charts]
        return charts
    else:
        return None


def _extract_color_chart_mcc(img: np.ndarray, get_patch_size: bool = False):
    """
    Extract color patches from a ColorChecker chart in the image (from v1_2_01).

    Uses a per-image cache so repeated calls on the same image (different
    pipeline stages) skip the expensive MCC detection.

    Args:
        img: BGR image (uint8)
        get_patch_size: If True, return patch size info

    Returns:
        tuple: (patches, img_draw, dims)
            - patches: RGB values (24, 3) float64 in [0,1] range or None
            - img_draw: Image with drawn chart
            - dims: Patch size info
    """
    # ── Cache lookup ──
    cached = _chart_cache.get(img)
    if cached is not None:
        src, img_draw, dims = cached
        if not get_patch_size:
            return np.array(src), img_draw.copy(), []
        return np.array(src), img_draw.copy(), dims

    # ── Full detection (expensive) ──
    img_blur = cv2.medianBlur(img, 5)
    detector = _create_detector()
    checkers = detect_refine_charts(detector, img_blur, n_charts=1, params=CDP)
    best_checker = checkers[0] if checkers else None

    if best_checker is None:
        logger.warning("No color chart found in image")
        return None, None, None

    cdraw = _create_cdraw(best_checker)
    img_draw = img.copy()
    cdraw.draw(img_draw)

    chartSRGB = best_checker.getChartsRGB()
    w, _ = chartSRGB.shape[:2]
    roi = chartSRGB[0:w, 1]

    box_pts = best_checker.getBox()
    x1, x2 = int(min(box_pts[:, 0])), int(max(box_pts[:, 0]))
    y1, y2 = int(min(box_pts[:, 1])), int(max(box_pts[:, 1]))

    rows = roi.shape[0]
    src = chartSRGB[:, 1].reshape(int(rows / 3), 1, 3).reshape(24, 3)

    dims = []
    if get_patch_size:
        # Simple patch size estimation
        dims = [(x2 - x1) // 6, (y2 - y1) // 4]

    # ── Store in cache ──
    _chart_cache.put(img, (np.array(src), img_draw.copy(), dims))

    return np.array(src), img_draw, dims


def _extract_color_charts_mcc(img: np.ndarray, n_charts: int = 1):
    """
    Extract multiple color charts from image (from v1_2_01).

    Args:
        img: BGR image (uint8)
        n_charts: Number of charts to extract

    Returns:
        tuple: (charts_list, img_draw)
            - charts_list: List of (24, 3) arrays with RGB values
            - img_draw: Image with drawn charts
    """
    img_blur = cv2.medianBlur(img, 5)
    detector = _create_detector()

    checkers = detect_refine_charts(detector, img_blur, n_charts=n_charts, params=CDP)

    if checkers is None:
        return [], img

    charts = []
    img_draw = img.copy()

    for i, checker in enumerate(checkers):
        cdraw = _create_cdraw(checker)
        cdraw.draw(img_draw)

        # Add text to chart
        box_pts = checker.getBox()
        x1 = int(min(box_pts[:, 0]))
        y1 = int(min(box_pts[:, 1]))

        cv2.putText(
            img_draw,
            f"Chart {i+1}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            int(img.shape[0] / 600),
            (0, 0, 255),
            int(img.shape[0] / 400),
        )

        chartSRGB = checker.getChartsRGB()
        w, _ = chartSRGB.shape[:2]
        roi = chartSRGB[0:w, 1]

        rows = roi.shape[0]
        src = chartSRGB[:, 1].reshape(int(rows / 3), 1, 3).reshape(24, 3)
        charts.append(np.array(src))

    return charts, img_draw


def extract_color_chart(img: np.ndarray, get_patch_size: bool = False):
    """Extract Classic24 RGB values in [0,255] using the selected detector.

    The default uses the bundled model. Select ``backend='mcc'`` inside a
    ``detection_context`` for legacy MCC behavior. Model errors raise a specific
    ChartDetectionError unless MCC fallback was explicitly enabled.
    """
    from ..detection import model_extract_color_charts

    detected = model_extract_color_charts(img, n_charts=1)
    if detected is None:
        return _extract_color_chart_mcc(img, get_patch_size=get_patch_size)
    samples, drawing, dimensions = detected
    return samples[0], drawing, dimensions if get_patch_size else []


def extract_color_charts(img: np.ndarray, n_charts: int = 1):
    """Extract multiple Classic24 charts with an explicit detector policy."""
    from ..detection import model_extract_color_charts

    detected = model_extract_color_charts(img, n_charts=n_charts)
    if detected is None:
        return _extract_color_charts_mcc(img, n_charts=n_charts)
    samples, drawing, _ = detected
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
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < 24
        for i in neutral_indices
    ):
        raise ValueError("neutral_indices must be unique Classic24 cell positions")
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
    "clear_chart_cache",
    "extract_color_chart",
    "extract_color_charts",
    "extract_neutral_patches",
    "get_attr",
    "poly_func",
    "to_float64",
    "to_uint8",
]


def clear_chart_cache():
    """Clear the chart detection cache (call between different images)."""
    _chart_cache.clear()
