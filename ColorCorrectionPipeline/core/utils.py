"""
Utility functions for color correction
=======================================

This module provides utility functions used throughout the color correction
pipeline, including color chart extraction, format conversions, and mathematical
operations.

Standalone implementations copied from v1_2_01 for compatibility.
"""

from typing import Optional, Tuple
import gc

import numpy as np
import pandas as pd
import cv2
import colour

# Standalone implementations (no dependency on old package)

# ============================================================================
# TYPE CONVERSION FUNCTIONS
# ============================================================================

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float image to uint8 (from v1_2_01)."""
    return (img * 255).astype(np.uint8)

def to_float64(img: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float64 [0, 1] range (from v1_2_01)."""
    return img.astype(np.float64) / 255.0

def get_attr(obj, attr: str, default=None):
    """Get attribute from object with default."""
    return getattr(obj, attr, default)

def free_memory():
    """Free GPU memory if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()

# ============================================================================
# COLOR CHART DETECTION (OpenCV MCC)
# ============================================================================

# OpenCV mcc detector parameters (from v1_2_01)
CDP = cv2.mcc.DetectorParameters().create()

def detect_refine_charts(detector, img, n_charts, params=CDP):
    """Detect and refine color charts using OpenCV's mcc module (from v1_2_01)."""
    # Use a higher count for chart detection
    n = n_charts + 1  # increase n for extra chart detections

    def run_detection(current_params):
        detector.process(img, cv2.mcc.MCC24, nc=n, params=current_params)
        list_ = detector.getListColorChecker()
        if len(list_) < 0.5:
            print("Warning: No color chart found in Image")
            assert False
        return detector

    try:
        det_ = run_detection(params)
    except Exception:
        print("Warning: Retrying with Adjusted parameters...")
        DP = cv2.mcc.DetectorParameters().create()
        DP.adaptiveThreshWinSizeMin = 3
        DP.adaptiveThreshWinSizeStep = 8
        DP.confidenceThreshold = 0.50
        DP.maxError = 0.35
        DP.minGroupSize = 4
        det_ = run_detection(DP)

    if n_charts == 1:
        return [det_.getBestColorChecker()]

    elif n_charts > 1:
        charts = det_.getListColorChecker()
        charts = sorted(charts, key=lambda c: c.getCost())
        len_ = len(charts)
        if (len_ - 1) < n_charts:
            print(f"Warning: Detected {len(charts)-1} charts instead of {n_charts}.")
        elif (len_ - 1) >= n_charts:
            charts = sorted(charts, key=lambda c: c.getCost())[:n_charts]
        return charts
    else:
        return None

def extract_color_chart(img: np.ndarray, get_patch_size: bool = False):
    """
    Extract color patches from a ColorChecker chart in the image (from v1_2_01).
    
    Args:
        img: BGR image (uint8)
        get_patch_size: If True, return patch size info
        
    Returns:
        tuple: (patches, img_draw, dims) 
            - patches: RGB values (24, 3) float64 in [0,1] range or None
            - img_draw: Image with drawn chart
            - dims: Patch size info
    """
    img_blur = cv2.medianBlur(img, 5)
    detector = cv2.mcc.CCheckerDetector_create()
    best_checker = detect_refine_charts(detector, img_blur, n_charts=1, params=CDP)[0]

    if best_checker is None:
        print("Warning: No color chart found in Image")
        return None, None, None

    cdraw = cv2.mcc.CCheckerDraw_create(best_checker)
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
        img_roi = img[y1:y2, x1:x2]
        # Simple patch size estimation
        dims = [(x2-x1)//6, (y2-y1)//4]

    return np.array(src), img_draw, dims

def extract_color_charts(img: np.ndarray, n_charts: int = 1):
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
    detector = cv2.mcc.CCheckerDetector_create()

    checkers = detect_refine_charts(detector, img_blur, n_charts=n_charts, params=CDP)

    if checkers is None:
        return [], img

    charts = []
    img_draw = img.copy()

    for i, checker in enumerate(checkers):
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        cdraw.draw(img_draw)
        
        # Add text to chart
        box_pts = checker.getBox()
        x1, x2 = int(min(box_pts[:, 0])), int(max(box_pts[:, 0]))
        y1, y2 = int(min(box_pts[:, 1])), int(max(box_pts[:, 1]))

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

def extract_neutral_patches(img: np.ndarray, return_one: bool = True, show: bool = False):
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
    names_rows = patch_names[-6:]
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
                title=f"Charts found: {len(src_charts) if isinstance(src_charts, list) else 1}"
            )
        except:
            pass

    if src_charts is None:
        print("Warning: No charts detected in Image")
        return None, None

    if return_one:
        # will return only the first chart as DataFrame
        srgb_chart = to_float64(src_charts)
        n_patch = srgb_chart[-6:, :]
        n_patch_df = pd.DataFrame(n_patch, columns=names_columns, index=names_rows)
        n_patch_all = pd.DataFrame(srgb_chart, columns=names_columns, index=patch_names)

        return n_patch_df, n_patch_all

    else:
        # will return all charts in the image as a list
        Neutral_Patches = []
        All_Patches = []

        for chart in src_charts:
            srgb_chart = to_float64(chart)

            n_patch = srgb_chart[-6:, :]
            n_patch_df = pd.DataFrame(n_patch, columns=names_columns, index=names_rows)
            n_patch_all = pd.DataFrame(
                srgb_chart, columns=names_columns, index=patch_names
            )

            Neutral_Patches.append(n_patch_df)
            All_Patches.append(n_patch_all)

        return Neutral_Patches, All_Patches

# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

def compute_diag(mat_ref, mat_det, rf=0.95):
    """Compute diagonal correction matrix (from v1_2_01)."""
    factors = np.nanmedian(mat_ref / mat_det, axis=0)
    diag = np.diag(rf * factors)
    return diag

def compute_temperature(xyz: np.ndarray) -> float:
    """Compute color temperature."""
    raise NotImplementedError("compute_temperature requires ColorCorrectionPipeline package")

def estimate_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit polynomial (from v1_2_01)."""
    return np.polyfit(x, y, degree)

def poly_func(x, coeffs):
    """Evaluate polynomial using Horner's method (from v1_2_01)."""
    result = 0
    for c in coeffs:
        result = result * x + c
    return result

def poly_func_torch(x, coeffs):
    """Evaluate polynomial using torch and Horner's method (from v1_2_01)."""
    try:
        import torch
        result = torch.zeros_like(x)
        for c in coeffs:
            result = result * x + c
        return result
    except:
        raise NotImplementedError("poly_func_torch requires torch")

__all__ = [
    "compute_diag",
    "compute_temperature",
    "estimate_fit",
    "extract_color_chart",
    "extract_color_charts",
    "extract_neutral_patches",
    "free_memory",
    "get_attr",
    "poly_func",
    "poly_func_torch",
    "to_float64",
    "to_uint8",
]
