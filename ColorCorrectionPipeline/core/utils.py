"""
Utility functions for color correction
=======================================

This module provides utility functions used throughout the color correction
pipeline, including color chart extraction, format conversions, and mathematical
operations.

Re-exports from ColorCorrectionPipeline.key_functions for compatibility.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Import from the old package for now
try:
    from ColorCorrectionPipeline.key_functions import (
        compute_diag,
        compute_temperature,
        estimate_fit,
        extract_color_chart,
        extract_color_charts,
        extract_neutral_patches,
        free_memory,
        get_attr,
        poly_func,
        poly_func_torch,
        to_float64,
        to_uint8,
    )
except ImportError:
    # Fallback implementations if old package not available
    print("Warning: ColorCorrectionPipeline not found. Some functions may not work.")
    
    def to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert float image to uint8."""
        if img.dtype == np.uint8:
            return img
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    
    def to_float64(img: np.ndarray) -> np.ndarray:
        """Convert image to float64 [0, 1] range."""
        if img.dtype == np.float64:
            # Already float64, check if needs scaling
            if img.max() <= 1.0:
                return img
            # Assume 0-255 range float
            return img / 255.0
        elif img.dtype == np.float32 or img.dtype == np.float16:
            # Float types - check range
            if img.max() <= 1.0:
                return img.astype(np.float64)
            return img.astype(np.float64) / 255.0
        elif img.dtype == np.uint8:
            return img.astype(np.float64) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float64) / 65535.0
        else:
            # Default: assume needs scaling from max value
            max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 255.0
            return img.astype(np.float64) / max_val
    
    def get_attr(obj, attr: str, default=None):
        """Get attribute from object with default."""
        return getattr(obj, attr, default)
    
    def extract_color_chart(img: np.ndarray, get_patch_size: bool = False):
        """
        Extract color patches from a ColorChecker chart in the image.
        
        Args:
            img: BGR image (uint8)
            get_patch_size: If True, return patch size info
            
        Returns:
            tuple: (patches, bbox, patch_size) or (patches, bbox) if get_patch_size=False
                - patches: RGB values (24, 3) float64 in [0,1] range
                - bbox: Bounding box of detected chart (x, y, w, h)
                - patch_size: Size of individual patches
        """
        try:
            from colour_checker_detection import detect_colour_checkers_segmentation
            import cv2
            
            # Detect color checker
            swatches_data = detect_colour_checkers_segmentation(img, show=False)
            
            if not swatches_data or len(swatches_data) == 0:
                return (None, None, None) if get_patch_size else (None, None)
            
            # Get first detected chart
            swatch = swatches_data[0]
            colour_checker_swatches = swatch.values
            
            # Extract mean color from each patch (already in RGB 0-1 range)
            patches = np.array([np.mean(patch, axis=0) for patch in colour_checker_swatches])
            
            # Estimate bounding box (rough estimate)
            bbox = (0, 0, img.shape[1], img.shape[0])
            patch_size = (50, 50)  # Rough estimate
            
            if get_patch_size:
                return patches, bbox, patch_size
            return patches, bbox
            
        except Exception as e:
            print(f"Warning: Color chart detection failed: {e}")
            return (None, None, None) if get_patch_size else (None, None)
    
    def extract_color_charts(img: np.ndarray, n_charts: int = 1):
        """
        Extract multiple color charts from image.
        
        Args:
            img: BGR image (uint8)
            n_charts: Number of charts to extract
            
        Returns:
            list: List of (patches, bbox) tuples for each detected chart
        """
        try:
            from colour_checker_detection import detect_colour_checkers_segmentation
            
            swatches_data = detect_colour_checkers_segmentation(img, show=False)
            
            if not swatches_data:
                return []
            
            results = []
            for swatch in swatches_data[:n_charts]:
                colour_checker_swatches = swatch.values
                patches = np.array([np.mean(patch, axis=0) for patch in colour_checker_swatches])
                bbox = (0, 0, img.shape[1], img.shape[0])
                results.append((patches, bbox))
            
            return results
            
        except Exception as e:
            print(f"Warning: Multiple color chart detection failed: {e}")
            return []
    
    def extract_neutral_patches(img: np.ndarray, return_one: bool = True, show: bool = False):
        """
        Extract neutral (gray) patches from a ColorChecker chart.
        
        A standard ColorChecker has 6 neutral patches (grayscale) in the bottom row:
        patches 18-23 (0-indexed).
        
        Args:
            img: BGR image (uint8)
            return_one: If True, return single chart; if False, return multiple
            show: Whether to display detected chart
            
        Returns:
            tuple: (all_patches, neutral_patches)
                - all_patches: All 24 patches RGB (24, 3) in [0,1] range or None
                - neutral_patches: 6 neutral patches RGB (6, 3) in [0,1] range or None
        """
        try:
            from colour_checker_detection import detect_colour_checkers_segmentation
            
            swatches_data = detect_colour_checkers_segmentation(img, show=show)
            
            if not swatches_data or len(swatches_data) == 0:
                return None, None
            
            # Get first chart if return_one, otherwise all
            charts_to_process = swatches_data[:1] if return_one else swatches_data
            
            all_results = []
            neutral_results = []
            
            for swatch in charts_to_process:
                colour_checker_swatches = swatch.values
                
                # Extract mean color from each patch (already in RGB 0-1 range)
                all_patches = np.array([np.mean(patch, axis=0) for patch in colour_checker_swatches])
                
                # Standard ColorChecker layout: neutral patches are indices 18-23 (bottom row)
                # These are the grayscale patches from white to black
                if len(all_patches) >= 24:
                    neutral_patches = all_patches[18:24]  # Last 6 patches
                else:
                    # Fallback: assume last 6 patches are neutral
                    neutral_patches = all_patches[-6:]
                
                all_results.append(all_patches)
                neutral_results.append(neutral_patches)
            
            if return_one:
                return all_results[0] if all_results else None, neutral_results[0] if neutral_results else None
            else:
                return all_results, neutral_results
                
        except Exception as e:
            print(f"Warning: Neutral patch extraction failed: {e}")
            return None, None
    
    def compute_diag(vec: np.ndarray) -> np.ndarray:
        """Create diagonal matrix from vector."""
        return np.diag(vec)
    
    def compute_temperature(xyz: np.ndarray) -> float:
        """Compute color temperature."""
        raise NotImplementedError("compute_temperature requires ColorCorrectionPipeline package")
    
    def estimate_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """Fit polynomial."""
        return np.polyfit(x, y, degree)
    
    def poly_func(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate polynomial."""
        return np.polyval(coeffs, x)
    
    def poly_func_torch(x, coeffs):
        """Evaluate polynomial using torch."""
        raise NotImplementedError("poly_func_torch requires ColorCorrectionPipeline package")
    
    def free_memory():
        """Free GPU memory."""
        pass

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
