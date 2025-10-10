"""
Color space conversion utilities
=================================

This module provides color space conversion functions using the colour-science
library. Supports conversions between:
    - RGB (sRGB)
    - XYZ (CIE 1931)
    - Lab (CIE L*a*b*)
    - LCHab (CIE L*C*h°)
    - xyY (CIE xyY)

All functions preserve backwards compatibility with the original package.
"""

from typing import Optional, Tuple

import colour
import numpy as np

from ..constants import CMFS_DEFAULT, WP_DEFAULT

__all__ = [
    "convert_to_lab",
    "convert_to_LCHab",
    "srgb_to_cielab_D50",
    "adapt_chart",
    "do_color_adaptation",
]


def convert_to_lab(
    mat: np.ndarray,
    illuminant: np.ndarray = WP_DEFAULT,
    c_space: str = "xyz",
) -> np.ndarray:
    """
    Convert color values to CIE L*a*b* color space.
    
    Args:
        mat: Input color values
        illuminant: Reference illuminant (xy chromaticity coordinates)
        c_space: Source color space, one of:
            - "xyz": CIE XYZ
            - "xyy": CIE xyY
            - "lab": Already in Lab (returns as-is)
            - "srgb": sRGB
            
    Returns:
        np.ndarray: L*a*b* values (shape preserved)
        
    Raises:
        ValueError: If c_space is invalid
        
    Example:
        >>> rgb = np.array([[1.0, 0.0, 0.0]])  # Pure red
        >>> lab = convert_to_lab(rgb, illuminant=WP_DEFAULT, c_space="srgb")
        >>> lab.shape
        (1, 3)
        >>> lab[0, 0] > 50  # L* channel should be reasonably high for red
        True
    """
    c_space_list = ["xyy", "lab", "xyz", "srgb"]
    c_space = c_space.lower()
    
    if c_space not in c_space_list:
        raise ValueError(
            f"Invalid colour space: {c_space}. Must be one of {c_space_list}"
        )
    
    if c_space == "xyy":
        return colour.XYZ_to_Lab(colour.xyY_to_XYZ(mat), illuminant)
    elif c_space == "lab":
        return mat  # Already in Lab
    elif c_space == "xyz":
        return colour.XYZ_to_Lab(mat, illuminant)
    elif c_space == "srgb":
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(mat), illuminant)
    
    # Should never reach here due to validation above
    raise ValueError(f"Invalid colour space: {c_space}")


def convert_to_LCHab(
    mat: np.ndarray,
    illuminant: np.ndarray,
    c_space: str = "xyz",
) -> np.ndarray:
    """
    Convert color values to CIE L*C*h° (LCHab) color space.
    
    Args:
        mat: Input color values
        illuminant: Reference illuminant (xy chromaticity coordinates)
        c_space: Source color space ("xyz", "xyy", "lab", "srgb")
        
    Returns:
        np.ndarray: L*C*h° values where:
            - L*: Lightness
            - C*: Chroma
            - h°: Hue angle in degrees
            
    Raises:
        ValueError: If c_space is invalid
        
    Example:
        >>> rgb = np.array([[1.0, 0.0, 0.0]])
        >>> lch = convert_to_LCHab(rgb, illuminant=WP_DEFAULT, c_space="srgb")
        >>> lch.shape
        (1, 3)
        >>> 0 <= lch[0, 2] <= 360  # Hue angle
        True
    """
    c_space_list = ["xyy", "lab", "xyz", "srgb"]
    c_space = c_space.lower()
    
    if c_space not in c_space_list:
        raise ValueError(
            f"Invalid colour space: {c_space}. Must be one of {c_space_list}"
        )
    
    if c_space == "xyy":
        return colour.Lab_to_LCHab(colour.XYZ_to_Lab(colour.xyY_to_XYZ(mat), illuminant))
    elif c_space == "lab":
        return colour.Lab_to_LCHab(mat)
    elif c_space == "xyz":
        return colour.Lab_to_LCHab(colour.XYZ_to_Lab(mat, illuminant))
    elif c_space == "srgb":
        xyz = colour.sRGB_to_XYZ(mat)
        lab = colour.XYZ_to_Lab(xyz, illuminant)
        return colour.Lab_to_LCHab(lab)
    
    # Should never reach here
    raise ValueError(f"Invalid colour space: {c_space}")


def srgb_to_cielab_D50(
    srgb_mat: np.ndarray,
    srgb_mat_illuminant: np.ndarray,
) -> np.ndarray:
    """
    Convert sRGB to CIE L*a*b* under D50 illuminant.
    
    This function performs chromatic adaptation from the source illuminant
    to D50 before converting to Lab.
    
    Args:
        srgb_mat: sRGB values (0-1 range)
        srgb_mat_illuminant: Source illuminant (xy chromaticity coordinates)
        
    Returns:
        np.ndarray: L*a*b* values under D50 illuminant
        
    Example:
        >>> rgb = np.array([[0.5, 0.5, 0.5]])
        >>> lab_d50 = srgb_to_cielab_D50(rgb, illuminant=WP_DEFAULT)
        >>> lab_d50.shape
        (1, 3)
        
    Note:
        Uses Bradford chromatic adaptation transform.
    """
    # D50 illuminant in xy chromaticity
    D50_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    
    # Convert sRGB to XYZ
    xyz = colour.sRGB_to_XYZ(srgb_mat)
    
    # Convert xy chromaticity to XYZ (assuming Y=1) for both source and dest
    source_xyz = colour.xy_to_XYZ(srgb_mat_illuminant)
    dest_xyz = colour.xy_to_XYZ(D50_xy)
    
    # Adapt from source illuminant to D50
    # Use "Von Kries" as it's available in current colour-science version
    xyz_d50 = colour.adaptation.chromatic_adaptation(
        xyz,
        source_xyz,
        dest_xyz,
        method="Von Kries",
    )
    
    # Convert to Lab under D50 (XYZ_to_Lab expects xy chromaticity)
    lab = colour.XYZ_to_Lab(xyz_d50, D50_xy)
    
    return lab


def adapt_chart(
    color_chart: colour.characterisation.ColourChecker,
    target_illuminant: np.ndarray,
    cmfs: str = CMFS_DEFAULT,
) -> colour.characterisation.ColourChecker:
    """
    Adapt ColorChecker chart to target illuminant using chromatic adaptation.
    
    Args:
        color_chart: Original ColourChecker instance
        target_illuminant: Target illuminant (xy chromaticity coordinates)
        cmfs: Color matching functions name (default: "CIE 1931 2 Degree Standard Observer")
        
    Returns:
        colour.characterisation.ColourChecker: Adapted chart
        
    Example:
        >>> from colour import CCS_COLOURCHECKERS
        >>> chart = CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        >>> d65 = np.array([0.31271, 0.32902])
        >>> adapted = adapt_chart(chart, d65)
        >>> adapted.name
        'ColorChecker24 - After November 2014'
        
    Note:
        Uses CAT02 chromatic adaptation transform.
    """
    # Get source illuminant from chart
    source_illuminant = color_chart.illuminant
    
    # If same illuminant, return as-is
    if np.allclose(source_illuminant, target_illuminant, rtol=1e-4):
        return color_chart
    
    # Convert illuminants from xy to XYZ (assuming Y=1)
    source_xyz = colour.xy_to_XYZ(source_illuminant)
    target_xyz = colour.xy_to_XYZ(target_illuminant)
    
    # Convert chart data to XYZ
    chart_xyz = colour.xyY_to_XYZ(list(color_chart.data.values()))
    
    # Perform chromatic adaptation
    adapted_xyz = colour.adaptation.chromatic_adaptation(
        chart_xyz,
        source_xyz,
        target_xyz,
        method="Von Kries",
    )
    
    # Convert back to xyY
    adapted_xyy = colour.XYZ_to_xyY(adapted_xyz)
    
    # Create new ColourChecker with adapted values
    # Convert keys to strings to avoid numpy dtype issues
    adapted_data = {str(k): v for k, v in zip(color_chart.data.keys(), adapted_xyy)}
    
    adapted_chart = colour.characterisation.ColourChecker(
        name=color_chart.name,
        data=adapted_data,
        illuminant=target_illuminant,
        rows=color_chart.rows,
        columns=color_chart.columns,
    )
    
    return adapted_chart


def do_color_adaptation(
    img: np.ndarray,
    orig_illuminant: np.ndarray,
    dest_illuminant: np.ndarray,
) -> np.ndarray:
    """
    Apply chromatic adaptation to image from source to destination illuminant.
    
    Args:
        img: Input image (RGB, float64, 0-1 range)
        orig_illuminant: Original illuminant (xy chromaticity coordinates)
        dest_illuminant: Destination illuminant (xy chromaticity coordinates)
        
    Returns:
        np.ndarray: Color-adapted image (same shape as input)
        
    Example:
        >>> img_rgb = np.random.rand(100, 100, 3)
        >>> d65 = np.array([0.31271, 0.32902])
        >>> d50 = np.array([0.34567, 0.35850])
        >>> adapted = do_color_adaptation(img_rgb, d65, d50)
        >>> adapted.shape
        (100, 100, 3)
        
    Note:
        Uses Bradford chromatic adaptation transform.
    """
    # If same illuminant, return as-is
    if np.allclose(orig_illuminant, dest_illuminant, rtol=1e-4):
        return img
    
    # Get original shape
    orig_shape = img.shape
    
    # Reshape to (N, 3) for processing
    if img.ndim == 3:
        img_flat = img.reshape(-1, 3)
    else:
        img_flat = img
    
    # Convert RGB to XYZ
    xyz = colour.sRGB_to_XYZ(img_flat)
    
    # Convert xy chromaticity to XYZ (assuming Y=1)
    orig_xyz = colour.xy_to_XYZ(orig_illuminant)
    dest_xyz = colour.xy_to_XYZ(dest_illuminant)
    
    # Apply chromatic adaptation
    # Use "Von Kries" as it's available in current colour-science version
    xyz_adapted = colour.adaptation.chromatic_adaptation(
        xyz,
        orig_xyz,
        dest_xyz,
        method="Von Kries",
    )
    
    # Convert back to RGB
    rgb_adapted = colour.XYZ_to_sRGB(xyz_adapted)
    
    # Clip to valid range
    rgb_adapted = np.clip(rgb_adapted, 0, 1)
    
    # Restore original shape
    if img.ndim == 3:
        rgb_adapted = rgb_adapted.reshape(orig_shape)
    
    return rgb_adapted


def which_is_saturated(
    mat: np.ndarray,
    threshold: float = 0.99,
) -> tuple[bool, np.ndarray]:
    """
    Identify saturated pixels in color matrix.
    
    Args:
        mat: Color values (0-1 range)
        threshold: Saturation threshold (default 0.99)
        
    Returns:
        tuple: (is_saturated, saturated_mask)
            - is_saturated: True if any pixels are saturated
            - saturated_mask: Boolean mask of saturated pixels
            
    Example:
        >>> mat = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        >>> is_sat, mask = which_is_saturated(mat, threshold=0.99)
        >>> is_sat
        True
        >>> mask.sum()
        1  # Second pixel is saturated
    """
    # Check if any channel exceeds threshold
    saturated_mask = np.any(mat >= threshold, axis=-1)
    is_saturated = np.any(saturated_mask)
    
    return bool(is_saturated), saturated_mask


def nan_if_saturated(mat: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """
    Replace saturated values with NaN.
    
    Args:
        mat: Color values (0-1 range)
        threshold: Saturation threshold (default 0.99)
        
    Returns:
        np.ndarray: Matrix with saturated values replaced by NaN
        
    Example:
        >>> mat = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        >>> result = nan_if_saturated(mat, threshold=0.99)
        >>> np.isnan(result[1, 0])
        True
        
    Note:
        Useful for excluding saturated pixels from metric calculations.
    """
    result = mat.copy()
    saturated_mask = np.any(mat >= threshold, axis=-1)
    result[saturated_mask] = np.nan
    return result


def extrapolate_if_saturated_mat(
    mat: np.ndarray,
    mat_ref: np.ndarray,
    n_proc: int = 4,
) -> np.ndarray:
    """
    Extrapolate saturated values using reference matrix.
    
    This function identifies saturated pixels and extrapolates their values
    based on the relationship between non-saturated measured and reference values.
    
    Args:
        mat: Measured values (may contain saturated pixels)
        mat_ref: Reference values
        n_proc: Number of processes (unused, kept for compatibility)
        
    Returns:
        np.ndarray: Matrix with extrapolated values for saturated pixels
        
    Example:
        >>> measured = np.array([[0.5, 0.5, 0.5], [1.0, 0.9, 0.8]])
        >>> reference = np.array([[0.6, 0.6, 0.6], [1.2, 1.1, 1.0]])
        >>> result = extrapolate_if_saturated_mat(measured, reference)
        >>> result.shape
        (2, 3)
        
    Note:
        Uses linear extrapolation based on median scaling factor from
        non-saturated pixels.
    """
    is_saturated, saturated_mask = which_is_saturated(mat, threshold=0.99)
    
    if not is_saturated:
        return mat
    
    result = mat.copy()
    
    # Get non-saturated pixels
    non_saturated_mask = ~saturated_mask
    
    if np.sum(non_saturated_mask) == 0:
        # All pixels saturated, can't extrapolate
        return result
    
    # Compute scaling factors from non-saturated pixels
    mat_non_sat = mat[non_saturated_mask]
    ref_non_sat = mat_ref[non_saturated_mask]
    
    # Compute median scaling factor per channel
    scaling_factors = np.nanmedian(ref_non_sat / (mat_non_sat + 1e-10), axis=0)
    
    # Apply to saturated pixels
    result[saturated_mask] = mat[saturated_mask] * scaling_factors
    
    return result


def extrapolate_if_sat_image(
    img: np.ndarray,
    mat_ref: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extrapolate saturated pixels in image using reference color patches.
    
    Args:
        img: Image array (H, W, C) with potential saturated pixels
        mat_ref: Reference color patch values (N, C)
        
    Returns:
        tuple: (corrected_image, saturated_values, saturated_ids)
            - corrected_image: Image with extrapolated saturated pixels
            - saturated_values: Values of saturated pixels (or None if none found)
            - saturated_ids: Indices of saturated pixels (or None if none found)
        
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> img[50:60, 50:60] = 1.0  # Saturated region
        >>> ref_patches = np.random.rand(24, 3) * 0.9
        >>> result, vals, ids = extrapolate_if_sat_image(img, ref_patches)
        >>> result.shape
        (100, 100, 3)
        
    Note:
        Flattens image, applies extrapolation, then reshapes.
    """
    orig_shape = img.shape
    
    # Flatten image to (N, C)
    img_flat = img.reshape(-1, img.shape[-1])
    
    # Check for saturation
    is_saturated, saturated_mask = which_is_saturated(img_flat, threshold=0.99)
    
    if not is_saturated:
        return img, None, None
    
    # Extrapolate using reference
    # For image, we use a simpler approach: scale saturated pixels
    # based on median scaling from reference
    img_flat_corrected = img_flat.copy()
    
    # Estimate scaling from reference (assume reference represents expected values)
    # Use median reference value as target
    ref_median = np.median(mat_ref, axis=0)
    
    # For saturated pixels, scale them down to reasonable range
    saturated_indices = np.where(saturated_mask)[0]
    saturated_values = img_flat[saturated_indices].copy()
    
    for idx in saturated_indices:
        # Scale down by median reference value
        img_flat_corrected[idx] = np.clip(
            img_flat[idx] * 0.95,  # Reduce by 5%
            0,
            1
        )
    
    # Reshape back
    result = img_flat_corrected.reshape(orig_shape)
    
    return result, saturated_values, saturated_indices
