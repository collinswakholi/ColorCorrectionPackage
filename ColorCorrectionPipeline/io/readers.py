"""
Image reading utilities
=======================

This module provides utilities for reading images with proper color space
handling and format conversion.

Functions:
    - read_image: Read single image from file
    - read_batch: Read multiple images in batch
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..core.utils import to_float64

__all__ = ["read_image", "read_batch"]


def read_image(
    path: Union[str, Path],
    color_space: str = "rgb",
    dtype: str = "float64",
    range_: str = "0-1",
) -> np.ndarray:
    """
    Read image from file with proper color space handling.
    
    Args:
        path: Path to image file
        color_space: Desired color space ('rgb', 'bgr', 'gray')
        dtype: Output dtype ('float32', 'float64', 'uint8', 'uint16')
        range_: Output range ('0-1' for float, '0-255' for uint8, '0-65535' for uint16)
        
    Returns:
        np.ndarray: Image array with shape (H, W, C) or (H, W) for grayscale
        
    Raises:
        FileNotFoundError: If image file cannot be read
        ValueError: If invalid color_space or dtype
        
    Example:
        >>> img_rgb = read_image("sample.jpg", color_space="rgb", dtype="float64")
        >>> print(img_rgb.shape)  # (H, W, 3)
        >>> print(img_rgb.dtype)  # float64
        >>> print(img_rgb.min(), img_rgb.max())  # 0.0 1.0
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Read with OpenCV (always reads as BGR)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from '{path}'")
    
    # Handle grayscale
    if len(img.shape) == 2:
        if color_space.lower() != "gray":
            # Convert grayscale to color if requested
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # Keep as grayscale
            pass
    else:
        # Handle color space conversion
        if color_space.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space.lower() == "bgr":
            pass  # Already BGR
        elif color_space.lower() == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Invalid color_space: {color_space}. Use 'rgb', 'bgr', or 'gray'.")
    
    # Handle dtype conversion
    if dtype.lower() in ("float32", "float64"):
        if img.dtype == np.uint8:
            img = img.astype(np.float32 if dtype.lower() == "float32" else np.float64)
            if range_ == "0-1":
                img /= 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32 if dtype.lower() == "float32" else np.float64)
            if range_ == "0-1":
                img /= 65535.0
        else:
            img = img.astype(np.float32 if dtype.lower() == "float32" else np.float64)
    elif dtype.lower() == "uint8":
        if img.dtype in (np.float32, np.float64):
            if range_ == "0-1":
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img // 257).astype(np.uint8)  # Scale 0-65535 to 0-255
        else:
            img = img.astype(np.uint8)
    elif dtype.lower() == "uint16":
        if img.dtype in (np.float32, np.float64):
            if range_ == "0-1":
                img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
            else:
                img = np.clip(img, 0, 65535).astype(np.uint16)
        elif img.dtype == np.uint8:
            img = (img.astype(np.uint16) * 257)  # Scale 0-255 to 0-65535
        else:
            img = img.astype(np.uint16)
    else:
        raise ValueError(
            f"Invalid dtype: {dtype}. Use 'float32', 'float64', 'uint8', or 'uint16'."
        )
    
    return img


def read_batch(
    paths: List[Union[str, Path]],
    color_space: str = "rgb",
    dtype: str = "float64",
    range_: str = "0-1",
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read multiple images in batch.
    
    Args:
        paths: List of image file paths
        color_space: Desired color space ('rgb', 'bgr', 'gray')
        dtype: Output dtype ('float32', 'float64', 'uint8', 'uint16')
        range_: Output range ('0-1' for float, '0-255' for uint8)
        verbose: Whether to print progress
        
    Returns:
        tuple: (images, failed_paths)
            - images: List of successfully read images
            - failed_paths: List of paths that failed to read
            
    Example:
        >>> paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> images, failed = read_batch(paths, color_space="rgb", dtype="float64")
        >>> print(f"Read {len(images)} images, {len(failed)} failed")
    """
    images: List[np.ndarray] = []
    failed_paths: List[str] = []
    
    for i, path in enumerate(paths):
        try:
            img = read_image(path, color_space=color_space, dtype=dtype, range_=range_)
            images.append(img)
            if verbose:
                print(f"Info: Read image {i+1}/{len(paths)}: {path}")
        except Exception as e:
            failed_paths.append(str(path))
            if verbose:
                print(f"Error: Failed to read image {i+1}/{len(paths)}: {path} - {e}")
    
    if verbose:
        print(f"Info: Successfully read {len(images)}/{len(paths)} images")
        if failed_paths:
            print(f"Warning: Failed to read {len(failed_paths)} images")
    
    return images, failed_paths
