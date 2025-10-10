"""
Image writing utilities
=======================

This module provides utilities for writing images with proper color space
handling and format conversion.

Functions:
    - write_image: Write image to file
"""

import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

__all__ = ["write_image"]


def write_image(
    path: Union[str, Path],
    image: np.ndarray,
    color_space: str = "rgb",
    quality: int = 95,
    create_dirs: bool = True,
) -> bool:
    """
    Write image to file with proper color space handling.
    
    Args:
        path: Output file path
        image: Image array (H, W, C) or (H, W)
        color_space: Input color space ('rgb', 'bgr', 'gray')
        quality: JPEG quality (1-100) or PNG compression (0-9)
        create_dirs: Whether to create parent directories
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> img_rgb = np.random.rand(480, 640, 3).astype(np.float64)
        >>> success = write_image("output.jpg", img_rgb, color_space="rgb", quality=95)
        >>> print(f"Write {'succeeded' if success else 'failed'}")
    """
    path = Path(path)
    
    # Create parent directories if needed
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype in (np.float32, np.float64):
        # Assume 0-1 range for float
        img_out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        # Scale 0-65535 to 0-255
        img_out = (image // 257).astype(np.uint8)
    else:
        img_out = image.astype(np.uint8)
    
    # Handle color space conversion (OpenCV writes BGR)
    if len(img_out.shape) == 2:
        # Grayscale - no conversion needed
        pass
    else:
        if color_space.lower() == "rgb":
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        elif color_space.lower() == "bgr":
            pass  # Already BGR
        elif color_space.lower() == "gray":
            if len(img_out.shape) == 3:
                img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)
        else:
            print(f"Warning: Unknown color_space '{color_space}', assuming BGR")
    
    # Set encoding parameters
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        # PNG compression level (0-9)
        compression = min(9, max(0, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        params = []
    
    # Write image
    try:
        success = cv2.imwrite(str(path), img_out, params)
        if success:
            print(f"Info: Image written to {path}")
        else:
            print(f"Error: Failed to write image to {path}")
        return success
    except Exception as e:
        print(f"Error: Exception writing image to {path}: {e}")
        return False
