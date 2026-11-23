"""
Image reading utilities
=======================

This module provides utilities for reading images with proper color space
handling and format conversion.

Functions:
    - read_image: Read single image from file
    - read_batch: Read multiple images in batch
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union, cast

import cv2
import numpy as np

from ..core.utils import to_float64  # noqa: F401  (historical re-export)

__all__ = ["read_image", "read_batch"]

logger = logging.getLogger(__name__)

# Full-scale value of each float output range accepted by the legacy reader.
_RANGE_MAXIMA = {"0-1": 1.0, "0-255": 255.0, "0-65535": 65535.0}


def read_image(
    path: Union[str, Path],
    color_space: str = "rgb",
    dtype: str = "float64",
    range_: str = "0-1",
    *,
    strict: bool = False,
    image_options=None,
) -> np.ndarray:
    """
    Read image from file with proper color space handling.

    Args:
        path: Path to image file
        color_space: Desired color space ('rgb', 'bgr', 'gray')
        dtype: Output dtype ('float32', 'float64', 'uint8', 'uint16')
        range_: Output range ('0-1' for float, '0-255' for uint8, '0-65535' for uint16).
            Float output from 8/16-bit files is scaled to this range from the file's
            full scale (e.g. a uint16 file read with ``range_='0-255'`` is divided by
            257). Any other string returns the stored code values unscaled.

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
    if type(strict) is not bool:
        raise TypeError("strict must be boolean")
    if strict:
        from .image_data import read_image_data

        if color_space.lower() not in ("rgb", "bgr"):
            raise ValueError(
                "strict reader returns three-channel RGB/BGR; choose grayscale preprocessing explicitly"
            )
        if dtype.lower() not in ("float32", "float64", "uint8", "uint16"):
            raise ValueError("unsupported strict output dtype")
        expected_range = (
            "0-1"
            if dtype.lower().startswith("float")
            else "0-255" if dtype.lower() == "uint8" else "0-65535"
        )
        if range_ != expected_range:
            raise ValueError("strict reader range_ must match the selected output dtype")
        data = read_image_data(path, **(image_options or {}))
        pixels = data.pixels
        if color_space.lower() == "bgr":
            pixels = pixels[..., ::-1]
        if dtype.lower() in ("uint8", "uint16"):
            return cast(
                np.ndarray, np.rint(pixels * np.iinfo(dtype.lower()).max).astype(dtype.lower())
            )
        return cast(np.ndarray, pixels.astype(dtype.lower()))
    if image_options is not None:
        raise ValueError("image_options requires strict=True")
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

    # Handle dtype conversion. Integer sources are scaled from their own full scale
    # to the requested ``range_``; integer outputs round to the nearest code like
    # the strict reader (``np.rint``), instead of truncating.
    if dtype.lower() in ("float32", "float64"):
        float_type = np.float32 if dtype.lower() == "float32" else np.float64
        target = _RANGE_MAXIMA.get(range_)
        if img.dtype in (np.uint8, np.uint16) and target is not None:
            source = 255.0 if img.dtype == np.uint8 else 65535.0
            scaled: np.ndarray = img.astype(float_type)
            if source > target:
                scaled /= source / target  # 255, 65535 or 257: exact divisors
            elif source < target:
                scaled *= target / source  # 257
            img = scaled
        else:
            img = img.astype(float_type)
    elif dtype.lower() == "uint8":
        if img.dtype.kind == "f":
            if range_ == "0-1":
                img = np.rint(np.clip(img.astype(np.float64), 0, 1) * 255).astype(np.uint8)
            else:
                img = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = np.rint(img / 257.0).astype(np.uint8)  # Scale 0-65535 to 0-255
        else:
            img = img.astype(np.uint8)
    elif dtype.lower() == "uint16":
        if img.dtype.kind == "f":
            if range_ == "0-1":
                img = np.rint(np.clip(img.astype(np.float64), 0, 1) * 65535).astype(np.uint16)
            else:
                img = np.rint(np.clip(img, 0, 65535)).astype(np.uint16)
        elif img.dtype == np.uint8:
            img = img.astype(np.uint16) * 257  # Scale 0-255 to 0-65535
        else:
            img = img.astype(np.uint16)
    else:
        raise ValueError(f"Invalid dtype: {dtype}. Use 'float32', 'float64', 'uint8', or 'uint16'.")

    return img


def read_batch(
    paths: List[Union[str, Path]],
    color_space: str = "rgb",
    dtype: str = "float64",
    range_: str = "0-1",
    verbose: bool = True,
    *,
    strict: bool = False,
    image_options=None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read multiple images in batch.

    Args:
        paths: List of image file paths
        color_space: Desired color space ('rgb', 'bgr', 'gray')
        dtype: Output dtype ('float32', 'float64', 'uint8', 'uint16')
        range_: Output range ('0-1' for float, '0-255' for uint8)
        verbose: Whether to log progress (``ColorCorrectionPipeline.io.readers`` logger)

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
            img = read_image(
                path,
                color_space=color_space,
                dtype=dtype,
                range_=range_,
                strict=strict,
                image_options=image_options,
            )
            images.append(img)
            if verbose:
                logger.info("Read image %d/%d: %s", i + 1, len(paths), path)
        except Exception as e:
            failed_paths.append(str(path))
            if verbose:
                logger.warning("Failed to read image %d/%d: %s - %s", i + 1, len(paths), path, e)

    if verbose:
        logger.info("Successfully read %d/%d images", len(images), len(paths))
        if failed_paths:
            logger.warning("Failed to read %d images", len(failed_paths))

    return images, failed_paths
