"""
Pytest configuration and shared fixtures
=========================================

This file contains pytest configuration and shared fixtures used across
all test modules.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_rgb_image():
    """
    Create a simple RGB test image (100x100x3) with float64 values in [0, 1].
    
    Returns:
        np.ndarray: RGB image array.
    """
    # Create a gradient image for testing
    img = np.zeros((100, 100, 3), dtype=np.float64)
    for i in range(100):
        for j in range(100):
            img[i, j, 0] = i / 100.0  # Red channel gradient
            img[i, j, 1] = j / 100.0  # Green channel gradient
            img[i, j, 2] = 0.5        # Blue channel constant
    return img


@pytest.fixture
def sample_white_image():
    """
    Create a white reference image for flat-field correction testing.
    
    Returns:
        np.ndarray: White BGR image (uint8).
    """
    # Create a white image with slight intensity variation
    img = np.ones((100, 100, 3), dtype=np.uint8) * 250
    # Add some intensity variation
    for i in range(100):
        for j in range(100):
            variation = int(20 * np.sin(i / 10) * np.cos(j / 10))
            img[i, j] = np.clip(250 + variation, 200, 255)
    return img


@pytest.fixture
def reference_illuminant():
    """
    D65 reference illuminant (CIE 1931 2° Standard Observer).
    
    Returns:
        np.ndarray: Reference illuminant xy chromaticity coordinates.
    """
    # D65 standard illuminant
    return np.array([0.31271, 0.32902])


@pytest.fixture
def reference_color_patches():
    """
    Reference RGB values for a standard ColorChecker (24 patches).
    
    Returns:
        np.ndarray: 24x3 array of RGB values (float64, 0-1 range).
    """
    # Simplified ColorChecker reference values (normalized to 0-1)
    # These are approximate values for testing purposes
    patches = np.array([
        [0.443, 0.318, 0.239],  # Dark Skin
        [0.774, 0.569, 0.490],  # Light Skin
        [0.349, 0.431, 0.596],  # Blue Sky
        [0.337, 0.422, 0.267],  # Foliage
        [0.506, 0.506, 0.667],  # Blue Flower
        [0.486, 0.761, 0.643],  # Bluish Green
        [0.886, 0.486, 0.204],  # Orange
        [0.325, 0.345, 0.639],  # Purplish Blue
        [0.757, 0.349, 0.420],  # Moderate Red
        [0.329, 0.227, 0.373],  # Purple
        [0.620, 0.761, 0.318],  # Yellow Green
        [0.933, 0.667, 0.224],  # Orange Yellow
        [0.224, 0.267, 0.588],  # Blue
        [0.349, 0.608, 0.337],  # Green
        [0.749, 0.263, 0.263],  # Red
        [0.933, 0.796, 0.196],  # Yellow
        [0.757, 0.349, 0.553],  # Magenta
        [0.224, 0.490, 0.608],  # Cyan
        [0.957, 0.957, 0.957],  # White
        [0.784, 0.784, 0.784],  # Neutral 8
        [0.627, 0.627, 0.627],  # Neutral 6.5
        [0.467, 0.467, 0.467],  # Neutral 5
        [0.314, 0.314, 0.314],  # Neutral 3.5
        [0.196, 0.196, 0.196],  # Black
    ], dtype=np.float64)
    return patches


@pytest.fixture
def temp_directory(tmp_path):
    """
    Create a temporary directory for test outputs.
    
    Args:
        tmp_path: pytest's tmp_path fixture.
    
    Returns:
        Path: Temporary directory path.
    """
    return tmp_path


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")
