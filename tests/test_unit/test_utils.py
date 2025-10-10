"""
Unit tests for core/utils.py
=============================

Tests utility functions for image conversion, color chart extraction,
polynomial operations, and memory management.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ColorCorrectionPipeline.core.utils import (
    compute_diag,
    compute_temperature,
    estimate_fit,
    extract_color_chart,
    extract_neutral_patches,
    free_memory,
    get_attr,
    poly_func,
    poly_func_torch,
    to_float64,
    to_uint8,
)


class TestImageConversion:
    """Test image format conversion functions."""
    
    def test_to_float64_uint8_input(self):
        """Test conversion from uint8 to float64."""
        img_uint8 = np.array([0, 127, 255], dtype=np.uint8)
        result = to_float64(img_uint8)
        
        assert result.dtype == np.float64
        assert_array_almost_equal(result, [0.0, 0.498039, 1.0], decimal=5)
    
    def test_to_float64_uint16_input(self):
        """Test conversion from uint16 to float64."""
        img_uint16 = np.array([0, 32767, 65535], dtype=np.uint16)
        result = to_float64(img_uint16)
        
        assert result.dtype == np.float64
        assert_array_almost_equal(result, [0.0, 0.5, 1.0], decimal=5)
    
    def test_to_float64_float_input(self):
        """Test that float input is returned as-is."""
        img_float = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = to_float64(img_float)
        
        assert result.dtype == np.float64
        assert_array_almost_equal(result, [0.0, 0.5, 1.0])
    
    def test_to_float64_3d_image(self):
        """Test conversion with 3D image array."""
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = to_float64(img)
        
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_to_uint8_float_input(self):
        """Test conversion from float64 to uint8."""
        img_float = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = to_uint8(img_float)
        
        assert result.dtype == np.uint8
        assert_array_equal(result, [0, 127, 255])
    
    def test_to_uint8_clipping(self):
        """Test that out-of-range values are clipped."""
        img_float = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
        result = to_uint8(img_float)
        
        assert result.dtype == np.uint8
        assert_array_equal(result, [0, 127, 255])
    
    def test_to_uint8_uint8_input(self):
        """Test that uint8 input is returned as-is."""
        img_uint8 = np.array([0, 127, 255], dtype=np.uint8)
        result = to_uint8(img_uint8)
        
        assert result.dtype == np.uint8
        assert_array_equal(result, img_uint8)
    
    def test_roundtrip_conversion(self):
        """Test roundtrip uint8 -> float64 -> uint8."""
        original = np.array([0, 64, 127, 191, 255], dtype=np.uint8)
        float_version = to_float64(original)
        recovered = to_uint8(float_version)
        
        assert_array_equal(original, recovered)


class TestPolynomialFunctions:
    """Test polynomial evaluation functions."""
    
    def test_poly_func_linear(self):
        """Test linear polynomial (degree 1)."""
        coeffs = np.array([2.0, 3.0])  # 2x + 3
        x = np.array([0.0, 1.0, 2.0])
        expected = np.array([3.0, 5.0, 7.0])
        
        result = poly_func(x, coeffs)
        assert_array_almost_equal(result, expected)
    
    def test_poly_func_quadratic(self):
        """Test quadratic polynomial (degree 2)."""
        coeffs = np.array([1.0, 2.0, 3.0])  # x^2 + 2x + 3
        x = np.array([0.0, 1.0, 2.0])
        expected = np.array([3.0, 6.0, 11.0])
        
        result = poly_func(x, coeffs)
        assert_array_almost_equal(result, expected)
    
    def test_poly_func_vectorized(self):
        """Test polynomial with 2D input."""
        coeffs = np.array([1.0, 0.0, 2.0])  # x^2 + 2
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        expected = np.array([[2.0, 3.0], [6.0, 11.0]])
        
        result = poly_func(x, coeffs)
        assert_array_almost_equal(result, expected)
    
    def test_estimate_fit_linear(self):
        """Test linear curve fitting."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1
        
        coeffs = estimate_fit(x, y, degree=1)
        
        # Coefficients should be approximately [2.0, 1.0]
        assert len(coeffs) == 2
        assert abs(coeffs[0] - 2.0) < 0.1
        assert abs(coeffs[1] - 1.0) < 0.1
    
    def test_estimate_fit_quadratic(self):
        """Test quadratic curve fitting."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 5.0, 10.0])  # y = x^2 + 1
        
        coeffs = estimate_fit(x, y, degree=2)
        
        # Should have 3 coefficients
        assert len(coeffs) == 3
        # Leading coefficient should be close to 1.0
        assert abs(coeffs[0] - 1.0) < 0.1


class TestUtilityFunctions:
    """Test miscellaneous utility functions."""
    
    def test_get_attr_existing(self):
        """Test getting existing attribute."""
        class DummyClass:
            value = 42
        
        obj = DummyClass()
        result = get_attr(obj, "value", default=0)
        assert result == 42
    
    def test_get_attr_missing_with_default(self):
        """Test getting missing attribute returns default."""
        class DummyClass:
            pass
        
        obj = DummyClass()
        result = get_attr(obj, "missing", default=99)
        assert result == 99
    
    def test_get_attr_missing_no_default(self):
        """Test getting missing attribute with no default returns None."""
        class DummyClass:
            pass
        
        obj = DummyClass()
        result = get_attr(obj, "missing")
        assert result is None
    
    def test_compute_diag_simple(self):
        """Test diagonal matrix computation."""
        # Create diagonal matrix from vector
        vec = np.array([1.0, 2.0, 3.0])
        
        diag = compute_diag(vec)
        
        # Should return 3x3 diagonal matrix
        assert diag.shape == (3, 3)
        # Diagonal elements should match input
        assert diag[0, 0] == 1.0
        assert diag[1, 1] == 2.0
        assert diag[2, 2] == 3.0
        # Off-diagonal elements should be zero
        assert diag[0, 1] == 0.0
        assert diag[1, 0] == 0.0
        assert diag[0, 2] == 0.0
    
    def test_free_memory(self):
        """Test memory cleanup function."""
        # Should not raise any exceptions
        try:
            free_memory()
        except Exception as e:
            pytest.fail(f"free_memory() raised {e}")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_conversion_all_dtypes(dtype):
    """Test conversion works with all common image dtypes."""
    if dtype == np.uint8:
        img = np.array([0, 127, 255], dtype=dtype)
    elif dtype == np.uint16:
        img = np.array([0, 32767, 65535], dtype=dtype)
    else:
        img = np.array([0.0, 0.5, 1.0], dtype=dtype)
    
    # Convert to float64
    float_img = to_float64(img)
    assert float_img.dtype == np.float64
    assert float_img.min() >= 0.0
    assert float_img.max() <= 1.0
    
    # Convert back to uint8
    uint8_img = to_uint8(float_img)
    assert uint8_img.dtype == np.uint8
    assert uint8_img.min() >= 0
    assert uint8_img.max() <= 255


@pytest.mark.parametrize("shape", [(100,), (100, 100), (100, 100, 3), (10, 100, 100, 3)])
def test_conversion_various_shapes(shape):
    """Test conversion works with various array shapes."""
    img_uint8 = np.random.randint(0, 256, shape, dtype=np.uint8)
    
    # Forward conversion
    float_img = to_float64(img_uint8)
    assert float_img.shape == shape
    assert float_img.dtype == np.float64
    
    # Backward conversion
    uint8_img = to_uint8(float_img)
    assert uint8_img.shape == shape
    assert uint8_img.dtype == np.uint8
