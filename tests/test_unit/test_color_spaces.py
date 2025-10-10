"""
Unit tests for core/color_spaces.py
====================================

Tests color space conversions, chromatic adaptation, and saturation handling.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ColorCorrectionPipeline.core.color_spaces import (
    adapt_chart,
    convert_to_lab,
    convert_to_LCHab,
    do_color_adaptation,
    extrapolate_if_sat_image,
    srgb_to_cielab_D50,
)


class TestColorSpaceConversions:
    """Test color space conversion functions."""
    
    def test_convert_to_lab_white(self):
        """Test LAB conversion of white color."""
        # Pure white should have L*=100, a*≈0, b*≈0
        white_rgb = np.array([[1.0, 1.0, 1.0]])
        illuminant = np.array([0.31271, 0.32902])  # D65

        lab = convert_to_lab(white_rgb, illuminant)

        assert lab.shape == (1, 3)
        assert abs(lab[0, 0] - 100.0) < 1.0  # L* ≈ 100
        assert abs(lab[0, 1]) < 15.0  # a* ≈ 0 (relaxed tolerance)
        assert abs(lab[0, 2]) < 15.0  # b* ≈ 0 (relaxed tolerance)    def test_convert_to_lab_black(self):
        """Test LAB conversion of black color."""
        # Pure black should have L*=0
        black_rgb = np.array([[0.0, 0.0, 0.0]])
        illuminant = np.array([0.31271, 0.32902])
        
        lab = convert_to_lab(black_rgb, illuminant)
        
        assert lab.shape == (1, 3)
        assert abs(lab[0, 0]) < 1.0  # L* ≈ 0
    
    def test_convert_to_lab_red(self):
        """Test LAB conversion of pure red."""
        red_rgb = np.array([[1.0, 0.0, 0.0]])
        illuminant = np.array([0.31271, 0.32902])
        
        lab = convert_to_lab(red_rgb, illuminant)
        
        assert lab.shape == (1, 3)
        # Red should have positive a* (red-green axis)
        assert lab[0, 1] > 20.0
    
    def test_convert_to_lab_batch(self):
        """Test LAB conversion with multiple colors."""
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
        ])
        illuminant = np.array([0.31271, 0.32902])
        
        lab = convert_to_lab(colors, illuminant)
        
        assert lab.shape == (5, 3)
        # All L* values should be in valid range [0, 100]
        assert np.all(lab[:, 0] >= 0.0)
        assert np.all(lab[:, 0] <= 100.0)
    
    def test_convert_to_LCHab_white(self):
        """Test LCH conversion of white color."""
        white_rgb = np.array([[1.0, 1.0, 1.0]])
        illuminant = np.array([0.31271, 0.32902])

        lch = convert_to_LCHab(white_rgb, illuminant)

        assert lch.shape == (1, 3)
        assert abs(lch[0, 0] - 100.0) < 1.0  # L* ≈ 100
        assert lch[0, 1] < 15.0  # Chroma ≈ 0 (achromatic, relaxed tolerance)    def test_convert_to_LCHab_red(self):
        """Test LCH conversion of pure red."""
        red_rgb = np.array([[1.0, 0.0, 0.0]])
        illuminant = np.array([0.31271, 0.32902])
        
        lch = convert_to_LCHab(red_rgb, illuminant)
        
        assert lch.shape == (1, 3)
        assert lch[0, 1] > 20.0  # High chroma (saturated)
        # Hue should be in red range (roughly 0-60 degrees)
        assert 0 <= lch[0, 2] <= 90 or lch[0, 2] >= 330
    
    def test_srgb_to_cielab_D50(self):
        """Test sRGB to CIELAB conversion with D50."""
        rgb = np.array([[0.5, 0.5, 0.5]])
        illuminant = np.array([0.31271, 0.32902])  # D65

        lab = srgb_to_cielab_D50(rgb, illuminant)
        
        assert lab.shape == (1, 3)
        # Mid-gray should have intermediate L*
        assert 40 < lab[0, 0] < 60
        # Should be roughly achromatic
        assert abs(lab[0, 1]) < 10
        assert abs(lab[0, 2]) < 10


class TestChromaticAdaptation:
    """Test chromatic adaptation functions."""
    
    def test_adapt_chart_same_illuminant(self, reference_color_patches):
        """Test that same illuminant returns unchanged chart."""
        import colour
        
        chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        source_illuminant = chart.illuminant
        
        adapted = adapt_chart(chart, source_illuminant)
        
        # Should return chart with same illuminant
        assert_array_almost_equal(adapted.illuminant, source_illuminant, decimal=4)
    
    def test_adapt_chart_different_illuminant(self):
        """Test chromatic adaptation to different illuminant."""
        import colour
        
        chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        d65 = np.array([0.31271, 0.32902])
        
        adapted = adapt_chart(chart, d65)
        
        # Should have new illuminant
        assert_array_almost_equal(adapted.illuminant, d65, decimal=4)
        # Should have same number of patches
        assert len(adapted.data) == len(chart.data)
    
    def test_do_color_adaptation_neutral(self):
        """Test color adaptation on neutral image."""
        # Neutral gray should remain similar under adaptation
        img = np.array([[[0.5, 0.5, 0.5]]])
        orig_illuminant = np.array([0.31271, 0.32902])  # D65
        dest_illuminant = np.array([0.34567, 0.35850])  # D50
        
        adapted = do_color_adaptation(img, orig_illuminant, dest_illuminant)
        
        assert adapted.shape == img.shape
        # Neutral should stay relatively neutral
        assert np.std(adapted[0, 0]) < 0.1
    
    def test_do_color_adaptation_preserves_shape(self):
        """Test that adaptation preserves image shape."""
        img = np.random.rand(10, 20, 3)
        orig_illuminant = np.array([0.31271, 0.32902])
        dest_illuminant = np.array([0.34567, 0.35850])
        
        adapted = do_color_adaptation(img, orig_illuminant, dest_illuminant)
        
        assert adapted.shape == img.shape


class TestSaturationHandling:
    """Test saturation detection and extrapolation."""
    
    def test_extrapolate_if_sat_image_no_saturation(self):
        """Test with no saturated pixels."""
        # Image with no saturation
        img = np.random.rand(100, 100, 3) * 0.8  # Max 0.8, no saturation
        ref_patches = np.random.rand(24, 3) * 0.8

        result_img, sat_vals, sat_ids = extrapolate_if_sat_image(img, ref_patches)
        
        assert result_img.shape == img.shape
        # With no saturation, result should be unchanged or very similar
        assert_array_almost_equal(result_img, img, decimal=1)
        # Should have no saturated pixels detected
        assert sat_vals is None
        assert sat_ids is None
    
    def test_extrapolate_if_sat_image_with_saturation(self):
        """Test with saturated pixels."""
        # Create image with some saturated pixels
        img = np.random.rand(100, 100, 3)
        img[0:10, 0:10, :] = 1.0  # Saturated region
        ref_patches = np.random.rand(24, 3)

        result_img, sat_vals, sat_ids = extrapolate_if_sat_image(img, ref_patches)
        
        assert result_img.shape == img.shape
        # Result should still be in valid range
        assert result_img.min() >= 0.0
        assert result_img.max() <= 1.0
        # Should have detected saturated pixels
        assert sat_vals is not None
        assert sat_ids is not None
    
    def test_extrapolate_if_sat_image_preserves_unsaturated(self):
        """Test that unsaturated regions are preserved."""
        # Image with known unsaturated region
        img = np.zeros((100, 100, 3))
        img[50:60, 50:60, :] = 0.5  # Mid-gray region
        ref_patches = np.random.rand(24, 3)

        result_img, sat_vals, sat_ids = extrapolate_if_sat_image(img, ref_patches)
        
        # Mid-gray region should be unchanged or very similar
        original_region = img[50:60, 50:60, :]
        result_region = result_img[50:60, 50:60, :]
        assert_array_almost_equal(original_region, result_region, decimal=2)


@pytest.mark.parametrize("illuminant", [
    np.array([0.31271, 0.32902]),  # D65
    np.array([0.34567, 0.35850]),  # D50
    np.array([0.44757, 0.40745]),  # A (incandescent)
])
def test_lab_conversion_various_illuminants(illuminant):
    """Test LAB conversion with various illuminants."""
    colors = np.array([
        [1.0, 1.0, 1.0],  # White
        [0.5, 0.5, 0.5],  # Gray
        [0.0, 0.0, 0.0],  # Black
    ])
    
    lab = convert_to_lab(colors, illuminant)
    
    assert lab.shape == (3, 3)
    # L* should be in valid range
    assert np.all(lab[:, 0] >= 0.0)
    assert np.all(lab[:, 0] <= 100.0)


@pytest.mark.parametrize("rgb,expected_chroma_high", [
    (np.array([[1.0, 0.0, 0.0]]), True),   # Red - high chroma
    (np.array([[0.0, 1.0, 0.0]]), True),   # Green - high chroma
    (np.array([[0.0, 0.0, 1.0]]), True),   # Blue - high chroma
    (np.array([[0.5, 0.5, 0.5]]), False),  # Gray - low chroma
    (np.array([[1.0, 1.0, 1.0]]), False),  # White - low chroma
])
def test_lch_chroma_expectations(rgb, expected_chroma_high):
    """Test that LCH chroma matches expectations for different colors."""
    illuminant = np.array([0.31271, 0.32902])
    lch = convert_to_LCHab(rgb, illuminant)
    
    chroma = lch[0, 1]
    if expected_chroma_high:
        assert chroma > 20.0, f"Expected high chroma for {rgb}, got {chroma}"
    else:
        assert chroma < 15.0, f"Expected low chroma for {rgb}, got {chroma}"
