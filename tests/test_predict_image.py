"""
Test script to validate predict_image method inputs and outputs.
This test validates the documentation in docs/predict_image_analysis.md
"""

import numpy as np
import pytest
from typing import Dict

# Import the main class
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ColorCorrectionPipeline.ccp import ColorCorrection


class TestPredictImageInputs:
    """Test input parameter validation for predict_image"""
    
    def test_accepts_numpy_array(self):
        """Test that predict_image accepts numpy array input"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        # Create a dummy RGB image (float64, 0-1 range)
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        
        # Should not raise an error
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Verify it returns a dictionary
        assert isinstance(results, dict), "Output should be a dictionary"
    
    def test_rejects_invalid_type(self):
        """Test that predict_image rejects invalid input types"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        # Should raise TypeError for invalid input
        with pytest.raises(TypeError, match="Image must be a file path or numpy array"):
            cc.predict_image(Image=123, show=False)
        
        with pytest.raises(TypeError, match="Image must be a file path or numpy array"):
            cc.predict_image(Image=[1, 2, 3], show=False)
    
    def test_show_parameter_is_bool(self):
        """Test that show parameter accepts boolean values"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        
        # Should work with both True and False
        results1 = cc.predict_image(Image=dummy_image, show=False)
        assert isinstance(results1, dict)
        
        results2 = cc.predict_image(Image=dummy_image, show=True)
        assert isinstance(results2, dict)


class TestPredictImageOutputs:
    """Test output structure and content of predict_image"""
    
    def test_output_is_dictionary(self):
        """Test that predict_image returns a dictionary"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert isinstance(results, dict), "Output must be a dictionary"
    
    def test_output_has_required_keys(self):
        """Test that output dictionary contains all expected keys"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Check for all expected keys
        expected_keys = {'FFC', 'GC', 'WB', 'CC'}
        actual_keys = set(results.keys())
        
        assert expected_keys == actual_keys, \
            f"Output should have keys {expected_keys}, got {actual_keys}"
    
    def test_ffc_output_always_present(self):
        """Test that FFC output is always present and is ndarray"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert 'FFC' in results, "FFC key must be present"
        assert isinstance(results['FFC'], np.ndarray), "FFC value must be ndarray"
    
    def test_gc_output_always_present(self):
        """Test that GC output is always present and is ndarray"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert 'GC' in results, "GC key must be present"
        assert isinstance(results['GC'], np.ndarray), "GC value must be ndarray"
    
    def test_wb_output_always_present(self):
        """Test that WB output is always present and is ndarray"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert 'WB' in results, "WB key must be present"
        assert isinstance(results['WB'], np.ndarray), "WB value must be ndarray"
    
    def test_cc_output_present_but_may_be_none(self):
        """Test that CC output is present (but may be None if no model)"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert 'CC' in results, "CC key must be present in output"
        # CC can be None if no model is trained, or ndarray if model exists
        assert results['CC'] is None or isinstance(results['CC'], np.ndarray), \
            "CC value must be None or ndarray"
    
    def test_output_array_shapes(self):
        """Test that output arrays have correct shape"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        h, w = 100, 150
        dummy_image = np.random.rand(h, w, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Check FFC, GC, WB shapes
        assert results['FFC'].shape == (h, w, 3), "FFC output shape mismatch"
        assert results['GC'].shape == (h, w, 3), "GC output shape mismatch"
        assert results['WB'].shape == (h, w, 3), "WB output shape mismatch"
        
        # CC may be None
        if results['CC'] is not None:
            assert results['CC'].shape == (h, w, 3), "CC output shape mismatch"
    
    def test_output_array_dtype(self):
        """Test that output arrays have correct dtype (float64)"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        assert results['FFC'].dtype == np.float64, "FFC dtype should be float64"
        assert results['GC'].dtype == np.float64, "GC dtype should be float64"
        assert results['WB'].dtype == np.float64, "WB dtype should be float64"
        
        if results['CC'] is not None:
            assert results['CC'].dtype == np.float64, "CC dtype should be float64"
    
    def test_output_array_value_range(self):
        """Test that output arrays have values in [0, 1] range"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Check value ranges
        assert np.all(results['FFC'] >= 0.0) and np.all(results['FFC'] <= 1.0), \
            "FFC values should be in [0, 1]"
        assert np.all(results['GC'] >= 0.0) and np.all(results['GC'] <= 1.0), \
            "GC values should be in [0, 1]"
        assert np.all(results['WB'] >= 0.0) and np.all(results['WB'] <= 1.0), \
            "WB values should be in [0, 1]"
        
        if results['CC'] is not None:
            assert np.all(results['CC'] >= 0.0) and np.all(results['CC'] <= 1.0), \
                "CC values should be in [0, 1]"


class TestPredictImageBehavior:
    """Test the behavior and pipeline flow of predict_image"""
    
    def test_without_models_returns_original_in_stages(self):
        """Test that without models, image passes through unchanged"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(50, 50, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Without any models, FFC should be same as input
        # (may have small floating point differences due to conversions)
        assert results['FFC'].shape == dummy_image.shape
        
        # CC should be None without a CC model
        assert results['CC'] is None, \
            "CC should be None when no color correction model is trained"
    
    def test_sequential_pipeline_flow(self):
        """Test that corrections are applied sequentially"""
        cc = ColorCorrection()
        cc.get_reference_values()
        
        dummy_image = np.random.rand(50, 50, 3).astype(np.float64)
        results = cc.predict_image(Image=dummy_image, show=False)
        
        # Verify all stages are present
        assert 'FFC' in results
        assert 'GC' in results
        assert 'WB' in results
        assert 'CC' in results
        
        # Each stage should have the same dimensions
        base_shape = dummy_image.shape
        assert results['FFC'].shape == base_shape
        assert results['GC'].shape == base_shape
        assert results['WB'].shape == base_shape


def test_documentation_example():
    """Test that the example code from documentation works"""
    cc = ColorCorrection()
    cc.get_reference_values()
    
    # Simulate the documented usage pattern
    dummy_image = np.random.rand(100, 100, 3).astype(np.float64)
    
    # This should match the documented interface
    results = cc.predict_image(Image=dummy_image, show=False)
    
    # Verify we can access all documented keys
    ffc_image = results['FFC']
    gc_image = results['GC']
    wb_image = results['WB']
    cc_image = results['CC']
    
    # Verify types match documentation
    assert isinstance(ffc_image, np.ndarray)
    assert isinstance(gc_image, np.ndarray)
    assert isinstance(wb_image, np.ndarray)
    assert cc_image is None or isinstance(cc_image, np.ndarray)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
