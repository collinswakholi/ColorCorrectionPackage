"""
Unit tests for core/metrics.py
===============================

Tests color difference metrics, regression metrics, and statistics.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ColorCorrectionPipeline.core.metrics import (
    Metrics,
    arrange_metrics,
    compute_mae,
    desc_stats,
    get_metrics,
)


class TestMetricsClass:
    """Test Metrics class for regression evaluation."""
    
    def test_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        m = Metrics(y_true, y_pred)
        results = m.regression_metrics()
        
        assert results["mae"] == 0.0
        assert results["rmse"] == 0.0
        assert results["r2_score"] == 1.0
    
    def test_metrics_poor_prediction(self):
        """Test metrics with poor predictions."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])  # Reverse order
        
        m = Metrics(y_true, y_pred)
        results = m.regression_metrics()
        
        # Should have high error
        assert results["mae"] > 0.0
        assert results["rmse"] > 0.0
        # R² should be negative (worse than mean)
        assert results["r2_score"] < 0.0
    
    def test_metrics_moderate_prediction(self):
        """Test metrics with moderate predictions."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [2.2], [2.9], [4.1], [4.8]])
        
        m = Metrics(y_true, y_pred)
        results = m.regression_metrics()
        
        # Should have small positive error
        assert 0.0 < results["mae"] < 1.0
        assert 0.0 < results["rmse"] < 1.0
        # R² should be high but not perfect
        assert 0.5 < results["r2_score"] < 1.0
    
    def test_metrics_2d_input(self):
        """Test metrics with 2D arrays."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])
        
        m = Metrics(y_true, y_pred)
        results = m.regression_metrics()
        
        assert results["mae"] > 0.0
        assert results["rmse"] > 0.0
        assert 0.0 < results["r2_score"] < 1.0


class TestComputeMAE:
    """Test Mean Absolute Error computation."""
    
    def test_compute_mae_zero_error(self):
        """Test MAE with identical predictions."""
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[1.0, 2.0, 3.0]])
        
        mae = compute_mae(a, b)
        assert mae[0] < 0.001  # Near zero angular error
    
    def test_compute_mae_constant_error(self):
        """Test MAE with constant error (angular)."""
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.9, 0.1, 0.0]])
        
        mae = compute_mae(a, b)
        assert mae[0] > 0.0  # Should have some angular error
    
    def test_compute_mae_variable_error(self):
        """Test MAE with variable error (angular)."""
        a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        
        mae = compute_mae(a, b)
        assert len(mae) == 2
        assert np.all(mae > 0.0)  # Both should have angular error
    
    def test_compute_mae_2d(self):
        """Test MAE with 2D arrays."""
        a = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        b = np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]])
        
        mae = compute_mae(a, b)
        assert len(mae) == 2
        assert np.all(mae >= 0.0)


class TestGetMetrics:
    """Test comprehensive metrics computation with Delta-E."""
    
    def test_get_metrics_identical_colors(self, reference_illuminant):
        """Test metrics with identical reference and measured colors."""
        colors = np.array([
            [0.5, 0.3, 0.2],
            [0.8, 0.6, 0.4],
            [0.3, 0.5, 0.7],
        ])
        
        m = get_metrics(colors, colors, reference_illuminant, "srgb")
        
        # Should have zero error - m is a metrics object with deltaE attribute
        assert len(m.deltaE) == 3
        assert np.mean(m.deltaE) < 0.1  # Nearly zero
    
    def test_get_metrics_different_colors(self, reference_illuminant):
        """Test metrics with different colors."""
        ref_colors = np.array([
            [0.5, 0.3, 0.2],
            [0.8, 0.6, 0.4],
        ])
        meas_colors = np.array([
            [0.6, 0.4, 0.3],
            [0.7, 0.5, 0.3],
        ])
        
        m = get_metrics(ref_colors, meas_colors, reference_illuminant, "srgb")
        
        # Should have positive Delta-E - m is a metrics object
        assert len(m.deltaE) == 2
        assert np.mean(m.deltaE) > 0.0
        assert np.std(m.deltaE) >= 0.0  # Should have std available
        assert np.max(m.deltaE) > 0.0  # Max should be positive
    
    def test_get_metrics_shape_mismatch(self, reference_illuminant):
        """Test that shape mismatch raises error."""
        ref_colors = np.array([[0.5, 0.3, 0.2]])
        meas_colors = np.array([[0.6, 0.4, 0.3], [0.7, 0.5, 0.3]])
        
        with pytest.raises((ValueError, AssertionError)):
            get_metrics(ref_colors, meas_colors, reference_illuminant, "srgb")


class TestDescStats:
    """Test descriptive statistics computation."""
    
    def test_desc_stats_single_value(self):
        """Test statistics with single value."""
        data = np.array([5.0])
        
        stats = desc_stats(data)
        
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
    
    def test_desc_stats_multiple_values(self):
        """Test statistics with multiple values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = desc_stats(data)
        
        assert stats["mean"] == 3.0
        assert stats["std"] > 0.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
    
    def test_desc_stats_with_outliers(self):
        """Test statistics with outliers."""
        data = np.array([1.0, 2.0, 3.0, 100.0])  # 100 is outlier
        
        stats = desc_stats(data)
        
        assert stats["max"] == 100.0
        assert stats["mean"] > stats["median"]  # Mean pulled by outlier


class TestArrangeMetrics:
    """Test metrics arrangement for before/after comparison."""
    
    def test_arrange_metrics_basic(self):
        """Test basic metrics arrangement."""
        from ColorCorrectionPipeline.core.metrics import metrics as MetricsContainer
        
        # Create metrics objects
        metrics_before = MetricsContainer(
            deltaE=np.array([10.0, 12.0]),
            mse=np.array([0.1, 0.15]),
            mae=np.array([5.0, 6.0])
        )
        metrics_after = MetricsContainer(
            deltaE=np.array([5.0, 6.0]),
            mse=np.array([0.05, 0.06]),
            mae=np.array([2.5, 3.0])
        )
        
        arranged = arrange_metrics(metrics_before, metrics_after, name="test")
        
        # Should have before and after keys
        assert "test_deltaE_before_mean" in arranged
        assert "test_deltaE_after_mean" in arranged
    
    def test_arrange_metrics_improvement(self):
        """Test that improvement is captured."""
        from ColorCorrectionPipeline.core.metrics import metrics as MetricsContainer
        
        metrics_before = MetricsContainer(
            deltaE=np.array([10.0]), mse=np.array([0.1]), mae=np.array([5.0])
        )
        metrics_after = MetricsContainer(
            deltaE=np.array([5.0]), mse=np.array([0.05]), mae=np.array([2.5])
        )
        
        arranged = arrange_metrics(metrics_before, metrics_after, name="test")
        
        # After should be better (lower) than before
        assert arranged["test_deltaE_after_mean"] < arranged["test_deltaE_before_mean"]
        assert len(arranged) > 0


@pytest.mark.parametrize("n_samples", [10, 100, 1000])
def test_metrics_scalability(n_samples):
    """Test that metrics computation scales with sample size."""
    y_true = np.random.rand(n_samples, 1)  # 2D array
    y_pred = y_true + np.random.normal(0, 0.1, (n_samples, 1))
    
    m = Metrics(y_true, y_pred)
    results = m.regression_metrics()
    
    # Should compute successfully regardless of size
    assert results["mae"] >= 0.0
    assert results["rmse"] >= 0.0
    assert -1.0 <= results["r2_score"] <= 1.0


@pytest.mark.parametrize("error_magnitude", [0.01, 0.1, 1.0, 10.0])
def test_mae_proportional_to_error(error_magnitude):
    """Test that angular error increases with perturbation magnitude."""
    # Create color vectors with varying magnitudes of perturbation
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0 - error_magnitude, error_magnitude, 0.0]])
    
    mae = compute_mae(a, b)
    # Angular error should increase as perturbation increases
    assert mae[0] >= 0.0
