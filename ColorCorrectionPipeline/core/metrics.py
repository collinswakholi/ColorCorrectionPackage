"""
Color difference metrics and statistics
========================================

This module provides functions for computing color difference metrics:
    - Delta-E (CIE color difference)
    - Mean Squared Error (MSE)
    - Mean Angular Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - Descriptive statistics

Preserves backwards compatibility with the original package.
"""

from typing import Any, Dict, Optional

import colour
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as sklearn_metrics

from ..constants import EPSILON
from .color_spaces import convert_to_lab

__all__ = [
    "Metrics",
    "desc_stats",
    "get_metrics",
    "arrange_metrics",
    "compute_mae",
]


class metrics:
    """
    Container for color difference metrics.
    
    Stores Delta-E, MSE, and MAE metrics for a set of color comparisons.
    
    Attributes:
        deltaE: Delta-E color difference values
        mse: Mean squared error values
        mae: Mean angular error values
        
    Example:
        >>> deltaE = np.array([2.3, 3.1, 1.8])
        >>> mse = np.array([0.01, 0.02, 0.015])
        >>> mae = np.array([1.5, 2.0, 1.2])
        >>> m = metrics(deltaE, mse, mae)
        >>> m.deltaE.mean()
        2.4
    """
    
    def __init__(
        self,
        deltaE: np.ndarray,
        mse: np.ndarray,
        mae: np.ndarray,
    ):
        """
        Initialize metrics container.
        
        Args:
            deltaE: Delta-E values
            mse: MSE values
            mae: MAE values
        """
        self.deltaE = np.asarray(deltaE)
        self.mse = np.asarray(mse)
        self.mae = np.asarray(mae)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert metrics to dictionary."""
        return {
            "deltaE": self.deltaE,
            "mse": self.mse,
            "mae": self.mae,
        }
    
    def mean(self) -> Dict[str, float]:
        """Compute mean of each metric."""
        return {
            "deltaE_mean": float(np.mean(self.deltaE)),
            "mse_mean": float(np.mean(self.mse)),
            "mae_mean": float(np.mean(self.mae)),
        }
    
    def median(self) -> Dict[str, float]:
        """Compute median of each metric."""
        return {
            "deltaE_median": float(np.median(self.deltaE)),
            "mse_median": float(np.median(self.mse)),
            "mae_median": float(np.median(self.mae)),
        }
    
    def std(self) -> Dict[str, float]:
        """Compute standard deviation of each metric."""
        return {
            "deltaE_std": float(np.std(self.deltaE)),
            "mse_std": float(np.std(self.mse)),
            "mae_std": float(np.std(self.mae)),
        }


class Metrics:
    """
    Advanced metrics computation class for regression and color analysis.
    
    Computes comprehensive metrics including:
        - Regression metrics (RMSE, MAE, MAPE, R², etc.)
        - Image quality metrics (PSNR, SSIM)
        - Descriptive statistics
        
    Attributes:
        gt: Ground truth values
        pred: Predicted values
        
    Example:
        >>> gt = np.array([[1.0, 0.5, 0.3]])
        >>> pred = np.array([[0.95, 0.52, 0.28]])
        >>> m = Metrics(gt, pred)
        >>> results = m.regression_metrics()
        >>> "rmse" in results
        True
    """
    
    def __init__(self, gt: np.ndarray, pred: np.ndarray):
        """
        Initialize Metrics instance.
        
        Args:
            gt: Ground truth values
            pred: Predicted values
            
        Raises:
            ValueError: If gt and pred shapes don't match
        """
        self.gt = np.asarray(gt)
        self.pred = np.asarray(pred)
        
        if self.gt.shape != self.pred.shape:
            raise ValueError(
                f"Ground truth and predictions must have same shape. "
                f"Got gt: {self.gt.shape}, pred: {self.pred.shape}"
            )
    
    @staticmethod
    def compute_mae(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        """
        Compute mean angular error between two matrices.
        
        Args:
            mat1: First matrix (N, C)
            mat2: Second matrix (N, C)
            
        Returns:
            np.ndarray: Angular errors in degrees
            
        Example:
            >>> mat1 = np.array([[1.0, 0.0, 0.0]])
            >>> mat2 = np.array([[0.9, 0.1, 0.0]])
            >>> mae = Metrics.compute_mae(mat1, mat2)
            >>> mae.shape
            (1,)
        """
        mat1 = np.asarray(mat1)
        mat2 = np.asarray(mat2)
        
        assert mat1.shape == mat2.shape, "Matrices must have same shape"
        
        # Flatten to (N, C)
        mat1_flat = mat1.reshape(-1, mat1.shape[-1])
        mat2_flat = mat2.reshape(-1, mat2.shape[-1])
        
        # Compute norms
        norms = np.linalg.norm(mat1_flat, axis=1) * np.linalg.norm(mat2_flat, axis=1)
        
        # Compute dot products
        dot_prods = np.einsum("ij,ij->i", mat1_flat, mat2_flat)
        
        # Compute angular error
        dot_product = np.clip(dot_prods / (norms + EPSILON), -1, 1)
        angular_error = np.degrees(np.arccos(dot_product))
        
        return angular_error
    
    def regression_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive regression metrics.
        
        Returns:
            dict: Metrics including:
                - rmse: Root mean squared error
                - mae: Mean absolute error
                - mape: Mean absolute percentage error
                - r2_score: R² coefficient of determination
                - max_error: Maximum error
                - mean_angular_error: Mean angular error in degrees
                - pairwise_euclidean: Pairwise Euclidean distances
                
        Example:
            >>> gt = np.random.rand(10, 3)
            >>> pred = gt + np.random.randn(10, 3) * 0.01
            >>> m = Metrics(gt, pred)
            >>> results = m.regression_metrics()
            >>> results["r2_score"] > 0.9
            True
        """
        metrics_dict = {}
        
        gt_flat = self.gt.flatten()
        pred_flat = self.pred.flatten()
        m_out = "uniform_average"
        
        # Core metrics
        try:
            rmse = sklearn_metrics.root_mean_squared_error(
                self.gt, self.pred, multioutput=m_out
            )
        except AttributeError:
            # Fallback for older sklearn versions
            rmse = np.sqrt(sklearn_metrics.mean_squared_error(
                self.gt, self.pred, multioutput=m_out
            ))
        
        mae = sklearn_metrics.mean_absolute_error(
            self.gt, self.pred, multioutput=m_out
        )
        
        try:
            mape = sklearn_metrics.mean_absolute_percentage_error(
                self.gt, self.pred, multioutput=m_out
            )
        except Exception:
            # Fallback if MAPE fails (e.g., division by zero)
            mape = np.nan
        
        r2_score = sklearn_metrics.r2_score(
            self.gt, self.pred, multioutput=m_out
        )
        
        max_error = sklearn_metrics.max_error(gt_flat, pred_flat)
        
        # Pairwise distances
        p_dist = sklearn_metrics.pairwise_distances(
            self.gt, self.pred, metric="euclidean", n_jobs=1
        )
        
        # Gamma deviance (may fail for some inputs)
        try:
            gamma_deviance = sklearn_metrics.mean_gamma_deviance(gt_flat, pred_flat)
        except Exception:
            gamma_deviance = np.nan
        
        # Angular error
        mean_angular_error = np.mean(self.compute_mae(self.gt, self.pred))
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = stats.pearsonr(gt_flat, pred_flat)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = stats.spearmanr(gt_flat, pred_flat)
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan
        
        metrics_dict.update({
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2_score": float(r2_score),
            "max_error": float(max_error),
            "gamma_deviance": float(gamma_deviance),
            "mean_angular_error": float(mean_angular_error),
            "pairwise_euclidean_mean": float(np.mean(p_dist)),
            "pairwise_euclidean_std": float(np.std(p_dist)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        })
        
        return metrics_dict
    
    def image_quality_metrics(self) -> Dict[str, float]:
        """
        Compute image quality metrics (PSNR, SSIM).
        
        Returns:
            dict: Metrics including:
                - psnr: Peak Signal-to-Noise Ratio
                - ssim: Structural Similarity Index
                
        Note:
            Requires gt and pred to be image arrays (H, W, C).
        """
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        metrics_dict = {}
        
        # PSNR
        try:
            psnr = peak_signal_noise_ratio(
                self.gt,
                self.pred,
                data_range=1.0,
            )
            metrics_dict["psnr"] = float(psnr)
        except Exception:
            metrics_dict["psnr"] = np.nan
        
        # SSIM
        try:
            # SSIM requires at least 7x7 images
            if self.gt.shape[0] >= 7 and self.gt.shape[1] >= 7:
                ssim = structural_similarity(
                    self.gt,
                    self.pred,
                    multichannel=True,
                    channel_axis=-1,
                    data_range=1.0,
                )
                metrics_dict["ssim"] = float(ssim)
            else:
                metrics_dict["ssim"] = np.nan
        except Exception:
            metrics_dict["ssim"] = np.nan
        
        return metrics_dict


def desc_stats(mat: np.ndarray) -> dict:
    """
    Compute descriptive statistics for a matrix.
    
    Args:
        mat: Input matrix (any shape)
        
    Returns:
        dict: Statistics including:
            - mean, median, std, var
            - min, max, range
            - q25, q75 (quartiles)
            - skewness, kurtosis
            - count (number of elements)
            
    Example:
        >>> mat = np.random.rand(100, 3)
        >>> stats_dict = desc_stats(mat)
        >>> "mean" in stats_dict
        True
        >>> stats_dict["count"]
        300
    """
    mat_flat = mat.flatten()
    
    # Remove NaNs for statistics
    mat_clean = mat_flat[~np.isnan(mat_flat)]
    
    if len(mat_clean) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "count": 0,
        }
    
    stats_dict = {
        "mean": float(np.mean(mat_clean)),
        "median": float(np.median(mat_clean)),
        "std": float(np.std(mat_clean)),
        "var": float(np.var(mat_clean)),
        "min": float(np.min(mat_clean)),
        "max": float(np.max(mat_clean)),
        "range": float(np.ptp(mat_clean)),
        "q25": float(np.percentile(mat_clean, 25)),
        "q75": float(np.percentile(mat_clean, 75)),
        "skewness": float(stats.skew(mat_clean)),
        "kurtosis": float(stats.kurtosis(mat_clean)),
        "count": int(len(mat_clean)),
    }
    
    return stats_dict


def compute_mae(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Compute mean angular error between two matrices.
    
    This is a convenience wrapper around Metrics.compute_mae().
    
    Args:
        mat1: First matrix
        mat2: Second matrix
        
    Returns:
        np.ndarray: Angular errors in degrees
        
    Example:
        >>> mat1 = np.array([[1.0, 0.0, 0.0]])
        >>> mat2 = np.array([[0.9, 0.1, 0.0]])
        >>> mae = compute_mae(mat1, mat2)
        >>> mae.shape
        (1,)
    """
    return Metrics.compute_mae(mat1, mat2)


def get_metrics(
    mat1: np.ndarray,
    mat2: np.ndarray,
    illuminant: np.ndarray,
    c_space: str = "srgb",
) -> np.ndarray:
    """
    Compute color difference metrics between two color matrices.
    
    Args:
        mat1: First color matrix (reference)
        mat2: Second color matrix (test)
        illuminant: Reference illuminant (xy chromaticity coordinates)
        c_space: Color space of input matrices ("srgb", "xyz", "lab", "xyy")
        
    Returns:
        metrics: Metrics object containing deltaE, mse, and mae
        
    Example:
        >>> ref = np.array([[1.0, 0.0, 0.0]])
        >>> test = np.array([[0.95, 0.05, 0.0]])
        >>> illuminant = np.array([0.31271, 0.32902])
        >>> m = get_metrics(ref, test, illuminant, "srgb")
        >>> m.deltaE.shape
        (1,)
        
    Note:
        Delta-E is computed in CIE L*a*b* color space.
    """
    # Convert to Lab
    mat1_lab = convert_to_lab(mat1, illuminant, c_space)
    mat2_lab = convert_to_lab(mat2, illuminant, c_space)
    
    # Compute Delta-E (CIE 2000)
    try:
        deltaE = colour.delta_E(mat1_lab, mat2_lab, method="CIE 2000")
    except Exception:
        # Fallback to CIE 1976 if CIE 2000 fails
        deltaE = colour.delta_E(mat1_lab, mat2_lab, method="CIE 1976")
    
    # Mean squared error (in original color space)
    mse = np.mean((mat1 - mat2) ** 2, axis=-1)
    
    # Mean angular error
    mae = compute_mae(mat1, mat2)
    
    return metrics(deltaE, mse, mae)


def arrange_metrics(
    metrics_before: Optional[metrics],
    metrics_after: Optional[metrics],
    name: str = "",
) -> dict:
    """
    Arrange metrics before and after correction into a dictionary.
    
    Args:
        metrics_before: Metrics before correction
        metrics_after: Metrics after correction
        name: Name prefix for metrics keys
        
    Returns:
        dict: Organized metrics with keys like:
            - {name}_deltaE_before_mean
            - {name}_deltaE_after_mean
            - {name}_improvement_percent
            
    Example:
        >>> ref = np.array([[1.0, 0.0, 0.0]])
        >>> before = np.array([[0.8, 0.1, 0.0]])
        >>> after = np.array([[0.95, 0.05, 0.0]])
        >>> illum = np.array([0.31271, 0.32902])
        >>> m_b = get_metrics(ref, before, illum, "srgb")
        >>> m_a = get_metrics(ref, after, illum, "srgb")
        >>> result = arrange_metrics(m_b, m_a, name="WB")
        >>> "WB_deltaE_improvement_percent" in result
        True
    """
    result = {}
    
    if name:
        prefix = f"{name}_"
    else:
        prefix = ""
    
    # Before metrics
    if metrics_before is not None:
        result.update({
            f"{prefix}deltaE_before_mean": float(np.mean(metrics_before.deltaE)),
            f"{prefix}deltaE_before_median": float(np.median(metrics_before.deltaE)),
            f"{prefix}deltaE_before_std": float(np.std(metrics_before.deltaE)),
            f"{prefix}mse_before_mean": float(np.mean(metrics_before.mse)),
            f"{prefix}mae_before_mean": float(np.mean(metrics_before.mae)),
        })
    
    # After metrics
    if metrics_after is not None:
        result.update({
            f"{prefix}deltaE_after_mean": float(np.mean(metrics_after.deltaE)),
            f"{prefix}deltaE_after_median": float(np.median(metrics_after.deltaE)),
            f"{prefix}deltaE_after_std": float(np.std(metrics_after.deltaE)),
            f"{prefix}mse_after_mean": float(np.mean(metrics_after.mse)),
            f"{prefix}mae_after_mean": float(np.mean(metrics_after.mae)),
        })
    
    # Compute improvement
    if metrics_before is not None and metrics_after is not None:
        deltaE_before = np.mean(metrics_before.deltaE)
        deltaE_after = np.mean(metrics_after.deltaE)
        
        if deltaE_before > 0:
            improvement = ((deltaE_before - deltaE_after) / deltaE_before) * 100
        else:
            improvement = 0.0
        
        result[f"{prefix}deltaE_improvement_percent"] = float(improvement)
        result[f"{prefix}deltaE_reduction"] = float(deltaE_before - deltaE_after)
    
    return result
