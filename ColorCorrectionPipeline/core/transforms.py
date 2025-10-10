"""
Polynomial transformations and feature generation
==================================================

This module provides polynomial feature generation with GPU acceleration
support for color correction applications.

Functions:
    - get_poly_features: Generate polynomial features (CPU/GPU)
    - generate_powers_with_combinations_torch: GPU-accelerated feature generation
    - estimate_chunk_size: Estimate optimal chunk size for GPU processing
    - process_in_chunks: Process large arrays in chunks to avoid OOM
"""

import gc
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "get_poly_features",
    "generate_powers_with_combinations_torch",
    "process_in_chunks",
    "estimate_chunk_size",
]

# CUDA setup
is_cuda = torch.cuda.is_available()
device_ = torch.device("cuda" if is_cuda else "cpu")


def estimate_chunk_size(RGB: np.ndarray) -> int:
    """
    Estimate optimal chunk size for GPU processing based on available memory.
    
    Args:
        RGB: Input RGB array (N, 3)
        
    Returns:
        int: Number of chunks to split the array into
        
    Example:
        >>> rgb = np.random.rand(1000000, 3)
        >>> n_chunks = estimate_chunk_size(rgb)
        >>> n_chunks >= 1
        True
    """
    if not is_cuda:
        return 1
    
    try:
        # Get available GPU memory in bytes
        available_memory = torch.cuda.get_device_properties(device_).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_)
        free_memory = available_memory - allocated_memory
        
        # Estimate memory needed per element (conservative estimate)
        # Each float32 element needs 4 bytes, with overhead for operations
        bytes_per_element = 4 * 10  # Conservative factor of 10 for operations
        
        # Calculate how many elements we can process at once
        max_elements = free_memory // bytes_per_element
        total_elements = RGB.shape[0]
        
        # Calculate number of chunks needed
        n_chunks = max(1, int(np.ceil(total_elements / max_elements)))
        
        # Limit to reasonable number of chunks
        n_chunks = min(n_chunks, 100)
        
        return n_chunks
    except Exception:
        # Fallback if memory query fails
        return 4


def generate_powers_with_combinations_torch(
    features: torch.Tensor,
    names: list,
    degree: int = 1,
) -> Tuple[torch.Tensor, list]:
    """
    Generate polynomial features with combinations using PyTorch (GPU accelerated).
    
    This function computes all polynomial combinations up to the specified degree,
    including interaction terms.
    
    Args:
        features: Input features tensor (N, C) on GPU
        names: Feature names (e.g., ["r", "g", "b"])
        degree: Maximum polynomial degree
        
    Returns:
        tuple: (feature_tensor, feature_names)
            - feature_tensor: Polynomial features (N, M) where M is number of features
            - feature_names: List of feature name strings
            
    Example:
        >>> features = torch.rand(100, 3).cuda()
        >>> names = ["r", "g", "b"]
        >>> poly_features, poly_names = generate_powers_with_combinations_torch(
        ...     features, names, degree=2
        ... )
        >>> poly_features.shape[1] >= 10  # 1 + 3 + 3 + 3 = 10 features minimum
        True
        
    Note:
        For degree=2 with 3 features: generates 1, r, g, b, r², g², b², rg, rb, gb
        Approximately (degree + num_features)! / (degree! * num_features!) features
    """
    n_samples, n_features = features.shape
    
    # Start with constant term (all ones)
    poly_features = [torch.ones((n_samples, 1), dtype=features.dtype, device=features.device)]
    poly_names = ["1"]
    
    # Degree 1: original features
    poly_features.append(features)
    poly_names.extend(names)
    
    if degree >= 2:
        # Generate higher degree terms
        from itertools import combinations_with_replacement
        
        for d in range(2, degree + 1):
            # Generate all combinations with replacement for degree d
            for combo in combinations_with_replacement(range(n_features), d):
                # Compute product of features for this combination
                result = torch.ones((n_samples,), dtype=features.dtype, device=features.device)
                name_parts = []
                
                for idx in combo:
                    result = result * features[:, idx]
                    name_parts.append(names[idx])
                
                # Add to list
                poly_features.append(result.unsqueeze(1))
                poly_names.append("".join(name_parts))
    
    # Concatenate all features
    poly_tensor = torch.cat(poly_features, dim=1)
    
    return poly_tensor, poly_names


def process_in_chunks(
    RGB: np.ndarray,
    feature_names: list,
    degree: int,
    n_chunk: int,
) -> Tuple[np.ndarray, list]:
    """
    Process RGB matrix in smaller chunks to avoid GPU out-of-memory errors.
    
    Args:
        RGB: Input RGB values (N, 3)
        feature_names: Feature names (e.g., ["r", "g", "b"])
        degree: Polynomial degree
        n_chunk: Number of chunks to split into
        
    Returns:
        tuple: (polynomial_features, feature_names)
            - polynomial_features: Full polynomial feature matrix (N, M)
            - feature_names: List of feature name strings
            
    Example:
        >>> rgb = np.random.rand(10000, 3)
        >>> poly_feats, names = process_in_chunks(rgb, ["r", "g", "b"], 2, 4)
        >>> poly_feats.shape[0] == 10000
        True
    """
    X_poly_chunks = []
    names = None
    
    # Split matrix into n_chunk equal parts
    chunks = np.array_split(RGB, n_chunk, axis=0)
    
    for i in range(n_chunk):
        # Extract chunk
        RGB_chunk = chunks[i]
        
        # Move to GPU
        RGB_torch = torch.from_numpy(RGB_chunk).pin_memory().to(device_, dtype=torch.float32)
        
        # Compute polynomial features
        X_poly_chunk, names = generate_powers_with_combinations_torch(
            RGB_torch, feature_names, degree=degree
        )
        
        # Move back to CPU and store result
        X_poly_chunks.append(X_poly_chunk.cpu().numpy())
        
        # Free GPU memory
        del RGB_torch, X_poly_chunk
        if is_cuda:
            torch.cuda.empty_cache()
    
    # Combine processed chunks
    X_poly = np.vstack(X_poly_chunks)
    
    gc.collect()
    
    return X_poly, names


def get_poly_features(
    RGB: np.ndarray,
    degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate polynomial features from RGB values with GPU acceleration support.
    
    This function automatically selects CPU or GPU processing based on:
        - CUDA availability
        - Input size
        - Available GPU memory
        
    Args:
        RGB: Input RGB values (N, 3) in range [0, 1]
        degree: Polynomial degree (1-5 typically)
        
    Returns:
        tuple: (polynomial_features, feature_names)
            - polynomial_features: Feature matrix (N, M)
            - feature_names: Array of feature name strings
            
    Example:
        >>> rgb = np.random.rand(1000, 3)
        >>> poly_feats, names = get_poly_features(rgb, degree=2)
        >>> poly_feats.shape[0] == 1000
        True
        >>> len(names) >= 10  # At least 10 features for degree 2
        True
        
    Note:
        For large inputs, automatically uses chunked processing on GPU.
        Falls back to CPU (sklearn) if GPU processing fails.
    """
    feature_names = ["r", "g", "b"]
    gc.collect()
    
    try:
        if is_cuda and RGB.shape[0] > 1000:  # Use GPU for large inputs
            # Estimate chunk size based on available memory
            chunk_size = estimate_chunk_size(RGB)
            
            if chunk_size > 1:
                # Process in chunks
                X_poly, names = process_in_chunks(RGB, feature_names, degree, chunk_size)
            else:
                # Process all at once
                RGB_torch = torch.from_numpy(RGB).pin_memory().to(device_, dtype=torch.float32)
                X_poly_torch, names = generate_powers_with_combinations_torch(
                    RGB_torch, feature_names, degree=degree
                )
                X_poly = X_poly_torch.cpu().numpy()
                
                # Free memory
                del RGB_torch, X_poly_torch
                if is_cuda:
                    torch.cuda.empty_cache()
        else:
            # CPU case - use sklearn
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(RGB)
            names = poly.get_feature_names_out(feature_names)
    
    except Exception as e:
        # Fallback to CPU if GPU processing fails
        try:
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(RGB)
            names = poly.get_feature_names_out(feature_names)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to generate polynomial features. GPU error: {e}, CPU error: {e2}"
            )
    
    gc.collect()
    
    return X_poly, np.array(names)


def fit_polynomial_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """
    Fit a polynomial surface to 3D data points.
    
    Used primarily for flat-field correction to model illumination variations.
    
    Args:
        x: X coordinates (N,)
        y: Y coordinates (N,)
        z: Z values (intensity, N,)
        degree: Polynomial degree for surface
        
    Returns:
        np.ndarray: Polynomial coefficients
        
    Example:
        >>> x = np.random.rand(100)
        >>> y = np.random.rand(100)
        >>> z = x**2 + y**2 + np.random.randn(100) * 0.01
        >>> coeffs = fit_polynomial_surface(x, y, z, degree=2)
        >>> len(coeffs) >= 6  # At least 6 coefficients for degree 2
        True
        
    Note:
        Returns coefficients that can be used with numpy.polyval2d
    """
    # Create polynomial features from x and y
    xy = np.column_stack([x, y])
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(xy)
    
    # Fit using least squares
    coeffs, _, _, _ = np.linalg.lstsq(X_poly, z, rcond=None)
    
    return coeffs


def apply_polynomial_surface(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Apply polynomial surface coefficients to compute z values.
    
    Args:
        x: X coordinates (N,)
        y: Y coordinates (N,)
        coeffs: Polynomial coefficients from fit_polynomial_surface
        degree: Polynomial degree used in fitting
        
    Returns:
        np.ndarray: Predicted z values (N,)
        
    Example:
        >>> x = np.array([0, 1, 2])
        >>> y = np.array([0, 1, 2])
        >>> coeffs = np.array([1, 0, 0, 0, 0, 0])  # Constant surface
        >>> z = apply_polynomial_surface(x, y, coeffs, degree=2)
        >>> np.allclose(z, [1, 1, 1])
        True
    """
    # Create polynomial features
    xy = np.column_stack([x, y])
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(xy)
    
    # Apply coefficients
    z = X_poly @ coeffs
    
    return z
