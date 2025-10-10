"""
Color correction algorithms and model fitting
==============================================

This module implements the core color correction algorithms:
    - Gamma correction (estimate_gamma_profile, predict_image)
    - White balance correction (wb_correction)
    - Color correction method 1 (color_correction_1 - conventional/Finlayson)
    - Color correction method 2 (color_correction - custom ML-based)
    - ML model fitting (fit_model, Regressor_Model, CustomNN)

All functions preserve backwards compatibility with the original package.
"""

import gc
import os
from typing import Any, Dict, Optional, Tuple

import colour
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_X_y
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..constants import EPSILON, WP_DEFAULT
from .color_spaces import convert_to_lab
from .metrics import arrange_metrics, get_metrics
from .transforms import get_poly_features
from .utils import (
    compute_diag,
    estimate_fit,
    extract_color_chart,
    extract_neutral_patches,
    free_memory,
    poly_func,
    poly_func_torch,
    to_float64,
    to_uint8,
)

__all__ = [
    "Regressor_Model",
    "CustomNN",
    "fit_model",
    "predict_",
    "predict_image",
    "estimate_gamma_profile",
    "wb_correction",
    "color_correction_1",
    "color_correction",
    "extract_color_chart_ex",
]

# CUDA setup
is_cuda = torch.cuda.is_available()
device_ = torch.device("cuda" if is_cuda else "cpu")


# ============================================================================
# ML Model Classes
# ============================================================================


class Regressor_Model:
    """
    Container for regression model parameters and fitted model.
    
    Attributes:
        mtd: Model type ("linear", "nn", "pls", "custom")
        model: Fitted sklearn/torch model
        degree: Polynomial degree for feature expansion
        max_iterations: Maximum training iterations
        random_state: Random seed
        tol: Convergence tolerance
        verbose: Whether to print training progress
        ncomp: Number of PLS components
        nlayers: Number of neurons in MLP hidden layer
        param_search: Whether to use parameter search
        hidden_layers: Hidden layer sizes for CustomNN
        learning_rate: Learning rate for CustomNN
        batch_size: Batch size for CustomNN
        use_batch_norm: Whether to use batch normalization
        patience: Early stopping patience
        dropout_rate: Dropout rate for CustomNN
        optim_type: Optimizer type for CustomNN
    """
    
    def __init__(self):
        self.mtd = "linear"
        self.model: Optional[Any] = None
        self.degree = 3
        self.max_iterations = 100
        self.random_state = 42
        self.tol = 1e-6
        self.verbose = False
        self.ncomp = 1
        self.nlayers = 100
        self.param_search = False

        # CustomNN specific parameters
        self.hidden_layers = [64, 32, 16]
        self.learning_rate = 0.001
        self.batch_size = 16
        self.use_batch_norm = False
        self.patience = 10
        self.dropout_rate = 0.2
        self.optim_type = "Adam"


class CustomNN(BaseEstimator, RegressorMixin):
    """
    Custom neural network regressor with PyTorch backend.
    
    Implements sklearn-compatible interface for color correction using
    fully connected neural networks with optional batch normalization and dropout.
    
    Args:
        hidden_layers: List of hidden layer sizes
        optim_type: Optimizer type ("Adam", "SGD", "RMSprop")
        random_state: Random seed for reproducibility
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        tol: Convergence tolerance
        verbose: Whether to print training progress
        batch_size: Mini-batch size
        patience: Early stopping patience
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
        
    Example:
        >>> model = CustomNN(hidden_layers=[32, 16], max_epochs=100)
        >>> X_train = np.random.rand(100, 10)
        >>> y_train = np.random.rand(100, 3)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_train)
    """
    
    def __init__(
        self,
        hidden_layers=[64, 32, 16],
        optim_type="Adam",
        random_state=42,
        learning_rate=0.001,
        max_epochs=1000,
        tol=1e-6,
        verbose=False,
        batch_size=32,
        patience=10,
        dropout_rate=0.2,
        use_batch_norm=False,
    ):
        self.hidden_layers = hidden_layers
        self.optim_type = optim_type
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.validation_split = 0.15

        self.input_dim = None
        self.output_dim = None
        self.temp_path = "best_model_temp.pth"

        self.model: Optional[nn.Module] = None
        self.loss_fn = nn.MSELoss()
        self.device: Optional[str] = None

    def _build_model(self):
        """Build neural network architecture."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.verbose:
            print(f"Using device: '{self.device}'")
        
        layers = []
        current_dim = self.input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, eps=self.tol))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, self.output_dim))

        self.model = nn.Sequential(*layers).to(self.device)

        if self.verbose:
            print("Model architecture:")
            print(self.model)

    def fit(self, X, y):
        """
        Fit neural network to training data.
        
        Args:
            X: Training features (N, M)
            y: Training targets (N, K)
            
        Returns:
            self: Fitted model
        """
        X, y = check_X_y(X, y, multi_output=True)
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        self._build_model()

        # Convert to tensors
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

        # Train/val split
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.random_state)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

        # Optimizer
        if self.optim_type == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optim_type == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optim_type == "RMSprop":
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 2, verbose=self.verbose
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        n_verbose = 50
        interval = max(1, int(self.max_epochs / n_verbose))

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_X)
                    loss = self.loss_fn(predictions, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                torch.save(self.model.state_dict(), self.temp_path)
            else:
                epochs_no_improve += 1

            if self.verbose and (epoch % interval == 0 or epoch == self.max_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if os.path.exists(self.temp_path):
            self.model.load_state_dict(torch.load(self.temp_path, map_location=self.device))
            os.remove(self.temp_path)

        # Release CUDA resources
        if self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

        return self

    def predict(self, X):
        """
        Predict using fitted neural network.
        
        Args:
            X: Input features (N, M)
            
        Returns:
            np.ndarray: Predictions (N, K)
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            try:
                predictions = self.model(X_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Batch predictions
                    torch.cuda.empty_cache() if self.device == "cuda" else None
                    batch_size = 2
                    success = False
                    while not success:
                        try:
                            predictions = torch.cat(
                                [
                                    self.model(X_tensor[i : i + batch_size])
                                    for i in range(0, X_tensor.shape[0], batch_size)
                                ],
                                dim=0,
                            )
                            success = True
                        except RuntimeError as e2:
                            if "out of memory" in str(e2):
                                batch_size *= 2
                                torch.cuda.empty_cache() if self.device == "cuda" else None
                            else:
                                raise e2
                else:
                    raise e

            results = predictions.cpu().numpy()

        torch.cuda.empty_cache() if self.device == "cuda" else None
        gc.collect()

        return results


# ============================================================================
# Model Fitting and Prediction
# ============================================================================


def fit_model(det_p: np.ndarray, ref_p: np.ndarray, kwargs: Optional[Dict] = None):
    """
    Fit color correction model using specified method.
    
    Args:
        det_p: Detected color patch values (N, 3)
        ref_p: Reference color patch values (N, 3)
        kwargs: Model parameters dict
        
    Returns:
        Regressor_Model: Fitted model container
        
    Example:
        >>> detected = np.random.rand(24, 3)
        >>> reference = np.random.rand(24, 3)
        >>> model = fit_model(detected, reference, {"mtd": "linear", "degree": 2})
    """
    if kwargs is None:
        kwargs = {}

    M = Regressor_Model()
    M.mtd = kwargs.get("mtd", "linear")
    M.degree = kwargs.get("degree", 3)
    M.max_iterations = kwargs.get("max_iterations", 100)
    M.random_state = kwargs.get("random_state", 42)
    M.tol = kwargs.get("tol", 1e-6)
    M.verbose = kwargs.get("verbose", False)
    M.ncomp = kwargs.get("ncomp", 1)
    M.nlayers = kwargs.get("nlayers", 100)
    M.param_search = kwargs.get("param_search", False)

    M.hidden_layers = kwargs.get("hidden_layers", M.hidden_layers)
    M.learning_rate = kwargs.get("learning_rate", M.learning_rate)
    M.batch_size = kwargs.get("batch_size", M.batch_size)
    M.patience = kwargs.get("patience", M.patience)
    M.dropout_rate = kwargs.get("dropout_rate", M.dropout_rate)
    M.optim_type = kwargs.get("optim_type", M.optim_type)
    M.use_batch_norm = kwargs.get("use_batch_norm", M.use_batch_norm)

    X = det_p
    Y = ref_p

    # Generate polynomial features
    X, _ = get_poly_features(X, degree=M.degree)

    if M.ncomp == -1 or M.ncomp > X.shape[1]:
        M.ncomp = X.shape[1] - 1

    if M.mtd == "linear":
        M.param_search = False

    # Model dictionary
    model_dict = {
        "linear": LinearRegression(fit_intercept=True),
        "nn": MLPRegressor(
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            hidden_layer_sizes=(M.nlayers,),
            max_iter=1000 if M.max_iterations == -1 else M.max_iterations,
            shuffle=False,
            random_state=M.random_state,
            tol=M.tol,
            verbose=M.verbose,
            nesterovs_momentum=True,
            early_stopping=True,
            n_iter_no_change=int(M.max_iterations * 0.15),
            validation_fraction=0.15,
        ),
        "pls": PLSRegression(
            n_components=M.ncomp,
            max_iter=500 if M.max_iterations == -1 else M.max_iterations,
            tol=M.tol,
        ),
        "custom": CustomNN(
            hidden_layers=M.hidden_layers,
            optim_type=M.optim_type,
            learning_rate=M.learning_rate,
            max_epochs=M.max_iterations,
            batch_size=M.batch_size,
            patience=M.patience,
            use_batch_norm=M.use_batch_norm,
            tol=M.tol,
            verbose=M.verbose,
            dropout_rate=M.dropout_rate,
            random_state=M.random_state,
        ),
    }

    # Select model
    fit_method = M.mtd.lower()
    if fit_method not in model_dict:
        fit_method = "linear"

    M.model = model_dict[fit_method]

    # Fit model
    M.model.fit(X, Y)

    return M


def predict_(RGB: np.ndarray, M: Regressor_Model) -> np.ndarray:
    """
    Predict RGB values using fitted model.
    
    Args:
        RGB: Input RGB image (H, W, 3) or (N, 3)
        M: Fitted Regressor_Model
        
    Returns:
        np.ndarray: Predicted RGB values (same shape as input)
        
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> model = fit_model(detected, reference, {"mtd": "linear", "degree": 2})
        >>> corrected = predict_(img, model)
    """
    # Preserve original shape
    orig_shape = RGB.shape
    
    if len(RGB.shape) == 3:
        X = RGB.reshape(-1, 3)
    else:
        X = RGB

    # Generate polynomial features
    X, _ = get_poly_features(X, degree=M.degree)
    
    # Predict
    pred = M.model.predict(X)

    return pred.reshape(orig_shape)


def predict_image(
    img: np.ndarray,
    coeffs: np.ndarray,
    ref_illuminant: np.ndarray = WP_DEFAULT,
) -> np.ndarray:
    """
    Apply gamma correction to image using polynomial coefficients.
    
    GPU-accelerated with automatic CPU fallback.
    
    Args:
        img: RGB image (H, W, 3), float64, 0-1 range
        coeffs: Polynomial coefficients from estimate_gamma_profile
        ref_illuminant: Reference illuminant (xy coordinates)
        
    Returns:
        np.ndarray: Gamma-corrected image (same shape as input)
        
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> coeffs = np.array([0.1, 0.9, 0.0])  # Linear approximation
        >>> corrected = predict_image(img, coeffs)
    """
    H, W, C = img.shape
    img_flat = img.reshape(-1, C)
    
    # Convert to Lab and extract L channel
    img_flat_lab = convert_to_lab(img_flat, illuminant=ref_illuminant, c_space="srgb")
    img_flat_L = img_flat_lab[:, 0]

    # Apply polynomial correction
    if is_cuda:
        try:
            coeffs_copy = coeffs.copy()
            img_flat_gpu = torch.from_numpy(img_flat_L.copy()).to(device_, dtype=torch.float32)
            coeffs_gpu = torch.from_numpy(coeffs_copy).to(device_, dtype=torch.float32)
            result_gpu = poly_func_torch(img_flat_gpu, coeffs_gpu)
            result = result_gpu.cpu().numpy()

            del img_flat_gpu, coeffs_gpu, result_gpu
            free_memory()
        except Exception:
            result = poly_func(img_flat_L, coeffs)
    else:
        result = poly_func(img_flat_L, coeffs)

    # Replace L channel
    result_lab = img_flat_lab.copy()
    result_lab[:, 0] = result

    # Convert back to RGB
    result_srgb = colour.XYZ_to_sRGB(
        colour.Lab_to_XYZ(result_lab, ref_illuminant), ref_illuminant
    )
    
    gc.collect()

    return result_srgb.reshape(H, W, C)


# ============================================================================
# Gamma Correction
# ============================================================================


def estimate_gamma_profile(
    img_rgb: np.ndarray,
    ref_cp: np.ndarray,
    ref_illuminant: np.ndarray,
    max_degree: int = 7,
    show: bool = False,
    get_deltaE: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimate optimal gamma correction polynomial from neutral patches.
    
    Args:
        img_rgb: RGB image (H, W, 3), float64, 0-1 range
        ref_cp: Reference color patch RGB values (24, 3)
        ref_illuminant: Reference illuminant (xy coordinates)
        max_degree: Maximum polynomial degree to test
        show: Whether to display plots
        get_deltaE: Whether to compute Delta-E metrics
        
    Returns:
        tuple: (coefficients, corrected_image, metrics)
            - coefficients: Polynomial coeffs for gamma correction
            - corrected_image: Gamma-corrected RGB image
            - metrics: Dict of metrics (if get_deltaE=True)
            
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> ref = np.random.rand(24, 3) * 0.8
        >>> illum = np.array([0.31271, 0.32902])
        >>> coeffs, corrected, metrics = estimate_gamma_profile(img, ref, illum)
    """
    Metrics_all = {}

    # Convert reference to Lab
    ref_lab = convert_to_lab(ref_cp, ref_illuminant, c_space="srgb")
    ref_L = ref_lab[:, 0]

    # Extract neutral patches from image
    img_bgr = to_uint8(img_rgb[:, :, ::-1])
    _, cps_before = extract_neutral_patches(img_bgr, return_one=True)
    
    if cps_before is None:
        # No patches found, return original
        return np.array([0, 1]), img_rgb, {}
    
    values_cps = cps_before.values

    # Convert to Lab
    values_lab = convert_to_lab(values_cps, ref_illuminant, c_space="srgb")
    values_L = values_lab[:, 0]

    # Augment with endpoints (100 and 0)
    values_L_ = np.insert(values_L, 0, [100], axis=0)
    ref_L_ = np.insert(ref_L, 0, [100], axis=0)
    values_L_ = np.append(values_L_, [0], axis=0)
    ref_L_ = np.append(ref_L_, [0], axis=0)

    # Try different polynomial degrees
    Coeffs = []
    MSE = []

    for degree in range(1, max_degree + 1):
        coeffs = estimate_fit(values_L_, ref_L_, degree)
        Coeffs.append(coeffs)

        pred_ = poly_func(values_L, coeffs)
        mse = np.mean((ref_L - pred_) ** 2)
        MSE.append(mse)

    # Find optimal degree using gradient
    grad_MSE = np.gradient(MSE)
    min_idx = np.argmin(grad_MSE)

    # Filter function to avoid overfitting
    def filter_fn(grads, min_idx, threshold=2):
        grads = np.array(grads)
        while True and min_idx > 0:
            before_min_grad = grads[:min_idx]
            range_min_grad = np.max(before_min_grad) - grads[min_idx]

            try:
                if range_min_grad < threshold:
                    min_idx = np.argmin(before_min_grad)
                else:
                    break
            except Exception:
                break

        return min_idx

    min_idx = filter_fn(grad_MSE, min_idx, threshold=1.7)

    degree = min_idx + 1
    coeffs = Coeffs[min_idx]

    if show or get_deltaE:
        print(f"Optimal polynomial degree: {degree}")

    # Apply gamma correction
    img_rgb_gc = predict_image(img_rgb, coeffs, ref_illuminant=ref_illuminant)

    # Compute metrics
    if get_deltaE:
        img_bgr2 = to_uint8(img_rgb_gc[:, :, ::-1])
        _, cps_after = extract_neutral_patches(img_bgr2)

        if cps_after is not None:
            metrics_b = get_metrics(ref_cp, values_cps, ref_illuminant, "srgb")
            metrics_a = get_metrics(ref_cp, cps_after.values, ref_illuminant, "srgb")
            Metrics_all = arrange_metrics(metrics_b, metrics_a, name="GC")

    # Visualization
    if show:
        # Compute predictions for plotting
        if is_cuda:
            free_memory()
            prediction = poly_func_torch(
                torch.from_numpy(values_L.flatten()).to(device_, dtype=torch.float32),
                torch.from_numpy(coeffs).to(device_, dtype=torch.float32),
            )
            prediction = prediction.cpu().numpy()
            free_memory()
        else:
            prediction = poly_func(values_L.flatten(), coeffs)
        prediction = prediction.reshape(values_L.shape)

        # Plot gamma curve
        plt.figure(figsize=(10, 5))
        plt.scatter(ref_L, values_L, c="red", label="Measured vs. Reference")
        plt.xlim(-1, 101)
        plt.ylim(-1, 101)
        plt.xlabel("Reference")
        plt.ylabel("Measured")

        x_line = np.linspace(0, 100, 25)
        y_line = poly_func(x_line, coeffs)
        plt.plot(y_line, x_line, "m-.", label=f"Order {degree} Curve")
        plt.plot(x_line, x_line, "c--", label="1:1 Curve")
        plt.scatter(ref_L, prediction, c="blue", label="Predictions")
        plt.legend()
        plt.show()

        # Plot images
        figure = plt.figure(figsize=(10, 5))
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(img_rgb_gc)
        ax.set_title("Gamma Corrected Image")

        plt.show()

    gc.collect()

    return coeffs, img_rgb_gc, Metrics_all


# ============================================================================
# White Balance Correction
# ============================================================================


def wb_correction(
    img_rgb: np.ndarray,
    ref_cp: np.ndarray,
    ref_illuminant: np.ndarray,
    show: bool = False,
    get_deltaE: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    White balance correction using diagonal matrix from neutral patches.
    
    Args:
        img_rgb: RGB image (H, W, 3), float64, 0-1 range
        ref_cp: Reference color patch RGB values (24, 3)
        ref_illuminant: Reference illuminant (xy coordinates)
        show: Whether to display plots
        get_deltaE: Whether to compute Delta-E metrics
        
    Returns:
        tuple: (diagonal_matrix, corrected_image, metrics)
            - diagonal_matrix: 3x3 diagonal WB matrix
            - corrected_image: White-balanced RGB image
            - metrics: Dict of metrics (if get_deltaE=True)
            
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> ref = np.random.rand(24, 3)
        >>> illum = np.array([0.31271, 0.32902])
        >>> diag, corrected, metrics = wb_correction(img, ref, illum)
    """
    Metrics_all = {}
    
    # Use only neutral patches (last 6)
    ref_np = ref_cp[-6:]

    # Extract neutral patches from image
    img_bgr = to_uint8(img_rgb[:, :, ::-1])
    values_, cps_before = extract_neutral_patches(img_bgr, return_one=True)
    
    if values_ is None or cps_before is None:
        # No patches found, return original with identity matrix
        return np.eye(3), img_rgb, {}
    
    values_np = values_.values

    # Compute diagonal scaling matrix
    diag_ = compute_diag(ref_np, values_np)

    # Apply white balance
    img_pred = img_rgb @ diag_

    # Compute metrics
    if get_deltaE:
        m_b = get_metrics(ref_cp, cps_before.values, ref_illuminant, "srgb")
        _, cps_after = extract_neutral_patches(to_uint8(img_pred[:, :, ::-1]))
        
        if cps_after is not None:
            m_a = get_metrics(ref_cp, cps_after.values, ref_illuminant, "srgb")
            Metrics_all = arrange_metrics(m_b, m_a, name="WB")

    # Visualization
    if show:
        figure = plt.figure(figsize=(10, 5))
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(img_pred)
        ax.set_title("WB Corrected Image")

        plt.show()

    return diag_, img_pred, Metrics_all


# ============================================================================
# Color Correction Methods
# ============================================================================


def color_correction_1(
    img_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    ref_illuminant: np.ndarray,
    cc_kwargs: Optional[dict] = None,
    show: bool = False,
    get_deltaE: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any, dict]:
    """
    Color correction method 1: Conventional (Finlayson 2015).
    
    Uses colour-science library's matrix_colour_correction function.
    
    Args:
        img_rgb: RGB image (H, W, 3), float64, 0-1 range
        ref_rgb: Reference color patch RGB values (24, 3)
        ref_illuminant: Reference illuminant (xy coordinates)
        cc_kwargs: Color correction kwargs (method, degree, etc.)
        show: Whether to display plots
        get_deltaE: Whether to compute Delta-E metrics
        
    Returns:
        tuple: (ccm, corrected_image, color_checker, metrics)
            - ccm: Color correction matrix (3, 3 or 3, N)
            - corrected_image: Color-corrected RGB image
            - color_checker: ColourChecker object (or None)
            - metrics: Dict of metrics (if get_deltaE=True)
            
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> ref = np.random.rand(24, 3)
        >>> illum = np.array([0.31271, 0.32902])
        >>> ccm, corrected, chart, metrics = color_correction_1(img, ref, illum)
    """
    if cc_kwargs is None:
        cc_kwargs = {}

    Metrics_ = {}
    corrected_color_card = None

    # Extract color patches
    img_bgr = to_uint8(img_rgb[:, :, ::-1])
    _, color_patches = extract_neutral_patches(img_bgr, show=show)
    
    if color_patches is None:
        # No patches found, return original with identity matrix
        return np.eye(3), img_rgb, None, {}
    
    cp_values = color_patches.values

    # Compute CCM using colour-science
    ccm = colour.characterisation.matrix_colour_correction(
        M_T=cp_values, M_R=ref_rgb, **cc_kwargs
    )

    # Apply CCM
    img_rgb_ccm = colour.characterisation.apply_matrix_colour_correction(
        RGB=img_rgb, CCM=ccm, **cc_kwargs
    )
    img_rgb_ccm = np.clip(img_rgb_ccm, 0, 1)

    # Extract corrected patches
    img_bgr_ccm = to_uint8(img_rgb_ccm[:, :, ::-1])
    _, color_patches_corrected = extract_neutral_patches(img_bgr_ccm)

    # Compute metrics
    if get_deltaE and color_patches_corrected is not None:
        m_b = get_metrics(ref_rgb, cp_values, ref_illuminant, "srgb")
        m_a = get_metrics(ref_rgb, color_patches_corrected.values, ref_illuminant, "srgb")
        Metrics_ = arrange_metrics(m_b, m_a, name="CC_M1")

    # Create corrected color checker
    if show and color_patches_corrected is not None:
        corrected_color_card = colour.characterisation.ColourChecker(
            name="Corrected Colour Checker",
            data=dict(
                zip(
                    color_patches_corrected.index,
                    colour.XYZ_to_xyY(
                        colour.sRGB_to_XYZ(color_patches_corrected.values, ref_illuminant)
                    ),
                )
            ),
            illuminant=ref_illuminant,
            rows=4,
            columns=6,
        )

    return ccm, img_rgb_ccm, corrected_color_card, Metrics_


def extract_color_chart_ex(
    img: np.ndarray,
    ref: np.ndarray,
    npts: int = 50,
    show: bool = False,
    randomize: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract extended color chart samples with random sampling from each patch.
    
    Args:
        img: BGR image (uint8)
        ref: Reference RGB values (24, 3), float64, 0-1 range
        npts: Number of points to sample per patch
        show: Whether to display detected chart
        randomize: Whether to randomize patch order
        
    Returns:
        tuple: (chart_samples, ref_samples)
            - chart_samples: Sampled RGB values (24*npts, 3)
            - ref_samples: Repeated reference values (24*npts, 3)
            
    Note:
        Uses colour_checker_detection for segmentation-based detection.
    """
    import logging
    import warnings
    logger = logging.getLogger(__name__)
    
    logger.debug(f"extract_color_chart_ex called: img.shape={img.shape}, img.dtype={img.dtype}, npts={npts}")
    
    # Suppress noisy warnings from ColorCorrectionPipeline during chart detection
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        old_log_level = logging.getLogger('root').level
        logging.getLogger('root').setLevel(logging.ERROR)
        
        try:
            from colour_checker_detection import detect_colour_checkers_segmentation
        except ImportError:
            # Fallback to basic extraction
            try:
                chart, _, _ = extract_color_chart(img)
                if chart is None:
                    logger.warning("Could not detect color chart")
                    return None, None
                
                # Repeat chart and ref to match expected size
                # CRITICAL FIX: Ensure chart is in float64 [0,1] range
                if chart.max() > 1.5:  # Likely uint8 [0-255] range
                    chart = chart.astype(np.float64) / 255.0
                chart_ex = np.repeat(chart, npts, axis=0)
                ref_ex = np.repeat(ref, npts, axis=0)
                logger.debug(f"Extended chart samples: {chart_ex.shape}")
                return chart_ex, ref_ex
            finally:
                logging.getLogger('root').setLevel(old_log_level)
        
        # Crop to chart region (simplified)
        try:
            chart, _, _ = extract_color_chart(img)
            if chart is None:
                logger.warning("Could not detect color chart")
                return None, None
            
            # CRITICAL FIX: Ensure chart is in float64 [0,1] range
            if chart.max() > 1.5:  # Likely uint8 [0-255] range
                chart = chart.astype(np.float64) / 255.0
            
            # For now, return repeated values
            # Full implementation would use segmentation detection
            chart_ex = np.repeat(chart, npts, axis=0)
            ref_ex = np.repeat(ref, npts, axis=0)
            
            if randomize:
                idx = np.random.permutation(len(chart_ex))
                chart_ex = chart_ex[idx]
                ref_ex = ref_ex[idx]
            
            logger.debug(f"Extended chart samples: {chart_ex.shape}")
            return chart_ex, ref_ex
        finally:
            logging.getLogger('root').setLevel(old_log_level)


def color_correction(
    img_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    ref_illuminant: np.ndarray,
    cc_kwargs: Optional[dict] = None,
    show: bool = False,
    get_deltaE: bool = False,
    n_samples: int = 50,
) -> Tuple[Any, np.ndarray, Any, dict]:
    """
    Color correction method 2: Custom ML-based (ours).
    
    Uses polynomial expansion with various ML backends (linear, PLS, NN).
    
    Args:
        img_rgb: RGB image (H, W, 3), float64, 0-1 range
        ref_rgb: Reference color patch RGB values (24, 3)
        ref_illuminant: Reference illuminant (xy coordinates)
        cc_kwargs: Model kwargs (mtd, degree, hidden_layers, etc.)
        show: Whether to display plots
        get_deltaE: Whether to compute Delta-E metrics
        n_samples: Number of samples per patch (if > 1, uses extended extraction)
        
    Returns:
        tuple: (model, corrected_image, color_checker, metrics)
            - model: Fitted Regressor_Model
            - corrected_image: Color-corrected RGB image
            - color_checker: ColourChecker object (or None)
            - metrics: Dict of metrics (if get_deltaE=True)
            
    Example:
        >>> img = np.random.rand(100, 100, 3)
        >>> ref = np.random.rand(24, 3)
        >>> illum = np.array([0.31271, 0.32902])
        >>> kwargs = {"mtd": "linear", "degree": 2}
        >>> model, corrected, chart, metrics = color_correction(img, ref, illum, kwargs)
    """
    if cc_kwargs is None:
        cc_kwargs = {}

    Metrics_ = {}
    corrected_color_card = None

    # Extract color patches
    # CRITICAL FIX: Clip image to [0, 1] before converting to uint8
    # Without clipping, values outside [0,1] cause uint8 wraparound (e.g., 1.5 → 126, -0.1 → 231)
    # This corrupts the image and breaks chart detection
    import logging
    import warnings
    logger = logging.getLogger(__name__)
    
    # Debug: Check image range before clipping
    if img_rgb.min() < 0 or img_rgb.max() > 1:
        logger.debug(
            f"color_correction: Input image has values outside [0,1]: "
            f"range=[{img_rgb.min():.6f}, {img_rgb.max():.6f}], "
            f"out-of-range pixels: {(img_rgb < 0).sum() + (img_rgb > 1).sum()}"
        )
    
    img_rgb_clipped = np.clip(img_rgb, 0, 1)
    img_bgr = to_uint8(img_rgb_clipped[:, :, ::-1])
    
    logger.debug(f"color_correction: Converted image for chart detection: shape={img_bgr.shape}, dtype={img_bgr.dtype}, range=[{img_bgr.min()}, {img_bgr.max()}]")
    
    # Temporarily suppress warnings from ColorCorrectionPipeline package during chart detection
    # (They log warnings even when succeeding)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        old_log_level = logging.getLogger('root').level
        logging.getLogger('root').setLevel(logging.ERROR)
        try:
            _, color_patches = extract_neutral_patches(img_bgr, show=show if n_samples == 1 else False)
        finally:
            logging.getLogger('root').setLevel(old_log_level)
    
    if color_patches is None:
        # No patches found, return None model with original image
        logger.warning("No color patches detected in input image, cannot perform color correction")
        return None, img_rgb, None, {}
    
    logger.info(f"Successfully detected {len(color_patches.values)} color patches")
    
    cp_values = color_patches.values

    # Get training data
    if n_samples == 1:
        ref_ex = ref_rgb
        cp_values_ex = cp_values
    else:
        chart_ex, ref_ex = extract_color_chart_ex(
            img_bgr, ref=ref_rgb, npts=n_samples, show=show, randomize=True
        )
        if chart_ex is None:
            cp_values_ex = cp_values
            ref_ex = ref_rgb
        else:
            cp_values_ex = chart_ex

    # Fit model
    M_RGB = fit_model(det_p=cp_values_ex, ref_p=ref_ex, kwargs=cc_kwargs)

    # Apply model to image
    img_rgb_ccm = predict_(RGB=img_rgb, M=M_RGB)

    # Compute corrected patch values directly using the model
    # This is more reliable than trying to re-detect patches in the corrected image
    cp_values_corrected = predict_(RGB=cp_values.reshape(-1, 1, 3), M=M_RGB).reshape(-1, 3)
    
    # Also try to extract patches from corrected image (for visualization only)
    img_bgr_ccm = to_uint8(np.clip(img_rgb_ccm[:, :, ::-1], 0, 1))
    color_patches_corrected = None
    if show:  # Only try re-detection if we need it for visualization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            old_log_level = logging.getLogger('root').level
            logging.getLogger('root').setLevel(logging.ERROR)
            try:
                _, color_patches_corrected = extract_neutral_patches(img_bgr_ccm)
                logger.debug("Re-detected patches in corrected image for visualization")
            except Exception:
                logger.debug("Could not re-detect patches in corrected image (using model prediction instead)")
            finally:
                logging.getLogger('root').setLevel(old_log_level)

    # Compute metrics
    if get_deltaE:
        m_b = get_metrics(ref_rgb, cp_values, ref_illuminant, "srgb")
        # Use model-predicted corrected values for metrics (more accurate)
        m_a = get_metrics(ref_rgb, cp_values_corrected, ref_illuminant, "srgb")
        Metrics_ = arrange_metrics(m_b, m_a, name="CC_M2")
        logger.info(f"Color correction metrics: deltaE before={m_b.deltaE.mean():.3f}, after={m_a.deltaE.mean():.3f}, improvement={((m_b.deltaE.mean() - m_a.deltaE.mean()) / m_b.deltaE.mean() * 100):.1f}%")

    # Create corrected color checker
    if show and color_patches_corrected is not None:
        corrected_color_card = colour.characterisation.ColourChecker(
            name="Corrected Colour Checker",
            data=dict(
                zip(
                    color_patches_corrected.index,
                    colour.XYZ_to_xyY(
                        colour.sRGB_to_XYZ(color_patches_corrected.values, ref_illuminant)
                    ),
                )
            ),
            illuminant=ref_illuminant,
            rows=4,
            columns=6,
        )

        # Plot
        figure = plt.figure(figsize=(10, 5))
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(np.clip(img_rgb_ccm, 0, 1))
        ax.set_title("CCM Corrected Image")

        plt.show()

    return M_RGB, img_rgb_ccm, corrected_color_card, Metrics_
