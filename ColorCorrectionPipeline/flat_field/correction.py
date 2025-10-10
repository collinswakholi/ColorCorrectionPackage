"""
Flat-field correction module
=============================

This module implements flat-field correction (FFC) to compensate for uneven
illumination across the field of view. It detects or manually crops a white
reference plane, fits a polynomial surface to describe the intensity distribution,
and applies correction to images.

Key Features:
    - YOLO-based automatic white plane detection
    - Manual ROI selection fallback
    - Polynomial surface fitting (configurable degree)
    - Multiple ML backends (linear, NN, PLS, SVM)
    - L*a*b* color space processing
    - Visualization utilities

Example:
    >>> import cv2
    >>> from color_correc_optim.flat_field import FlatFieldCorrection
    >>> 
    >>> white_img = cv2.imread("white_background.jpg")
    >>> ffc_params = {"manual_crop": False, "bins": 50, "smooth_window": 5}
    >>> fit_params = {"degree": 5, "fit_method": "nn", "max_iter": 1000}
    >>> 
    >>> ffc = FlatFieldCorrection(white_img, **ffc_params)
    >>> multiplier = ffc.compute_multiplier(**fit_params)
    >>> corrected = ffc.apply_ffc(test_img, multiplier, show=True)
"""

import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from ultralytics import YOLO

from ..constants import MODEL_PATH

__all__ = ["FlatFieldCorrection"]

# Type aliases
FLOAT = np.float32
UINT8 = np.uint8

# Colormaps for visualization
CMAPS = ["viridis", "plasma", "jet", "Greys", "cividis"]


class FlatFieldCorrection:
    """
    Flat-field correction using white plane detection and polynomial surface fitting.
    
    This class performs flat-field correction by:
    1. Detecting or manually selecting a white reference region
    2. Computing intensity multiplier from the white region
    3. Fitting a polynomial surface to extrapolate across full image
    4. Applying correction via L channel multiplication
    
    Attributes:
        img: Input BGR image (uint8)
        model_path: Path to YOLO model for plane detection
        manual_crop: Whether to use manual ROI selection
        show: Whether to display intermediate plots
        bins: Number of bins for intensity sampling
        smooth_window: Window size for Gaussian smoothing
        crop_rect: Manual crop rectangle [x1, y1, x2, y2]
        model: YOLO model instance (if not manual_crop)
        img_cropped: Cropped white plane region
        cropped_multiplier: Multiplier computed from cropped region
        final_multiplier: Full-image multiplier surface
        is_color: Whether image is color (vs grayscale)
        
    Example:
        >>> ffc = FlatFieldCorrection(
        ...     img=white_img,
        ...     model_path="path/to/yolo.pt",
        ...     manual_crop=False,
        ...     bins=50,
        ...     smooth_window=5,
        ...     show=False
        ... )
        >>> multiplier = ffc.compute_multiplier(
        ...     degree=5,
        ...     fit_method="nn",
        ...     max_iter=1000,
        ...     tol=1e-8
        ... )
        >>> corrected_img = ffc.apply_ffc(test_img, multiplier, show=True)
    """
    
    def __init__(self, img: Optional[np.ndarray] = None, **kwargs):
        """
        Initialize FlatFieldCorrection.
        
        Args:
            img: BGR image (uint8), typically white background image
            **kwargs: Configuration parameters
                model_path: Path to YOLO model (default: MODEL_PATH constant)
                manual_crop: Force manual ROI selection (default: False)
                show: Display intermediate plots (default: False)
                bins: Bins for intensity sampling (default: 50)
                smooth_window: Gaussian smoothing window size (default: 5)
                crop_rect: Pre-defined crop rectangle [x1, y1, x2, y2]
        """
        self.img = img
        self.model_path = kwargs.get("model_path", MODEL_PATH)
        self.manual_crop = kwargs.get("manual_crop", False)
        
        # Auto-enable manual crop if model not found
        if self.model_path == "" or not os.path.exists(self.model_path):
            self.manual_crop = True
            
        self.show = kwargs.get("show", False)
        self.bins = kwargs.get("bins", 50)
        self.smooth_window = kwargs.get("smooth_window", 5)
        self.crop_rect = kwargs.get("crop_rect", None)
        
        self.model: Optional[YOLO] = None
        self.img_cropped: Optional[np.ndarray] = None
        self.cropped_multiplier: Optional[np.ndarray] = None
        self.final_multiplier: Optional[np.ndarray] = None
        self.is_color = self.check_color(self.img) if self.img is not None else None
        
        # Load YOLO model if not using manual crop
        if not self.manual_crop:
            self.model = YOLO(self.model_path)
            
    def check_color(self, img: np.ndarray) -> bool:
        """Check if image is color (3 channels) vs grayscale."""
        self.is_color = img.ndim == 3 and img.shape[2] == 3
        return self.is_color
    
    def resize_image(
        self,
        img: np.ndarray,
        factor: Optional[float] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image by factor or to specific size.
        
        Args:
            img: Input image
            factor: Scaling factor (if provided)
            size: Target size (height, width) (if provided)
            
        Returns:
            np.ndarray: Resized image
        """
        img_ = img
        if factor is not None:
            img_ = cv2.resize(
                img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
            )
        if size is not None:
            img_ = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        return img_
    
    def transform_extremity(
        self, x: np.ndarray, cut_off: float = 1.5, max_val: float = 2.0
    ) -> np.ndarray:
        """
        Transform extreme multiplier values using tanh to prevent over-correction.
        
        Args:
            x: Multiplier array
            cut_off: Threshold above which to apply transformation
            max_val: Maximum value parameter for tanh scaling
            
        Returns:
            np.ndarray: Transformed multiplier array
        """
        x_ = x.flatten()
        mask = x_ > cut_off
        x_[mask] = cut_off + (max_val - cut_off) * np.tanh(
            max_val * (x_[mask] - cut_off)
        )
        return x_.reshape(x.shape)
    
    def show_results(self, img_correct: np.ndarray, img_original: np.ndarray):
        """Display side-by-side comparison of corrected vs original image."""
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        
        if len(img_correct.shape) == 3:
            ax[0].imshow(cv2.cvtColor(img_correct, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        else:
            img_correct = cv2.normalize(
                img_correct, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            img_original = cv2.normalize(
                img_original, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            ax[0].imshow(img_correct, cmap="gray")
            ax[1].imshow(img_original, cmap="gray")
            
        ax[0].set_title("FF Corrected Image")
        ax[1].set_title("Original Image")
        plt.show()
        
        return fig
    
    def plot_intensity_distribution(
        self, Z: np.ndarray, Z_flat: np.ndarray, half: bool = False
    ):
        """
        Plot 3D intensity distribution before and after correction.
        
        Args:
            Z: Original intensity surface
            Z_flat: Flattened/corrected intensity surface
            half: Whether to plot only half of the surface
        """
        if half:
            shape = Z.shape
            wh_ = shape[0] // 2
            Z = Z[0:wh_, :]
            Z_flat = Z_flat[0:wh_, :]
        
        try:
            w, h, _ = Z.shape
        except:
            w, h = Z.shape
        
        bins = self.bins
        if w > self.bins or h > bins:
            x = np.linspace(0, w - 1, bins)
            y = np.linspace(0, h - 1, bins)
            X, Y = np.meshgrid(x, y)
            h_win = int((self.smooth_window - 1) / 2)
            
            Z_ = np.zeros_like(X)
            Z_flat_ = np.zeros_like(X)
            
            for i, x_ in enumerate(x):
                for j, y_ in enumerate(y):
                    x_, y_ = int(x_), int(y_)
                    x_bounds = max(0, x_ - h_win), min(w, x_ + h_win)
                    y_bounds = max(0, y_ - h_win), min(h, y_ + h_win)
                    
                    Z_[i, j] = np.mean(
                        Z[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]]
                    )
                    Z_flat_[i, j] = np.mean(
                        Z_flat[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]]
                    )
            
            Z_ = np.array(Z_)
            Z_flat_ = np.array(Z_flat_)
        else:
            Z_ = Z
            Z_flat_ = Z_flat
        
        fig = go.Figure(
            data=[
                go.Surface(z=Z_ - 10, opacity=1, colorscale="Viridis"),
                go.Surface(z=Z_flat_, opacity=0.3, colorscale="Jet"),
            ]
        )
        fig.update_layout(
            title="Intensity distribution",
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90),
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Intensity*"),
        )
        fig.update_scenes(zaxis_range=[0, 256])
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )
        fig.show()
    
    def plot_multiplier(self, multiplier: np.ndarray):
        """Display multiplier as 2D heatmap."""
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Multiplier")
        im = ax.imshow(multiplier, cmap="jet")
        plt.colorbar(im, ax=ax, orientation="vertical")
        plt.show()
        return fig
    
    def show_3d(self, img_list: List[np.ndarray], names: Optional[List[str]] = None):
        """
        Display multiple surfaces in 3D plot.
        
        Args:
            img_list: List of 2D arrays to plot as surfaces
            names: Optional names for legend
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        for i, img in enumerate(img_list):
            x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
            randi = np.random.randint(0, len(CMAPS))
            p = ax.plot_surface(x, y, img, cmap=CMAPS[randi], alpha=0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
        if names is not None:
            ax.legend(names)
        
        fig.colorbar(p, ax=ax)
        plt.show()
        return fig
    
    def detect_and_crop(self):
        """
        Detect white plane using YOLO or manual ROI selection.
        
        Sets self.crop_rect and self.img_cropped.
        """
        if not self.manual_crop:
            # YOLO detection
            sr = 0.95  # Shrink ratio to avoid edges
            results = self.model.predict(
                source=self.img,
                half=False,
                show=False,
                save=False,
                save_txt=False,
                conf=0.7,
                iou=0.6,
            )
            
            boxes = []
            probs = []
            for result in results:
                box_cpu = np.round(result.boxes.xyxy.cpu().numpy()).astype(int)
                prob_cpu = result.boxes.conf.cpu().numpy()
                boxes.append(box_cpu[0, :])
                probs.append(prob_cpu[0])
                
            boxes = np.array(boxes)
            probs = np.array(probs)
            
            if len(boxes) > 1:
                print(f"Warning: {len(boxes)} objects detected, using largest")
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_index = np.argmax(areas)
                selected_box = boxes[max_index]
                selected_prob = probs[max_index]
                print(
                    f"Selected object {max_index}: BB={selected_box}, prob={selected_prob:.3f}"
                )
            else:
                selected_box = boxes[0]
                selected_prob = probs[0]
            
            x1, y1, x2, y2 = selected_box
            x1 = int(x1 + (1 - sr) * (x2 - x1))
            y1 = int(y1 + (1 - sr) * (y2 - y1))
            x2 = int(x2 - (1 - sr) * (x2 - x1))
            y2 = int(y2 - (1 - sr) * (y2 - y1))
            
        else:
            # Manual ROI selection
            print('Select ROI manually. Press "ENTER" when done selecting ROI')
            cv2.namedWindow("Press 'ENTER' when done", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Press 'ENTER' when done", 1200, 800)
            rect = cv2.selectROI("Press 'ENTER' when done", self.img, True)
            cv2.destroyAllWindows()
            
            try:
                x1 = int(rect[0])
                y1 = int(rect[1])
                x2 = int(rect[0] + rect[2])
                y2 = int(rect[1] + rect[3])
            except:
                print("Warning: ROI not selected, using full image")
                x1 = 0
                y1 = 0
                x2 = self.img.shape[1]
                y2 = self.img.shape[0]
        
        self.crop_rect = [x1, y1, x2, y2]
        self.img_cropped = self.img[y1:y2, x1:x2]
        
        if self.show:
            img = self.img.copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.namedWindow("Image ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image ROI", 1200, 800)
            cv2.imshow("Image ROI", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_L(
        self, img: np.ndarray, smooth: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract L channel from image.
        
        Args:
            img: BGR image (uint8)
            smooth: Whether to apply Gaussian smoothing
            
        Returns:
            tuple: (L_channel, LAB_image)
                - L_channel: Luminance channel (uint8)
                - LAB_image: Full LAB image if color, else None
        """
        is_color = self.check_color(img)
        img_LAB = None
        
        if is_color:
            img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L = img_LAB[:, :, 0]
        else:
            L = img
            
        if smooth:
            L = cv2.GaussianBlur(L, (self.smooth_window, self.smooth_window), 0)
        
        return L, img_LAB
    
    def polynomial_features(
        self, X: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, PolynomialFeatures]:
        """
        Generate polynomial features from coordinates.
        
        Args:
            X: Coordinate array (N, 2)
            **kwargs: Parameters
                degree: Polynomial degree (default: 5)
                interactions: Whether to include only interaction terms
                
        Returns:
            tuple: (X_poly, feature_names, poly_object)
        """
        degree = kwargs.get("degree", 5)
        interactions = not (kwargs.get("interactions", False))
        poly = PolynomialFeatures(degree=degree, interaction_only=interactions)
        X_poly = poly.fit_transform(X)
        names = poly.get_feature_names_out(["x", "y"])
        return X_poly, names, poly
    
    def fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """
        Fit regression model to polynomial features.
        
        Args:
            X: Features (polynomial expanded coordinates)
            y: Target values (normalized multiplier)
            **kwargs: Model parameters
                fit_method: Method ("linear", "nn", "pls", "svm")
                max_iter: Maximum iterations
                tol: Convergence tolerance
                verbose: Whether to print progress
                rand_seed: Random seed
                
        Returns:
            Fitted sklearn model
        """
        method = kwargs.get("fit_method", "nn")
        max_iter = kwargs.get("max_iter", 1000)
        tol = kwargs.get("tol", 1e-8)
        verbose = kwargs.get("verbose", False)
        rand_seed = kwargs.get("rand_seed", 0)
        
        options = ["linear", "nn", "pls", "svm"]
        
        # Match method name
        fit_method = method.lower()
        if fit_method not in options:
            fit_method = "linear"
        
        model_dict = {
            "linear": LinearRegression(fit_intercept=True, n_jobs=8),
            "nn": MLPRegressor(
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                learning_rate_init=0.001,
                hidden_layer_sizes=(100,),
                max_iter=max_iter,
                shuffle=True,
                random_state=rand_seed,
                tol=tol,
                verbose=verbose,
                nesterovs_momentum=True,
                early_stopping=True,
                n_iter_no_change=int(max_iter * 0.1),
                validation_fraction=0.15,
            ),
            "pls": PLSRegression(
                n_components=np.shape(X)[1] - 1, max_iter=max_iter, tol=tol
            ),
            "svm": SVR(
                kernel="rbf",
                degree=3,
                verbose=verbose,
                epsilon=0.1,
                tol=tol,
                max_iter=max_iter,
            ),
        }
        
        if fit_method not in options:
            print(
                f"Warning: fit method '{fit_method}' not recognized. Using 'linear'."
            )
            fit_method = "linear"
        
        print(f"FFC fitting using method '{fit_method}'...")
        model = model_dict[fit_method]
        model.fit(X, y)
        
        return model
    
    def compute_multiplier(self, **kwargs) -> np.ndarray:
        """
        Compute flat-field multiplier from white image.
        
        This is the main method that:
        1. Detects/crops white plane region
        2. Computes local multiplier from cropped region
        3. Fits polynomial surface to extrapolate across full image
        4. Returns full-image multiplier
        
        Args:
            **kwargs: Fitting parameters
                degree: Polynomial degree (default: 5)
                interactions: Include interaction terms (default: False)
                fit_method: ML method (default: "nn")
                max_iter: Maximum iterations (default: 1000)
                tol: Tolerance (default: 1e-8)
                verbose: Print progress (default: False)
                rand_seed: Random seed (default: 0)
                
        Returns:
            np.ndarray: Multiplier surface (same size as input image)
            
        Example:
            >>> multiplier = ffc.compute_multiplier(
            ...     degree=5,
            ...     fit_method="nn",
            ...     max_iter=1000,
            ...     tol=1e-8
            ... )
        """
        # 1. Detect and crop white plane
        self.detect_and_crop()
        
        img_full = self.img.copy()
        img_cropped = self.img_cropped.copy()
        
        # Extract L channels
        L_full, _ = self.get_L(img_full, smooth=True)
        L_cropped, _ = self.get_L(img_cropped, smooth=True)
        
        # Compute cropped multiplier (inverse of normalized L)
        L_float = L_cropped.astype(FLOAT) / 255
        self.cropped_multiplier = np.max(L_float) / L_float
        
        flat_cropped = L_float * self.cropped_multiplier
        flat_cropped = (255 * flat_cropped).astype(UINT8)
        
        # 2. Extrapolate multiplier to full image using polynomial fitting
        if self.crop_rect is None:
            y1, x1, y2, x2 = 0, 0, self.img.shape[1], self.img.shape[0]
        else:
            y1, x1, y2, x2 = self.crop_rect
        
        # Sample multiplier at bins locations within cropped region
        x = np.linspace(x1, x2 - 1, self.bins)
        y = np.linspace(y1, y2 - 1, self.bins)
        X, Y = np.meshgrid(x, y)
        
        x_c = np.linspace(0, L_cropped.shape[0] - 1, self.bins)
        y_c = np.linspace(0, L_cropped.shape[1] - 1, self.bins)
        X_c, Y_c = np.meshgrid(x_c, y_c)
        
        h_win = int((self.smooth_window - 1) / 2)
        
        Z_m = np.ones_like(X_c)
        for i, x_ in enumerate(x_c):
            for j, y_ in enumerate(y_c):
                x_, y_ = int(x_), int(y_)
                x_l, x_h = [
                    max(0, x_ - h_win),
                    min(L_cropped.shape[0] - 1, x_ + h_win),
                ]
                y_l, y_h = [
                    max(0, y_ - h_win),
                    min(L_cropped.shape[1] - 1, y_ + h_win),
                ]
                Z_m[i, j] = np.mean(self.cropped_multiplier[x_l:x_h, y_l:y_h])
        
        Z_m = np.array(Z_m)
        
        # Flatten coordinates and values
        x_flat, y_flat = X.flatten(), Y.flatten()
        z_flat = Z_m.flatten()
        
        # Normalize to [0, 1] for stable fitting
        min_x, max_x = 0, L_full.shape[0]
        min_y, max_y = 0, L_full.shape[1]
        min_z, max_z = np.min(z_flat), np.max(z_flat)
        
        eps = 1e-15
        x_flat = (x_flat - min_x) / (max_x - min_x + eps)
        y_flat = (y_flat - min_y) / (max_y - min_y + eps)
        z_flat = (z_flat - min_z) / (max_z - min_z + eps)
        
        # Generate polynomial features
        xy_flat = np.stack([x_flat, y_flat], axis=1)
        xy_flat, names, poly = self.polynomial_features(xy_flat, **kwargs)
        
        # Fit model
        model = self.fit_model(xy_flat, z_flat, **kwargs)
        
        # Predict on full grid
        x_full = np.linspace(0, L_full.shape[0] - 1, self.bins)
        y_full = np.linspace(0, L_full.shape[1] - 1, self.bins)
        X_full, Y_full = np.meshgrid(x_full, y_full)
        
        X_full_flat = X_full.flatten()
        Y_full_flat = Y_full.flatten()
        
        x_full_flat = (X_full_flat - min_x) / (max_x - min_x + eps)
        y_full_flat = (Y_full_flat - min_y) / (max_y - min_y + eps)
        
        xy_full_flat = np.stack([x_full_flat, y_full_flat], axis=1)
        xy_full_flat = poly.transform(xy_full_flat)
        
        # Predict and denormalize
        f_multiplier = model.predict(xy_full_flat)
        f_multiplier = (f_multiplier * (max_z - min_z) + min_z).reshape(
            self.bins, self.bins
        )
        
        # Transform extreme values to prevent over-correction
        f_multiplier = self.transform_extremity(
            f_multiplier, max_val=1.8, cut_off=1.3
        )
        
        # Resize to full image dimensions
        f_multiplier = cv2.resize(
            f_multiplier,
            (L_full.shape[1], L_full.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        self.final_multiplier = f_multiplier
        
        # 3. Apply FFC to the image (for visualization if requested)
        img_corrected = self.apply_ffc(img_full)
        
        if self.show:
            self.show_3d([flat_cropped, L_cropped], names=["Flat", "Original"])
            self.show_3d([self.final_multiplier], names=["Final Multiplier"])
            self.show_results(img_corrected, img_full)
        
        gc.collect()
        return self.final_multiplier
    
    def apply_ffc(
        self,
        img: np.ndarray,
        multiplier: Optional[np.ndarray] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Apply flat-field correction to image.
        
        Multiplies L channel by multiplier surface and converts back to BGR.
        
        Args:
            img: BGR image (uint8) to correct
            multiplier: Multiplier surface (if None, uses self.final_multiplier)
            show: Whether to display before/after comparison
            
        Returns:
            np.ndarray: Corrected BGR image (uint8)
            
        Example:
            >>> corrected = ffc.apply_ffc(test_img, multiplier, show=True)
        """
        img_orig = img if not show else img.copy()
        assert img_orig.dtype == UINT8, "Image must be of type UINT8"
        
        if multiplier is not None:
            self.final_multiplier = multiplier
        
        w, h = img.shape[:2]
        w_o, h_o = self.final_multiplier.shape[:2]
        
        if (w, h) != (w_o, h_o):
            print(
                f"Warning: Image size {w}x{h} != multiplier size {w_o}x{h_o}. "
                f"Resizing image to match."
            )
            img = self.resize_image(img, size=(w_o, h_o))
        
        # Extract L channel
        L, img_LAB = self.get_L(img, smooth=False)
        
        # Multiply L channel by final_multiplier
        L_ = (L.astype(FLOAT) * self.final_multiplier).astype(UINT8)
        
        # Reconstruct image
        if self.check_color(img):
            img_LAB[:, :, 0] = L_
            img_corrected = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
        else:
            img_corrected = L_
        
        if show:
            self.show_results(img_corrected, img_orig)
        
        return np.clip(img_corrected, 0, 255)
