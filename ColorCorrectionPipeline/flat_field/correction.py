"""
Flat-field correction module
=============================

This module implements flat-field correction (FFC) to compensate for uneven
illumination across the field of view. It detects or manually crops a white
reference plane, fits a polynomial surface to describe the intensity distribution,
and applies correction to images.

Key Features:
    - OpenCV DNN automatic white plane detection
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
from typing import Any, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from ..constants import MODEL_PATH

__all__ = ["FlatFieldCorrection"]

# Type aliases
FLOAT = np.float32
UINT8 = np.uint8

# Colormaps for visualization
CMAPS = ["viridis", "plasma", "jet", "Greys", "cividis"]


class _PowersOnlyFeatures:
    """Polynomial transformer that omits x*y interaction terms."""

    def __init__(self, degree: int):
        self.degree = max(1, int(degree))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        features = [np.ones(X.shape[0], dtype=X.dtype)]
        for power in range(1, self.degree + 1):
            features.append(X[:, 0] ** power)
            features.append(X[:, 1] ** power)
        return np.column_stack(features)

    def get_feature_names_out(self, input_features: List[str]) -> np.ndarray:
        x_name, y_name = input_features
        names = ["1"]
        for power in range(1, self.degree + 1):
            names.append(x_name if power == 1 else f"{x_name}^{power}")
            names.append(y_name if power == 1 else f"{y_name}^{power}")
        return np.asarray(names)


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
        model_path: Path to OpenCV DNN ONNX model for plane detection
        manual_crop: Whether to use manual ROI selection
        show: Whether to display intermediate plots
        bins: Number of bins for intensity sampling
        smooth_window: Window size for Gaussian smoothing
        crop_rect: Manual crop rectangle [x1, y1, x2, y2]
        model: OpenCV DNN model instance (if available)
        img_cropped: Cropped white plane region
        cropped_multiplier: Multiplier computed from cropped region
        final_multiplier: Full-image multiplier surface
        is_color: Whether image is color (vs grayscale)
        
    Example:
        >>> ffc = FlatFieldCorrection(
        ...     img=white_img,
        ...     model_path="path/to/plane_detector.onnx",
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
                model_path: Path to ONNX detection model (default: MODEL_PATH constant)
                manual_crop: Force manual ROI selection (default: False)
                show: Display intermediate plots (default: False)
                bins: Bins for intensity sampling (default: 50)
                smooth_window: Gaussian smoothing window size (default: 5)
                crop_rect: Pre-defined crop rectangle [x1, y1, x2, y2]
        """
        self.img = img
        self.model_path = kwargs.get("model_path", MODEL_PATH)
        self.manual_crop = kwargs.get("manual_crop", False)
        self.show = kwargs.get("show", False)
        self.bins = kwargs.get("bins", 50)
        self.smooth_window = kwargs.get("smooth_window", 5)
        self.crop_rect = kwargs.get("crop_rect", None)
        self.conf_threshold = kwargs.get("conf_threshold", 0.7)
        self.iou_threshold = kwargs.get("iou_threshold", 0.6)
        self.dnn_input_size = kwargs.get("dnn_input_size", 512)
        
        self.model: Optional[Any] = None
        self.img_cropped: Optional[np.ndarray] = None
        self.cropped_multiplier: Optional[np.ndarray] = None
        self.final_multiplier: Optional[np.ndarray] = None
        self.is_color = self.check_color(self.img) if self.img is not None else None
        
        if not self.manual_crop and self.crop_rect is None:
            self.model = self._load_detection_model()
            
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

    def _load_detection_model(self) -> Optional[Any]:
        """Load the ONNX white-plane detector with OpenCV DNN."""
        if self.model_path == "" or not os.path.exists(self.model_path):
            print(
                f"Warning: plane detection model not found at {self.model_path}. "
                "Using full image for flat-field correction."
            )
            return None

        try:
            return cv2.dnn.readNetFromONNX(self.model_path)
        except cv2.error as exc:
            print(
                f"Warning: could not load plane detection model at {self.model_path}: {exc}. "
                "Using full image for flat-field correction."
            )
            return None

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Resize image into the model square while preserving aspect ratio."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width = img.shape[:2]
        scale = min(self.dnn_input_size / width, self.dnn_input_size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.dnn_input_size - new_width) // 2
        pad_y = (self.dnn_input_size - new_height) // 2
        right = self.dnn_input_size - new_width - pad_x
        bottom = self.dnn_input_size - new_height - pad_y
        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_x, pad_y

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        img_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse YOLO-style ONNX output into original-image xyxy boxes."""
        predictions = np.squeeze(output)
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 20:
            predictions = predictions.T

        height, width = img_shape[:2]
        boxes_xywh = []
        boxes_xyxy = []
        confidences = []

        for row in predictions:
            if row.shape[0] < 5:
                continue

            score = float(np.max(row[4:]))
            if score < self.conf_threshold:
                continue

            cx, cy, box_w, box_h = row[:4].astype(float)
            x1 = (cx - box_w / 2 - pad_x) / scale
            y1 = (cy - box_h / 2 - pad_y) / scale
            x2 = (cx + box_w / 2 - pad_x) / scale
            y2 = (cy + box_h / 2 - pad_y) / scale

            x1 = int(np.clip(round(x1), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            x2 = int(np.clip(round(x2), 0, width))
            y2 = int(np.clip(round(y2), 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append([x1, y1, x2, y2])
            confidences.append(score)

        if not boxes_xyxy:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(keep) == 0:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        keep_indices = np.asarray(keep).reshape(-1)
        return (
            np.asarray(boxes_xyxy, dtype=int)[keep_indices],
            np.asarray(confidences, dtype=float)[keep_indices],
        )

    def _detect_plane_opencv_dnn(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run plane detection through OpenCV DNN."""
        if self.model is None:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        model_img, scale, pad_x, pad_y = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(
            model_img,
            scalefactor=1 / 255.0,
            size=(self.dnn_input_size, self.dnn_input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()
        return self._parse_yolo_output(output, scale, pad_x, pad_y, img.shape)

    def _normalize_crop_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Clamp a crop rectangle to the image bounds."""
        height, width = self.img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in rect]
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        return x1, y1, x2, y2
    
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
            
            # Vectorised: blur + subsample instead of Python double loop
            kernel = max(1, 2 * h_win + 1)
            Z_blurred = cv2.blur(Z.astype(np.float64) if Z.ndim == 2
                                 else cv2.cvtColor(Z.astype(np.float64), cv2.COLOR_BGR2GRAY)
                                 if Z.ndim == 3 else Z.astype(np.float64),
                                 (kernel, kernel))
            Zf_blurred = cv2.blur(Z_flat.astype(np.float64) if Z_flat.ndim == 2
                                  else cv2.cvtColor(Z_flat.astype(np.float64), cv2.COLOR_BGR2GRAY)
                                  if Z_flat.ndim == 3 else Z_flat.astype(np.float64),
                                  (kernel, kernel))
            rows_idx = np.linspace(0, w - 1, bins).astype(int)
            cols_idx = np.linspace(0, h - 1, bins).astype(int)
            Z_ = Z_blurred[np.ix_(rows_idx, cols_idx)]
            Z_flat_ = Zf_blurred[np.ix_(rows_idx, cols_idx)]
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
        Detect white plane using OpenCV DNN or manual ROI selection.
        
        Sets self.crop_rect and self.img_cropped.
        """
        if self.img is None:
            raise ValueError("Image is required for flat-field correction.")

        if self.crop_rect is not None:
            x1, y1, x2, y2 = self._normalize_crop_rect(tuple(self.crop_rect))
        elif not self.manual_crop:
            # OpenCV DNN detection
            sr = 0.95  # Shrink ratio to avoid edges
            boxes, probs = self._detect_plane_opencv_dnn(self.img)

            if len(boxes) == 0:
                print("Warning: no white plane detected, using full image")
                x1 = 0
                y1 = 0
                x2 = self.img.shape[1]
                y2 = self.img.shape[0]
            else:
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
                    print(f"Detected white plane: BB={selected_box}, prob={selected_prob:.3f}")

                x1, y1, x2, y2 = selected_box
                margin_x = int(round((1 - sr) * (x2 - x1) / 2))
                margin_y = int(round((1 - sr) * (y2 - y1) / 2))
                x1 = int(x1 + margin_x)
                y1 = int(y1 + margin_y)
                x2 = int(x2 - margin_x)
                y2 = int(y2 - margin_y)
                x1, y1, x2, y2 = self._normalize_crop_rect((x1, y1, x2, y2))
            
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
        interactions = kwargs.get("interactions", False)
        if interactions:
            poly = PolynomialFeatures(degree=degree, interaction_only=False)
        else:
            poly = _PowersOnlyFeatures(degree=degree)
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
                random_seed: Random seed
                
        Returns:
            Fitted sklearn model
        """
        method = kwargs.get("fit_method", "nn")
        max_iter = kwargs.get("max_iter", 1000)
        tol = kwargs.get("tol", 1e-8)
        verbose = kwargs.get("verbose", False)
        rand_seed = kwargs.get("random_seed", kwargs.get("rand_seed", 0))
        
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
                random_seed: Random seed (default: 0)
                
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
            x1, y1, x2, y2 = 0, 0, self.img.shape[1], self.img.shape[0]
        else:
            x1, y1, x2, y2 = self.crop_rect
        
        # Sample multiplier at bins locations within cropped region
        x = np.linspace(x1, x2 - 1, self.bins)
        y = np.linspace(y1, y2 - 1, self.bins)
        X, Y = np.meshgrid(x, y)
        
        h_win = int((self.smooth_window - 1) / 2)
        
        # Vectorised sampling: cv2.blur + subsample instead of Python double loop
        kernel_size = max(1, 2 * h_win + 1)
        blurred_mult = cv2.blur(
            self.cropped_multiplier.astype(np.float64),
            (kernel_size, kernel_size),
        )
        rows_idx = np.linspace(0, L_cropped.shape[0] - 1, self.bins).astype(int)
        cols_idx = np.linspace(0, L_cropped.shape[1] - 1, self.bins).astype(int)
        Z_m = blurred_mult[np.ix_(rows_idx, cols_idx)]
        
        # Flatten coordinates and values
        x_flat, y_flat = X.flatten(), Y.flatten()
        z_flat = Z_m.flatten()
        
        # Normalize to [0, 1] for stable fitting
        min_x, max_x = 0, L_full.shape[1]
        min_y, max_y = 0, L_full.shape[0]
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
        x_full = np.linspace(0, L_full.shape[1] - 1, self.bins)
        y_full = np.linspace(0, L_full.shape[0] - 1, self.bins)
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
        
        # Multiply L channel by final_multiplier (clip to avoid uint8 wrap-around)
        L_ = np.clip(L.astype(FLOAT) * self.final_multiplier, 0, 255).astype(UINT8)
        
        # Reconstruct image
        if self.check_color(img):
            img_LAB[:, :, 0] = L_
            img_corrected = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
        else:
            img_corrected = L_
        
        if show:
            self.show_results(img_corrected, img_orig)
        
        return np.clip(img_corrected, 0, 255)
