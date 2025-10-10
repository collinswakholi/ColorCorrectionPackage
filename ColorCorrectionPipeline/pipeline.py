"""
Main color correction pipeline
===============================

This module provides the ColorCorrection class that orchestrates the complete
color correction workflow:
    1. Flat-field correction (FFC)
    2. Saturation check/extrapolation
    3. Gamma correction (GC)
    4. White balance (WB)
    5. Color correction (CC)

The pipeline can train on an image with color chart and then apply learned
corrections to new images.

Example:
    >>> from color_correc_optim import ColorCorrection, Config
    >>> import cv2
    >>> 
    >>> # Load images
    >>> img_bgr = cv2.imread("image_with_chart.jpg")
    >>> white_bgr = cv2.imread("white_background.jpg")
    >>> 
    >>> # Configure pipeline
    >>> config = Config(
    ...     do_ffc=True,
    ...     do_gc=True,
    ...     do_wb=True,
    ...     do_cc=True,
    ...     FFC_kwargs={"bins": 50, "show": False},
    ...     GC_kwargs={"max_degree": 5},
    ...     WB_kwargs={"get_deltaE": True},
    ...     CC_kwargs={"mtd": "nn", "degree": 2}
    ... )
    >>> 
    >>> # Run pipeline
    >>> cc = ColorCorrection()
    >>> metrics, images, error = cc.run(img_bgr, white_bgr, "test", config)
    >>> 
    >>> # Predict on new image
    >>> test_img = cv2.imread("new_image.jpg")
    >>> results = cc.predict_image(test_img, show=True)
"""

import gc
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import colour
import cv2
import numpy as np
import pandas as pd

from .config import Config
from .constants import MODEL_PATH, WP_DEFAULT
from .core import (
    adapt_chart,
    arrange_metrics,
    color_correction,
    color_correction_1,
    estimate_gamma_profile,
    extract_neutral_patches,
    extrapolate_if_sat_image,
    get_metrics,
    predict_,
    predict_image,
    to_float64,
    to_uint8,
    wb_correction,
)
from .flat_field import FlatFieldCorrection
from .models import MyModels

__all__ = ["ColorCorrection"]


def get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object or dict with default value."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


class ColorCorrection:
    """
    Main pipeline for color correction.
    
    This class provides a complete color correction workflow that can:
    - Train on images with color charts to learn correction parameters
    - Save trained models for later use
    - Apply learned corrections to new images
    
    The pipeline stages are:
    1. Flat-field correction (FFC) - Compensate for uneven illumination
    2. Saturation check - Detect and extrapolate saturated color patches
    3. Gamma correction (GC) - Adjust tonal response curve
    4. White balance (WB) - Correct color cast
    5. Color correction (CC) - Match colors to reference chart
    
    Attributes:
        Image: Current working image (RGB, float64, 0-1)
        White_Image: White background image for FFC (BGR, uint8)
        models: MyModels instance storing trained models
        Models_path: Path to saved models directory
        REFERENCE_CHART: ColorChecker reference chart
        REF_ILLUMINANT: Reference illuminant (xy coordinates)
        REFERENCE_RGB_PD: Reference patch RGB values (DataFrame)
        REFERENCE_NEUTRAL_PATCHES_PD: Reference neutral patches (DataFrame)
        
    Example:
        >>> cc = ColorCorrection()
        >>> config = Config(do_ffc=True, do_gc=True, do_wb=True, do_cc=True)
        >>> metrics, images, error = cc.run(img, white_img, "sample", config)
        >>> test_results = cc.predict_image(test_img)
    """
    
    def __init__(self) -> None:
        """Initialize ColorCorrection pipeline."""
        self.Image: Optional[np.ndarray] = None
        self.White_Image: Optional[np.ndarray] = None
        self.models = MyModels()
        self.Models_path: Optional[str] = None
        
        # Reference data placeholders (populated by get_reference_values)
        self.REFERENCE_CHART: Any = None
        self.REF_ILLUMINANT: Any = None
        self.REFERENCE_RGB_PD: Optional[pd.DataFrame] = None
        self.REFERENCE_NEUTRAL_PATCHES_PD: Optional[pd.DataFrame] = None
    
    def get_reference_values(
        self, REF_ILLUMINANT: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Precompute reference patch RGB under a given illuminant.
        
        Loads the standard ColorChecker 24 chart and adapts it to the specified
        illuminant. Stores reference data in instance attributes.
        
        Args:
            REF_ILLUMINANT: Reference illuminant (xy coordinates).
                If None, defaults to D65.
                
        Returns:
            pd.DataFrame: Reference RGB values (24, 3) with patch names as index
            
        Sets:
            self.REFERENCE_CHART: ColorChecker object
            self.REF_ILLUMINANT: Reference illuminant
            self.REFERENCE_RGB_PD: DataFrame of all 24 patches
            self.REFERENCE_NEUTRAL_PATCHES_PD: DataFrame of last 6 neutral patches
            
        Example:
            >>> cc = ColorCorrection()
            >>> ref_rgb = cc.get_reference_values()
            >>> print(ref_rgb.shape)  # (24, 3)
        """
        if REF_ILLUMINANT is None:
            # Default to D65
            REF_ILLUMINANT = WP_DEFAULT
        
        self.REF_ILLUMINANT = REF_ILLUMINANT
        
        # Load standard ColorChecker 24 chart
        REFERENCE_CHART = colour.CCS_COLOURCHECKERS[
            "ColorChecker24 - After November 2014"
        ]
        
        # Chromatic adaptation if needed
        REFERENCE_CHART = adapt_chart(REFERENCE_CHART, REF_ILLUMINANT)
        
        data_xyY = list(REFERENCE_CHART.data.values())
        names = [str(k) for k in REFERENCE_CHART.data.keys()]  # Convert to strings
        
        xyz = colour.xyY_to_XYZ(data_xyY)
        rgb = colour.XYZ_to_sRGB(
            xyz, illuminant=REF_ILLUMINANT, apply_cctf_encoding=True
        )
        rgb_clipped = np.clip(rgb, 0.0, 1.0)
        
        REFERENCE_RGB_PD = pd.DataFrame(
            rgb_clipped, columns=["R", "G", "B"], index=names
        )
        
        # Last six are neutrals
        REFERENCE_NEUTRAL_PATCHES_PD = pd.DataFrame(
            rgb_clipped[-6:], columns=["R", "G", "B"], index=names[-6:]
        )
        
        self.REFERENCE_CHART = REFERENCE_CHART
        self.REFERENCE_RGB_PD = REFERENCE_RGB_PD
        self.REFERENCE_NEUTRAL_PATCHES_PD = REFERENCE_NEUTRAL_PATCHES_PD
        
        return REFERENCE_RGB_PD
    
    def do_flat_field_correction(
        self, Image: np.ndarray, do_ffc: bool = True, ffc_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform flat-field correction using FlatFieldCorrection class.
        
        Args:
            Image: RGB image (float64, 0-1)
            do_ffc: Whether to perform FFC
            ffc_kwargs: FFC parameters dict
            
        Returns:
            tuple: (corrected_image, metrics, error_flag)
                - corrected_image: RGB float64 0-1
                - metrics: Dict of metrics if get_deltaE requested
                - error_flag: True if exception raised
                
        Example:
            >>> ffc_kwargs = {"bins": 50, "degree": 5, "fit_method": "nn"}
            >>> img_ffc, metrics, err = cc.do_flat_field_correction(img, True, ffc_kwargs)
        """
        if not do_ffc or self.White_Image is None:
            print("Warning: Skipping flat field correction (disabled or no White_Image)")
            return Image, None, False
        
        try:
            # Convert to uint8 BGR for FFC
            img_bgr8 = to_uint8(Image[:, :, ::-1])
            assert (
                self.White_Image.shape == img_bgr8.shape
            ), "Image and white image must have same shape."
            
            # Extract parameters
            get_deltaE = get_attr(ffc_kwargs, "get_deltaE", True)
            ffc_params = {
                "model_path": get_attr(ffc_kwargs, "model_path", MODEL_PATH),
                "manual_crop": get_attr(ffc_kwargs, "manual_crop", False),
                "show": get_attr(ffc_kwargs, "show", False),
                "bins": get_attr(ffc_kwargs, "bins", 50),
                "smooth_window": get_attr(ffc_kwargs, "smooth_window", 5),
                "crop_rect": get_attr(ffc_kwargs, "crop_rect", None),
            }
            fit_params = {
                "degree": get_attr(ffc_kwargs, "degree", 3),
                "interactions": get_attr(ffc_kwargs, "interactions", False),
                "fit_method": get_attr(ffc_kwargs, "fit_method", "linear"),
                "max_iter": get_attr(ffc_kwargs, "max_iter", 1000),
                "tol": get_attr(ffc_kwargs, "tol", 1e-8),
                "verbose": get_attr(ffc_kwargs, "verbose", False),
                "random_seed": get_attr(ffc_kwargs, "random_seed", 0),
            }
            
            ffc = FlatFieldCorrection(self.White_Image, **ffc_params)
            multiplier = ffc.compute_multiplier(**fit_params)
            
            # Apply multiplier
            c_bgr = ffc.apply_ffc(
                img_bgr8, multiplier, show=get_attr(ffc_kwargs, "show", False)
            )
            c_rgb_f64 = to_float64(c_bgr[:, :, ::-1])  # Back to RGB float64
            
            metrics: Dict[str, Any] = {}
            if get_deltaE:
                # Compute deltaE before/after on neutral patches
                ref_vals = self.REFERENCE_RGB_PD.values
                illum = self.REF_ILLUMINANT
                
                _, cps_before = extract_neutral_patches(img_bgr8, return_one=True)
                _, cps_after = extract_neutral_patches(c_bgr, return_one=True)
                
                if cps_before is not None and cps_after is not None:
                    metrics_before = get_metrics(ref_vals, cps_before.values, illum, "srgb")
                    metrics_after = get_metrics(ref_vals, cps_after.values, illum, "srgb")
                    metrics = arrange_metrics(metrics_before, metrics_after, name="FFC")
            
            self.models.model_ffc = multiplier
            return c_rgb_f64, metrics, False
        
        except Exception as e:
            print(f"Error: FlatFieldCorrection error: {e}")
            return Image, None, True
    
    def _check_saturation(
        self, Image: np.ndarray, do_check: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect and extrapolate saturated patches if needed.
        
        Args:
            Image: RGB image (float64, 0-1)
            do_check: Whether to perform saturation check
            
        Returns:
            tuple: (possibly_corrected_image, saturation_values, saturation_patch_ids)
                - possibly_corrected_image: RGB float64 0-1
                - saturation_values: Values of saturated patches (N, 3) or None
                - saturation_patch_ids: Indices of saturated patches or None
                
        Example:
            >>> img_sat, sat_vals, sat_ids = cc._check_saturation(img, True)
        """
        if not do_check:
            print("Warning: Skipping saturation check")
            return Image, None, None
        
        try:
            img_out, values, ids = extrapolate_if_sat_image(
                Image, self.REFERENCE_RGB_PD.values
            )
            return img_out, values, ids
        except Exception as e:
            print(f"Error: Saturation check error: {e}")
            return Image, None, None
    
    def do_gamma_correction(
        self, Image: np.ndarray, do_gc: bool = True, gc_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Estimate gamma correction profile and apply it.
        
        Args:
            Image: RGB image (float64, 0-1)
            do_gc: Whether to perform gamma correction
            gc_kwargs: Gamma correction parameters dict
            
        Returns:
            tuple: (corrected_image, metrics, error_flag)
                - corrected_image: RGB float64 0-1
                - metrics: Dict of metrics if requested
                - error_flag: True if exception raised
                
        Example:
            >>> gc_kwargs = {"max_degree": 5, "get_deltaE": True}
            >>> img_gc, metrics, err = cc.do_gamma_correction(img, True, gc_kwargs)
        """
        if not do_gc:
            print("Warning: Skipping gamma correction")
            return Image, None, False
        
        try:
            params = {
                "max_degree": get_attr(gc_kwargs, "max_degree", 5),
                "show": get_attr(gc_kwargs, "show", False),
                "get_deltaE": get_attr(gc_kwargs, "get_deltaE", True),
            }
            coeffs_gc, img_gc, metrics_gc = estimate_gamma_profile(
                img_rgb=Image,
                ref_cp=self.REFERENCE_RGB_PD.values,
                ref_illuminant=self.REF_ILLUMINANT,
                **params,
            )
            self.models.model_gc = coeffs_gc
            # CRITICAL FIX: Clip output to [0,1] to prevent out-of-range values from
            # propagating to subsequent stages and causing polynomial extrapolation issues
            return np.clip(img_gc, 0.0, 1.0), metrics_gc, False
        
        except Exception as e:
            print(f"Error: Gamma correction error: {e}")
            return Image, None, True
    
    def do_white_balance(
        self, Image: np.ndarray, do_wb: bool = True, wb_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform white balance correction using diagonal matrix.
        
        Args:
            Image: RGB image (float64, 0-1)
            do_wb: Whether to perform white balance
            wb_kwargs: White balance parameters dict
            
        Returns:
            tuple: (corrected_image, metrics, error_flag)
                - corrected_image: RGB float64 0-1
                - metrics: Dict of metrics if requested
                - error_flag: True if exception raised
                
        Example:
            >>> wb_kwargs = {"get_deltaE": True, "show": False}
            >>> img_wb, metrics, err = cc.do_white_balance(img, True, wb_kwargs)
        """
        if not do_wb:
            print("Warning: Skipping white balance")
            return Image, None, False
        
        try:
            params = {
                "show": get_attr(wb_kwargs, "show", False),
                "get_deltaE": get_attr(wb_kwargs, "get_deltaE", True),
            }
            diag_wb, img_wb, metrics_wb = wb_correction(
                img_rgb=Image,
                ref_cp=self.REFERENCE_RGB_PD.values,
                ref_illuminant=self.REF_ILLUMINANT,
                **params,
            )
            self.models.model_wb = diag_wb
            # CRITICAL FIX: Clip output to [0,1] to prevent out-of-range values from
            # propagating to CC stage and causing polynomial extrapolation issues
            return np.clip(img_wb, 0.0, 1.0), metrics_wb, False
        
        except Exception as e:
            print(f"Error: White balance error: {e}")
            return Image, None, True
    
    def do_color_correction(
        self,
        Image: np.ndarray,
        do_cc: bool = True,
        cc_method: str = "ours",
        cc_kwargs: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform color correction (conventional or custom ML-based).
        
        Args:
            Image: RGB image (float64, 0-1)
            do_cc: Whether to perform color correction
            cc_method: Method ("conv" for conventional, "ours" for custom ML)
            cc_kwargs: Color correction parameters dict
            
        Returns:
            tuple: (corrected_image, metrics, error_flag)
                - corrected_image: RGB float64 0-1
                - metrics: Dict of metrics if requested
                - error_flag: True if exception raised
                
        Example:
            >>> cc_kwargs = {"mtd": "nn", "degree": 2, "get_deltaE": True}
            >>> img_cc, metrics, err = cc.do_color_correction(img, True, "ours", cc_kwargs)
        """
        if not do_cc:
            print("Warning: Skipping color correction")
            return Image, None, False
        
        try:
            if cc_method.lower() == "conv":
                # Conventional method (Finlayson 2015)
                params = {
                    "method": get_attr(cc_kwargs, "method", "Finlayson 2015"),
                    "degree": get_attr(cc_kwargs, "degree", 3),
                    "root_polynomial_expansion": None,
                    "terms": get_attr(cc_kwargs, "terms", None),
                }
                print(f"Info: Using conventional CC method: {params['method']}")
                ccm, img_cc, corrected_card, metrics_cc = color_correction_1(
                    img_rgb=Image,
                    ref_rgb=self.REFERENCE_RGB_PD.values,
                    ref_illuminant=self.REF_ILLUMINANT,
                    show=get_attr(cc_kwargs, "show", False),
                    get_deltaE=get_attr(cc_kwargs, "get_deltaE", True),
                    cc_kwargs=params,
                )
                if get_attr(cc_kwargs, "show", False):
                    colour.plotting.plot_multi_colour_checkers(
                        [self.REFERENCE_CHART, corrected_card]
                    )
                self.models.model_cc = (ccm, params, "conv")
                return np.clip(img_cc, 0.0, 1.0), metrics_cc, False
            
            elif cc_method.lower() == "ours":
                # Custom ML-based method
                params = {
                    "mtd": get_attr(cc_kwargs, "mtd", "linear"),
                    "degree": get_attr(cc_kwargs, "degree", 3),
                    "max_iterations": get_attr(cc_kwargs, "max_iterations", 1000),
                    "nlayers": get_attr(cc_kwargs, "nlayers", 100),
                    "ncomp": get_attr(cc_kwargs, "ncomp", -1),
                    "random_state": get_attr(cc_kwargs, "random_state", 0),
                    "tol": get_attr(cc_kwargs, "tol", 1e-8),
                    "verbose": get_attr(cc_kwargs, "verbose", False),
                    "param_search": get_attr(cc_kwargs, "param_search", False),
                    "hidden_layers": get_attr(cc_kwargs, "hidden_layers", [64, 32, 16]),
                    "learning_rate": get_attr(cc_kwargs, "learning_rate", 0.001),
                    "batch_size": get_attr(cc_kwargs, "batch_size", 16),
                    "patience": get_attr(cc_kwargs, "patience", 10),
                    "dropout_rate": get_attr(cc_kwargs, "dropout_rate", 0.2),
                    "use_batch_norm": get_attr(cc_kwargs, "use_batch_norm", False),
                    "optim_type": get_attr(cc_kwargs, "optim_type", "Adam"),
                }
                print(f"Info: Using custom CC method: {params['mtd']}")
                model, img_cc, corrected_card, metrics_cc = color_correction(
                    img_rgb=Image,
                    ref_rgb=self.REFERENCE_RGB_PD.values,
                    ref_illuminant=self.REF_ILLUMINANT,
                    show=get_attr(cc_kwargs, "show", False),
                    get_deltaE=get_attr(cc_kwargs, "get_deltaE", True),
                    cc_kwargs=params,
                    n_samples=get_attr(cc_kwargs, "n_samples", 50),
                )
                if get_attr(cc_kwargs, "show", False):
                    colour.plotting.plot_multi_colour_checkers(
                        [self.REFERENCE_CHART, corrected_card]
                    )
                self.models.model_cc = (model, params, "ours")
                return np.clip(img_cc, 0.0, 1.0), metrics_cc, False
            
            else:
                msg = f"Invalid color-correction method: {cc_method}. Use 'conv' or 'ours'."
                print(f"Error: {msg}")
                return Image, None, False
        
        except Exception as e:
            print(f"Error: Color correction error: {e}")
            import traceback
            traceback.print_exc()
            return Image, None, True
    
    def run(
        self,
        Image: Union[str, np.ndarray],
        White_Image: Optional[Union[str, np.ndarray]] = None,
        name_: str = "",
        config: Optional[Config] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], bool]:
        """
        Execute the full pipeline on a single (Image, White_Image) pair.
        
        Loads images from disk (if str) or uses ndarray directly.
        Runs: FFC → saturation check → GC → WB → CC
        Collects metrics at each step (if requested) and optionally saves them.
        
        Args:
            Image: Image with color chart
                - str: Path to image file
                - np.ndarray: RGB float64 in [0,1] range
            White_Image: White background image for flat-field correction (optional)
                - str: Path to image file (loaded as BGR uint8 via cv2.imread)
                - np.ndarray: 
                    * Preferred: uint8 BGR (standard cv2.imread format)
                    * Also accepts: float64 RGB in [0,1] (converted to uint8 BGR)
            name_: Identifier for this run (used in metric keys)
            config: Config object with pipeline parameters
            
        Returns:
            tuple: (ALL_METRICS, IMAGES, Error)
                - ALL_METRICS: Dict of metrics for each step
                - IMAGES: Dict of intermediate images (keys: '<name>_FFC', '<name>_GC', etc.)
                - Error: True if any step raised an exception
                
        Note:
            White_Image format matches ColorCorrectionPipeline convention:
            Use cv2.imread() or uint8 BGR array for best compatibility.
                
        Example:
            >>> import cv2
            >>> cc = ColorCorrection()
            >>> config = Config(do_ffc=True, do_gc=True, do_wb=True, do_cc=True)
            >>> img_rgb = read_image("sample.jpg")  # float64 RGB
            >>> white_bgr = cv2.imread("white.jpg")  # uint8 BGR
            >>> metrics, images, error = cc.run(img_rgb, white_bgr, "test", config)
            >>> print(metrics.keys())  # dict_keys(['test_FFC', 'test_GC', 'test_WB', 'test_CC'])
        """
        print("Info: Initializing ColorCorrection pipeline")
        
        # Load image data
        if isinstance(Image, str):
            img_bgr = cv2.imread(Image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read Image from '{Image}'")
            self.Image = to_float64(img_bgr[:, :, ::-1])  # BGR → RGB float64
        elif isinstance(Image, np.ndarray):
            self.Image = Image
        else:
            raise TypeError("Image must be a file path or numpy array")
        
        if isinstance(White_Image, str):
            w_bgr = cv2.imread(White_Image)
            if w_bgr is None:
                print(f"Error: Cannot read White_Image from '{White_Image}'")
                print("Warning: Skipping flat field correction (disabled or no White_Image)")
                self.White_Image = None
                if config is not None:
                    config.do_ffc = False
            else:
                # Ensure uint8 BGR format
                self.White_Image = w_bgr if w_bgr.dtype == np.uint8 else to_uint8(w_bgr)
        elif isinstance(White_Image, np.ndarray):
            # Expected format: uint8 BGR (matching cv2.imread output)
            # For compatibility, also accepts float64 RGB which will be converted
            if White_Image.dtype == np.uint8:
                # Already uint8, assume BGR format (standard cv2.imread format)
                self.White_Image = White_Image
            elif White_Image.dtype == np.float64 or White_Image.dtype == np.float32:
                # Float image: assume RGB in [0,1] range, convert to BGR uint8
                if White_Image.ndim == 3 and White_Image.shape[2] == 3:
                    self.White_Image = to_uint8(White_Image[:, :, ::-1])  # RGB -> BGR
                else:
                    # Grayscale float image
                    self.White_Image = to_uint8(White_Image)
            else:
                # Unexpected dtype, try to convert
                print(f"Warning: Unexpected White_Image dtype {White_Image.dtype}, converting to uint8")
                self.White_Image = to_uint8(White_Image if White_Image.ndim == 2 else White_Image[:, :, ::-1])
        else:
            print("Info: White_Image not provided")
            print("Warning: Skipping flat field correction (disabled or no White_Image)")
            self.White_Image = None
            if config is not None:
                config.do_ffc = False
        
        # Unpack configuration
        config = config or Config()
        do_ffc = get_attr(config, "do_ffc", True)
        do_gc = get_attr(config, "do_gc", True)
        do_wb = get_attr(config, "do_wb", True)
        do_cc = get_attr(config, "do_cc", True)
        check_sat = get_attr(config, "check_saturation", True)
        
        ffc_kwargs = get_attr(config, "FFC_kwargs", {})
        gc_kwargs = get_attr(config, "GC_kwargs", {})
        wb_kwargs = get_attr(config, "WB_kwargs", {})
        cc_kwargs = get_attr(config, "CC_kwargs", {})
        
        save_results = get_attr(config, "save", False)
        save_path = get_attr(config, "save_path", None)
        
        # Step 0: Get reference values
        self.get_reference_values(get_attr(config, "REF_ILLUMINANT", None))
        
        ALL_METRICS: Dict[str, Any] = {}
        IMAGES: Dict[str, np.ndarray] = {}
        
        # 1. Flat Field Correction
        print("Info: 1. Flat Field Correction")
        t0 = time.time()
        img_ffc, metrics_ffc, err_ffc = self.do_flat_field_correction(
            Image=self.Image, do_ffc=do_ffc, ffc_kwargs=ffc_kwargs
        )
        t1 = time.time()
        if do_ffc:
            print(f"Info: Flat Field done in {(t1 - t0):.2f}s")
            ALL_METRICS[f"{name_}_FFC"] = metrics_ffc
            IMAGES[f"{name_}_FFC"] = img_ffc
        else:
            img_ffc = self.Image
        
        # 2. Saturation check/extrapolation
        print("Info: 2. Saturation Check")
        t0 = time.time()
        if check_sat:
            img_sat, values_sat, ids_sat = self._check_saturation(
                Image=img_ffc, do_check=check_sat
            )
            if ids_sat is not None and save_results and save_path is not None:
                # Save saturation data
                sat_df = pd.DataFrame(
                    {
                        "Image": [name_] * len(ids_sat),
                        "ID": ids_sat,
                        "Value_R": values_sat[:, 0],
                        "Value_G": values_sat[:, 1],
                        "Value_B": values_sat[:, 2],
                    }
                )
                os.makedirs(save_path, exist_ok=True)
                sat_df.to_csv(
                    os.path.join(save_path, f"{name_}_Sat_data.csv"),
                    float_format="%.9f",
                    encoding="utf-8-sig",
                )
        else:
            img_sat = img_ffc
        t1 = time.time()
        if check_sat:
            print(f"Info: Saturation check done in {(t1 - t0):.2f}s")
        
        # 3. Gamma Correction
        print("Info: 3. Gamma Correction")
        t0 = time.time()
        img_gc, metrics_gc, err_gc = self.do_gamma_correction(
            Image=img_sat, do_gc=do_gc, gc_kwargs=gc_kwargs
        )
        t1 = time.time()
        if do_gc:
            print(f"Info: Gamma correction done in {(t1 - t0):.2f}s")
            ALL_METRICS[f"{name_}_GC"] = metrics_gc
            IMAGES[f"{name_}_GC"] = img_gc
        else:
            img_gc = img_sat
        
        # 4. White Balance
        print("Info: 4. White Balance")
        t0 = time.time()
        img_wb, metrics_wb, err_wb = self.do_white_balance(
            Image=img_gc, do_wb=do_wb, wb_kwargs=wb_kwargs
        )
        t1 = time.time()
        if do_wb:
            print(f"Info: White balance done in {(t1 - t0):.2f}s")
            ALL_METRICS[f"{name_}_WB"] = metrics_wb
            IMAGES[f"{name_}_WB"] = img_wb
        else:
            img_wb = img_gc
        
        # 5. Color Correction
        print("Info: 5. Color Correction")
        t0 = time.time()
        img_cc, metrics_cc, err_cc = self.do_color_correction(
            Image=img_wb,
            do_cc=do_cc,
            cc_method=get_attr(cc_kwargs, "cc_method", "ours"),
            cc_kwargs=cc_kwargs,
        )
        t1 = time.time()
        if do_cc:
            print(f"Info: Color correction done in {(t1 - t0):.2f}s")
            ALL_METRICS[f"{name_}_CC"] = metrics_cc
            IMAGES[f"{name_}_CC"] = img_cc
        else:
            img_cc = img_wb
        
        # Save models if requested
        if save_results and save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            models_file = os.path.join(save_path, f"{name_}_models.pkl")
            self.models.save(models_file)
            print(f"Info: Models saved to {models_file}")
            
            # Save metrics CSV
            if ALL_METRICS:
                metrics_df = pd.DataFrame.from_dict(ALL_METRICS)
                metrics_csv = os.path.join(save_path, f"{name_}_metrics.csv")
                metrics_df.to_csv(metrics_csv, float_format="%.9f", encoding="utf-8-sig")
                print(f"Info: Metrics saved to {metrics_csv}")
        
        # Check for errors
        Error = err_ffc or err_gc or err_wb or err_cc
        
        gc.collect()
        
        return ALL_METRICS, IMAGES, Error
    
    def predict_image(
        self, Image: Union[str, np.ndarray], show: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Apply saved models to a new image.
        
        Given a new image (path or ndarray), applies saved models (ffc, gc, wb, cc)
        in sequence and returns a dict of partial results.
        
        Args:
            Image: Image to correct (str path or RGB ndarray)
            show: Whether to display results (not implemented)
            
        Returns:
            Dict[str, np.ndarray]: Corrected images at each stage
                Keys: 'FFC', 'GC', 'WB', 'CC'
                Values: RGB float64 0-1 images
                
        Raises:
            FileNotFoundError: If image path cannot be read
            TypeError: If Image is not str or ndarray
            
        Example:
            >>> cc.models.load("saved_models.pkl")  # Load pre-trained models
            >>> results = cc.predict_image("test.jpg", show=True)
            >>> final_img = results['CC']  # Get fully corrected image
        """
        if isinstance(Image, str):
            img_bgr = cv2.imread(Image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read Image from '{Image}'")
            img = to_float64(img_bgr[:, :, ::-1])
        elif isinstance(Image, np.ndarray):
            img = Image
        else:
            raise TypeError("Image must be a file path or numpy array")
        
        start = time.time()
        out_images: Dict[str, np.ndarray] = {}
        
        # 1) Flat Field
        if self.models.model_ffc is not None:
            ffc_obj = FlatFieldCorrection()
            bgr8 = to_uint8(img[:, :, ::-1])
            ffc_out_bgr = ffc_obj.apply_ffc(img=bgr8, multiplier=self.models.model_ffc)
            img_ffc = to_float64(ffc_out_bgr[:, :, ::-1])
        else:
            img_ffc = img
        out_images["FFC"] = img_ffc
        
        # 2) Gamma correction
        if self.models.model_gc is not None:
            gc_out = predict_image(
                img=img_ffc,
                coeffs=self.models.model_gc,
                ref_illuminant=self.REF_ILLUMINANT,
            )
            img_gc = np.clip(gc_out, 0.0, 1.0)
        else:
            img_gc = img_ffc
        out_images["GC"] = img_gc
        
        # 3) White balance
        if self.models.model_wb is not None:
            img_wb = img_gc @ self.models.model_wb
            img_wb = np.clip(img_wb, 0.0, 1.0)
        else:
            img_wb = img_gc
        out_images["WB"] = img_wb
        
        # 4) Color correction
        img_cc: Optional[np.ndarray] = None
        if self.models.model_cc is not None:
            method = self.models.model_cc[2]
            if method == "conv":
                ccm = self.models.model_cc[0]
                cc_params = self.models.model_cc[1]
                img_cc = colour.characterisation.apply_matrix_colour_correction(
                    RGB=img_wb, CCM=ccm, **cc_params
                )
            elif method == "ours":
                model = self.models.model_cc[0]
                img_cc = predict_(RGB=img_wb, M=model)
            img_cc = np.clip(img_cc, 0.0, 1.0)
        else:
            img_cc = img_wb
        out_images["CC"] = img_cc
        
        end = time.time()
        print(f"Info: Prediction done in {(end - start):.2f}s")
        
        gc.collect()
        
        return out_images
