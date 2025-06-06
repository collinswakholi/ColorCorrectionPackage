"""
Core color-correction pipeline class. Performs (optionally):
1. Flat-Field Correction (FFC)
2. Saturation check / extrapolation
3. Gamma Correction (GC)
4. White Balance (WB)
5. Color Correction (CC)

You can train (run) on a pair of (Image, White_Image) and then use the saved models to predict on new images.
"""

import os
import time
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import colour
from typing import Any, Dict, Optional, Tuple, Union

from .models import MyModels
from .utils.logger_ import log_
from .key_functions import (
    to_uint8,
    to_float64,
    extrapolate_if_sat_image,
    extract_neutral_patches,
    get_metrics,
    arrange_metrics,
    estimate_gamma_profile,
    wb_correction,
    color_correction,
    color_correction_1,
    predict_image,
    predict_,
    get_attr,
    adapt_chart,
)
from .FFC.FF_correction import FlatFieldCorrection


gc.enable()


class Config:
    """
    Simple configuration container for the pipeline steps.
    Any attribute not set will default to None (or be read via get_attr).
    """
    def __init__(
        self,
        do_ffc: bool = True,
        do_gc: bool = True,
        do_wb: bool = True,
        do_cc: bool = True,
        save: bool = False,
        check_saturation : bool = True,
        save_path: Optional[str] = None,
        REF_ILLUMINANT: Optional[np.ndarray] = None,
        FFC_kwargs: Optional[Any] = None,
        GC_kwargs: Optional[Any] = None,
        WB_kwargs: Optional[Any] = None,
        CC_kwargs: Optional[Any] = None,
    ) -> None:
        self.do_ffc = do_ffc
        self.do_gc = do_gc
        self.do_wb = do_wb
        self.do_cc = do_cc
        self.save = save
        self.save_path = save_path
        self.REF_ILLUMINANT = REF_ILLUMINANT
        self.FFC_kwargs = FFC_kwargs
        self.GC_kwargs = GC_kwargs
        self.WB_kwargs = WB_kwargs
        self.CC_kwargs = CC_kwargs
        self.check_saturation = check_saturation


class ColorCorrection:
    """
    Main pipeline for color correction.

    Attributes:
        models: MyModels
        REFERENCE_CHART, REF_ILLUMINANT, REFERENCE_RGB_PD, REFERENCE_NEUTRAL_PATCHES_PD
            are set once `get_reference_values()` is called.
    """

    def __init__(self) -> None:
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
        self,
        REF_ILLUMINANT: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Precompute reference patch RGB under a given illuminant.
        Stores:
          - self.REFERENCE_CHART
          - self.REF_ILLUMINANT
          - self.REFERENCE_RGB_PD   (DataFrame with all 24 patches)
          - self.REFERENCE_NEUTRAL_PATCHES_PD (DataFrame with last 6 neutral patches)
        Returns:
            REFERENCE_RGB_PD (DataFrame of shape [24, 3])
        """
        if REF_ILLUMINANT is None:
            # default to D65 if not provided
            REF_ILLUMINANT = colour.CCS_ILLUMINANTS[
                "CIE 1931 2 Degree Standard Observer"]["D65"]
        self.REF_ILLUMINANT = REF_ILLUMINANT

        # Load the standard ColorChecker 24 chart data
        REFERENCE_CHART = colour.CCS_COLOURCHECKERS[
            "ColorChecker24 - After November 2014"
        ]
        # Chromatic adaptation if needed
        REFERENCE_CHART = adapt_chart(REFERENCE_CHART, REF_ILLUMINANT)

        data_xyY = list(REFERENCE_CHART.data.values())
        names = list(REFERENCE_CHART.data.keys())

        xyz = colour.xyY_to_XYZ(data_xyY)
        rgb = colour.XYZ_to_sRGB(
            xyz, illuminant=REF_ILLUMINANT, apply_cctf_encoding=True
        )
        rgb_clipped = np.clip(rgb, 0.0, 1.0)

        REFERENCE_RGB_PD = pd.DataFrame(
            rgb_clipped, columns=["R", "G", "B"], index=names
        )
        # last six are neutrals
        REFERENCE_NEUTRAL_PATCHES_PD = pd.DataFrame(
            REFERENCE_RGB_PD.iloc[-6:].values,
            columns=["R", "G", "B"],
            index=names[-6:]
        )

        self.REFERENCE_CHART = REFERENCE_CHART
        self.REFERENCE_RGB_PD = REFERENCE_RGB_PD
        self.REFERENCE_NEUTRAL_PATCHES_PD = REFERENCE_NEUTRAL_PATCHES_PD

        return REFERENCE_RGB_PD

    def do_flat_field_correction(
        self,
        Image: np.ndarray,
        do_ffc: bool = True,
        ffc_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform flat-field correction using `FlatFieldCorrection` class.
        Returns:
            corrected_image (RGB float64 0–1),
            Metrics (dict) if `get_deltaE` requested,
            error_flag (True if exception was raised)
        """
        if not do_ffc or self.White_Image is None:
            # LOGGER.warning("Skipping flat field correction (disabled or no White_Image).")
            log_(f"Skipping flat field correction (disabled or no White_Image).", 'yellow', 'italic', 'warn')
            return Image, None, False

        try:
            # Convert our working image to uint8 BGR for FFC
            img_bgr8 = to_uint8(Image[:, :, ::-1])
            assert self.White_Image.shape == img_bgr8.shape, \
                "Image and white image must have same shape."

            # Extract keyword arguments
            get_deltaE = get_attr(ffc_kwargs, "get_deltaE", True)
            ffc_params = {
                "model_path": get_attr(ffc_kwargs, "model_path", ""),
                "manual_crop": get_attr(
                    ffc_kwargs,
                    "manual_crop",
                    True if get_attr(ffc_kwargs, "model_path", "") == "" else False
                ),
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

            # Apply the multiplier
            c_bgr = ffc.apply_ffc(
                img_bgr8, multiplier, show=get_attr(ffc_kwargs, "show", False)
            )
            c_rgb_f64 = to_float64(c_bgr[:, :, ::-1])  # back to RGB float64

            metrics: Dict[str, Any] = {}
            if get_deltaE:
                # Compute deltaE before/after on neutral patches
                ref_vals = self.REFERENCE_RGB_PD.values
                illum = self.REF_ILLUMINANT

                _, cps_before = extract_neutral_patches(img_bgr8, return_one=True)
                _, cps_after = extract_neutral_patches(c_bgr, return_one=True)

                metrics_before = get_metrics(
                    ref_vals, cps_before.values, illum, "srgb"
                )
                metrics_after = get_metrics(
                    ref_vals, cps_after.values, illum, "srgb"
                )
                metrics = arrange_metrics(metrics_before, metrics_after, name="FFC")

            self.models.model_ffc = multiplier
            return c_rgb_f64, metrics, False

        except Exception as e:
            # LOGGER.error(f"FlatFieldCorrection error: {e}", exc_info=True)
            log_(f"FlatFieldCorrection error: {e}", 'red', 'italic', 'error')
            return Image, None, True

    def _check_saturation(
        self,
        Image: np.ndarray,
        do_check: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect and extrapolate saturated patches if needed.
        Returns:
            possibly_corrected_image, saturation_values, saturation_patch_ids
        """
        if not do_check:
            # LOGGER.warning("Skipping saturation check")
            log_(f"Skipping saturation check", 'yellow', 'italic', 'warn')
            return Image, None, None

        try:
            img_out, values, ids = extrapolate_if_sat_image(
                Image, self.REFERENCE_RGB_PD.values
            )
            return img_out, values, ids
        except Exception as e:
            # LOGGER.error(f"Saturation check error: {e}", exc_info=True)
            log_(f"Saturation check error: {e}", 'red', 'italic', 'error')
            return Image, None, None

    def do_gamma_correction(
        self,
        Image: np.ndarray,
        do_gc: bool = True,
        gc_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Estimate gamma correction profile and apply it.
        Returns:
            corrected_image, metrics (if requested), error_flag
        """
        if not do_gc:
            # LOGGER.warning("Skipping gamma correction")
            log_(f"Skipping gamma correction", 'yellow', 'italic', 'warn')
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
            return np.clip(img_gc, 0.0, 1.0), metrics_gc, False
        except Exception as e:
            # LOGGER.error(f"Gamma correction error: {e}", exc_info=True)
            log_(f"Gamma correction error: {e}", 'red', 'italic', 'error')
            return Image, None, True

    def do_white_balance(
        self,
        Image: np.ndarray,
        do_wb: bool = True,
        wb_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform diagonal white-balance correction.
        Returns:
            corrected_image, metrics (if requested), error_flag
        """
        if not do_wb:
            # LOGGER.warning("Skipping white balance")
            log_(f"Skipping white balance", 'yellow', 'italic', 'warn')
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
            return np.clip(img_wb, 0.0, 1.0), metrics_wb, False
        except Exception as e:
            # LOGGER.error(f"White balance error: {e}", exc_info=True)
            log_(f"White balance error: {e}", 'red', 'italic', 'error')
            return Image, None, True

    def do_color_correction(
        self,
        Image: np.ndarray,
        do_cc: bool = True,
        cc_method: str = "ours",
        cc_kwargs: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
        """
        Perform color correction. Supports two methods:
          - 'conv' (conventional Finlayson 2015)
          - 'ours' (your custom model)
        Returns:
            corrected_image, metrics (if requested), error_flag
        """
        if not do_cc:
            # LOGGER.warning("Skipping color correction")
            log_(f"Skipping color correction", 'yellow', 'italic', 'warn')
            return Image, None, False

        try:
            if cc_method.lower() == "conv":
                params = {
                    "method": get_attr(cc_kwargs, "method", "Finlayson 2015"),
                    "degree": get_attr(cc_kwargs, "degree", 3),
                    "root_polynomial_expansion":  None,
                    "terms": get_attr(cc_kwargs, "terms", None),
                }
                # LOGGER.info(f"Using conventional CC method: {params['method']}")
                log_(f"Using conventional CC method: {params['method']}", 'yellow', 'italic', 'info')
                ccm, img_cc, corrected_card, metrics_cc = color_correction_1(
                    img_rgb=Image,
                    ref_rgb=self.REFERENCE_RGB_PD.values,
                    ref_illuminant=self.REF_ILLUMINANT,
                    show=get_attr(cc_kwargs, "show", False),
                    get_deltaE=get_attr(cc_kwargs, "get_deltaE", True),
                    cc_kwargs=params,
                )
                if get_attr(cc_kwargs, "show", False):
                    colour.plotting.plot_multi_colour_checkers([
                        self.REFERENCE_CHART, corrected_card
                    ])
                self.models.model_cc = (ccm, params, "conv")
                return np.clip(img_cc, 0.0, 1.0), metrics_cc, False

            elif cc_method.lower() == "ours":
                params = {
                    "mtd": get_attr(cc_kwargs, "mtd", "linear"),
                    "degree": get_attr(cc_kwargs, "degree", 3),
                    "max_iterations": get_attr(cc_kwargs, "max_iterations", 1000),
                    "nlayers": get_attr(cc_kwargs, "nlayers", 100),
                    "ncomp": get_attr(cc_kwargs, "ncomp", -1),
                    "tol": get_attr(cc_kwargs, "tol", 1e-8),
                    "random_state": get_attr(cc_kwargs, "random_state", 0),
                    "verbose": get_attr(cc_kwargs, "verbose", False),
                    "param_search": get_attr(cc_kwargs, "param_search", False),
                    "hidden_layers": get_attr(cc_kwargs, "hidden_layers", [64, 32, 16]),
                    "learning_rate": get_attr(cc_kwargs, "learning_rate", 0.001),
                    "batch_size": get_attr(cc_kwargs, "batch_size", 32),
                    "patience": get_attr(cc_kwargs, "patience", 10),
                    "dropout_rate": get_attr(cc_kwargs, "dropout_rate", 0.1),
                    "use_batch_norm": get_attr(cc_kwargs, "use_batch_norm", False),
                    "optim_type": get_attr(cc_kwargs, "optim_type", "Adam"),
                }
                # LOGGER.info(f"Using custom CC method: {params['mtd']}")
                log_(f"Using custom CC method: {params['mtd']}", 'yellow', 'italic', 'info')
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
                    colour.plotting.plot_multi_colour_checkers([
                        self.REFERENCE_CHART, corrected_card
                    ])
                self.models.model_cc = (model, params, "ours")
                return np.clip(img_cc, 0.0, 1.0), metrics_cc, False

            else:
                msg = f"Invalid color‐correction method: {cc_method}. Use 'conv' or 'ours'."
                # LOGGER.error(msg)
                log_(msg, 'red', 'italic', 'error')
                return Image, None, False

        except Exception as e:
            # LOGGER.error(f"Color correction error: {e}", exc_info=True)
            log_(f"Color correction error: {e}", 'red', 'italic', 'error')
            return Image, None, True

    def run(
        self,
        Image: Union[str, np.ndarray],
        # White_Image: Union[str, np.ndarray], # make it optional
        White_Image: Optional[Union[str, np.ndarray]] = None,
        name_: str = "",
        config: Optional[Config] = None
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], bool]:
        """
        Execute the full pipeline on a single (Image, White_Image) pair.
        - Loads from disk (if str) or uses the ndarray directly.
        - Runs: FFC → saturation check → GC → WB → CC
        - Collects metrics at each step (if requested) and optionally saves them.
        Returns:
            ALL_METRICS: dict of metrics for each step
            IMAGES: dict of intermediate images (keys: '<name>_FFC', '<name>_GC', etc.)
            Error: True if any step raised an exception
        """

        # LOGGER.info("Initializing ColorCorrection pipeline")
        log_("Initializing ColorCorrection pipeline", 'light_blue', 'italic', 'info')
        # Load image data
        if isinstance(Image, str):
            img_bgr = cv2.imread(Image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read Image from '{Image}'")
            self.Image = to_float64(img_bgr[:, :, ::-1])  # convert BGR→RGB float64
        elif isinstance(Image, np.ndarray):
            self.Image = Image
        else:
            raise TypeError("Image must be a file path or numpy array")

        if isinstance(White_Image, str):
            w_bgr = cv2.imread(White_Image)
            if w_bgr is None:
                log_(f"Cannot read White_Image from '{White_Image}'", 'red', 'italic', 'error')
                log_(f"Skipping flat field correction (disabled or no White_Image).", 'yellow', 'italic', 'warn')
                self.White_Image = None
                config.do_ffc = False
            self.White_Image = w_bgr
        elif isinstance(White_Image, np.ndarray):
            self.White_Image = White_Image
        else:
            log_(f"White_Image is defined by user", 'yellow', 'italic', 'info')
            log_(f"Skipping flat field correction (disabled or no White_Image).", 'yellow', 'italic', 'warn')
            self.White_Image = None
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

        # Step 0: references
        self.get_reference_values(get_attr(config, "REF_ILLUMINANT", None))

        ALL_METRICS: Dict[str, Any] = {}
        IMAGES: Dict[str, np.ndarray] = {}

        # 1. Flat Field Correction
        log_("1. Flat Field Correction", 'cyan', 'normal', 'info')
        t0 = time.time()
        img_ffc, metrics_ffc, err_ffc = self.do_flat_field_correction(
            Image=self.Image, do_ffc=do_ffc, ffc_kwargs=ffc_kwargs
        )
        t1 = time.time()
        if do_ffc:
            # LOGGER.info(f"Flat Field done in {(t1 - t0):.2f}s")
            log_(f"Flat Field done in {(t1 - t0):.2f}s", 'cyan', 'italic', 'info')
            ALL_METRICS[f"{name_}_FFC"] = metrics_ffc
            IMAGES[f"{name_}_FFC"] = img_ffc
        else:
            img_ffc = self.Image

        # 2. Saturation check/extrapolation
        # LOGGER.info("2. Saturation Check")
        log_("2. Saturation Check", 'cyan', 'normal', 'info')
        t0 = time.time()
        if check_sat:
            img_sat, values_sat, ids_sat = self._check_saturation(
                Image=img_ffc, do_check=check_sat
            )
            if ids_sat is not None and save_results and save_path is not None:
                # Save saturation data
                sat_df = pd.DataFrame({
                    "Image": [name_] * len(ids_sat),
                    "ID": ids_sat,
                    "Value_R": values_sat[:, 0],
                    "Value_G": values_sat[:, 1],
                    "Value_B": values_sat[:, 2],
                })
                os.makedirs(save_path, exist_ok=True)
                sat_df.to_csv(
                    os.path.join(save_path, f"{name_}_Sat_data.csv"),
                    float_format="%.9f",
                    encoding="utf-8-sig"
                )
        else:
            img_sat = img_ffc
        t1 = time.time()
        if check_sat:
            # LOGGER.info(f"Saturation check done in {(t1 - t0):.2f}s")
            log_(f"Saturation check done in {(t1 - t0):.2f}s", 'cyan', 'italic', 'info')

        # 3. Gamma Correction
        # LOGGER.info("3. Gamma Correction")
        log_("3. Gamma Correction", 'cyan', 'normal', 'info')
        t0 = time.time()
        img_gc, metrics_gc, err_gc = self.do_gamma_correction(
            Image=img_sat, do_gc=do_gc, gc_kwargs=gc_kwargs
        )
        t1 = time.time()
        if do_gc:
            # LOGGER.info(f"Gamma correction done in {(t1 - t0):.2f}s")
            log_(f"Gamma correction done in {(t1 - t0):.2f}s", 'cyan', 'italic', 'info')
            ALL_METRICS[f"{name_}_GC"] = metrics_gc
            IMAGES[f"{name_}_GC"] = img_gc
        else:
            img_gc = img_sat

        # 4. White Balance
        # LOGGER.info("4. White Balance")
        log_("4. White Balance", 'cyan', 'normal', 'info')
        t0 = time.time()
        img_wb, metrics_wb, err_wb = self.do_white_balance(
            Image=img_gc, do_wb=do_wb, wb_kwargs=wb_kwargs
        )
        t1 = time.time()
        if do_wb:
            # LOGGER.info(f"White balance done in {(t1 - t0):.2f}s")
            log_(f"White balance done in {(t1 - t0):.2f}s", 'cyan', 'italic', 'info')
            ALL_METRICS[f"{name_}_WB"] = metrics_wb
            IMAGES[f"{name_}_WB"] = img_wb
        else:
            img_wb = img_gc

        # 5. Color Correction
        # LOGGER.info("5. Color Correction")
        log_("5. Color Correction", 'cyan', 'normal', 'info')
        t0 = time.time()
        img_cc, metrics_cc, err_cc = self.do_color_correction(
            Image=img_wb,
            do_cc=do_cc,
            cc_method=get_attr(config, "cc_method", "ours"),
            cc_kwargs=cc_kwargs
        )
        t1 = time.time()
        if do_cc:
            # LOGGER.info(f"Color correction done in {(t1 - t0):.2f}s")
            log_(f"Color correction done in {(t1 - t0):.2f}s", 'cyan', 'italic', 'info')
            ALL_METRICS[f"{name_}_CC"] = metrics_cc
            IMAGES[f"{name_}_CC"] = img_cc

        # Save models and metrics if requested
        if save_results and save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            # 1) Save models
            self.models.save(save_path, name_)
            # LOGGER.info(f"Saved models to {save_path}/{name_}.pkl")
            log_(f"Saved models to {save_path}/{name_}.pkl", 'light_green', 'italic', 'info')

            # 2) Concatenate metrics into DataFrames
            if ALL_METRICS:
                row_names = list(self.REFERENCE_CHART.data.keys())
                all_metrics_dfs = []
                summary_metrics = []
                for step_name, mm in ALL_METRICS.items():
                    if mm and "All" in mm and "Summary" in mm:
                        all_metrics_dfs.append(mm["All"])
                        summary_metrics.append(mm["Summary"])
                if all_metrics_dfs:
                    METRICS_DF = pd.concat(all_metrics_dfs, axis=1)
                    METRICS_DF.index = row_names
                    METRICS_DF.to_csv(
                        os.path.join(save_path, f"{name_}_All_Metrics.csv"),
                        float_format="%.9f",
                        encoding="utf-8",
                    )
                if summary_metrics:
                    SUMMARY_DF = pd.concat(summary_metrics, axis=0)
                    SUMMARY_DF.to_csv(
                        os.path.join(save_path, f"{name_}_Summary_Metrics.csv"),
                        float_format="%.9f",
                        encoding="utf-8",
                    )
                # LOGGER.info(f"Saved metrics CSVs to {save_path}")
                log_(f"Saved metrics CSVs to {save_path}", 'light_green', 'italic', 'info')

        # Free CUDA memory if used
        torch.cuda.empty_cache()
        gc.collect()

        Error = any([err_ffc, err_gc, err_wb, err_cc])
        return ALL_METRICS, IMAGES, Error

    def predict_image(
        self,
        Image: Union[str, np.ndarray],
        show: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Given a new image (path or ndarray), apply saved models (ffc, gc, wb, cc)
        in sequence and return a dict of partial results.
        Keys: 'FFC', 'GC', 'WB', 'CC'.
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
            ffc_out_bgr = ffc_obj.apply_ffc(
                img=bgr8, multiplier=self.models.model_ffc
            )
            img_ffc = to_float64(ffc_out_bgr[:, :, ::-1])
        else:
            img_ffc = img
        out_images["FFC"] = img_ffc

        # 2) Gamma correction
        if self.models.model_gc is not None:
            gc_out = predict_image(
                img=img_ffc, coeffs=self.models.model_gc,
                ref_illuminant=self.REF_ILLUMINANT
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
            if img_cc is not None:
                img_cc = np.clip(img_cc, 0.0, 1.0)
        out_images["CC"] = img_cc

        end = time.time()
        # LOGGER.info(f"Prediction elapsed: {(end - start):.2f}s")
        log_(f"Prediction elapsed: {(end - start):.2f}s", 'light_green', 'italic', 'info')

        if show:
            for name, im in out_images.items():
                if im is not None:
                    try:
                        colour.plotting.plot_image(im, title=name)
                    except Exception:
                        pass

        return out_images
