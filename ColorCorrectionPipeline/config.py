"""
Configuration module for color_correc_optim
============================================

Provides configuration classes for the color correction pipeline.
Supports TOML and INI file formats with validation.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .colorimetry import ColorSemantics, D65, white_xy
from .contracts import ReferenceProfile
from .stage_policies import StagePolicy
from .fitting import FittingPolicy

__all__ = ["Config"]


class Config:
    """
    Configuration container for the color correction pipeline.

    This class holds all configuration parameters for each stage of the pipeline:
    flat-field correction (FFC), gamma correction (GC), white balance (WB), and
    color correction (CC).

    Args:
        do_ffc: Enable flat-field correction. Default: True
        do_gc: Enable gamma correction. Default: True
        do_wb: Enable white balance correction. Default: True
        do_cc: Enable color correction. Default: True
        save: Save models and metrics to disk. Default: False
        check_saturation: Check for saturated patches before correction. Default: True
        save_path: Directory path for saving outputs. Default: None
        REF_ILLUMINANT: Reference illuminant as numpy array. Default: None (uses D65)
        FFC_kwargs: Keyword arguments for flat-field correction. Default: None
        GC_kwargs: Keyword arguments for gamma correction. Default: None
        WB_kwargs: Keyword arguments for white balance. Default: None
        CC_kwargs: Keyword arguments for color correction. Default: None

    Example:
        >>> config = Config(
        ...     do_ffc=True,
        ...     do_gc=True,
        ...     do_wb=True,
        ...     do_cc=True,
        ...     save=True,
        ...     save_path="results",
        ...     FFC_kwargs={"bins": 50, "degree": 3},
        ...     CC_kwargs={"mtd": "linear", "degree": 2}
        ... )
    
    Notes:
        - Any attribute not explicitly set defaults to None
        - Attributes are accessed via dot notation or get_attr utility
        - Supports serialization to/from TOML and INI formats
    """

    def __init__(
        self,
        do_ffc: bool = True,
        do_gc: bool = True,
        do_wb: bool = True,
        do_cc: bool = True,
        save: bool = False,
        check_saturation: bool = True,
        save_path: Optional[str] = None,
        REF_ILLUMINANT: Optional[np.ndarray] = None,
        FFC_kwargs: Optional[Dict[str, Any]] = None,
        GC_kwargs: Optional[Dict[str, Any]] = None,
        WB_kwargs: Optional[Dict[str, Any]] = None,
        CC_kwargs: Optional[Dict[str, Any]] = None,
        chart_detection: Optional[Dict[str, Any]] = None,
        color_semantics=None,
        reference_profile=None,
        image_options: Optional[Dict[str, Any]] = None,
        stage_policy=None,
        fitting_policy=None,
        validation_options: Optional[Dict[str, Any]] = None,
        model_format: str = "pickle",
    ) -> None:
        """Initialize configuration with default or provided values."""
        self.do_ffc = do_ffc
        self.do_gc = do_gc
        self.do_wb = do_wb
        self.do_cc = do_cc
        self.save = save
        self.save_path = save_path
        self.check_saturation = check_saturation
        self.REF_ILLUMINANT = deepcopy(REF_ILLUMINANT)
        
        # Stage-specific configuration
        self.FFC_kwargs = deepcopy(FFC_kwargs) if FFC_kwargs is not None else {}
        self.GC_kwargs = deepcopy(GC_kwargs) if GC_kwargs is not None else {}
        self.WB_kwargs = deepcopy(WB_kwargs) if WB_kwargs is not None else {}
        self.CC_kwargs = deepcopy(CC_kwargs) if CC_kwargs is not None else {}
        self.chart_detection = deepcopy(chart_detection) if chart_detection is not None else {}
        self.color_semantics = ColorSemantics.from_value(color_semantics).to_dict()
        self.reference_profile = (ReferenceProfile.from_dict(reference_profile)
                                  if isinstance(reference_profile, dict) else deepcopy(reference_profile))
        self.image_options = deepcopy(image_options) if image_options is not None else {}
        self.stage_policy = StagePolicy.from_value(stage_policy).to_dict()
        self.fitting_policy = FittingPolicy.from_value(fitting_policy).to_dict()
        self.validation_options = deepcopy(validation_options) if validation_options is not None else {}
        self.model_format = model_format

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return deepcopy({
            "do_ffc": self.do_ffc,
            "do_gc": self.do_gc,
            "do_wb": self.do_wb,
            "do_cc": self.do_cc,
            "save": self.save,
            "save_path": self.save_path,
            "check_saturation": self.check_saturation,
            "REF_ILLUMINANT": self.REF_ILLUMINANT.tolist() if self.REF_ILLUMINANT is not None else None,
            "FFC_kwargs": self.FFC_kwargs,
            "GC_kwargs": self.GC_kwargs,
            "WB_kwargs": self.WB_kwargs,
            "CC_kwargs": self.CC_kwargs,
            "chart_detection": self.chart_detection,
            "color_semantics": self.color_semantics,
            "reference_profile": None if self.reference_profile is None else self.reference_profile.to_dict(),
            "image_options": self.image_options,
            "stage_policy": self.stage_policy,
            "fitting_policy": self.fitting_policy,
            "validation_options": self.validation_options,
            "model_format": self.model_format,
        })

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            Config instance.
        """
        config_dict = deepcopy(config_dict)
        # Convert REF_ILLUMINANT back to numpy array if present
        if "REF_ILLUMINANT" in config_dict and config_dict["REF_ILLUMINANT"] is not None:
            config_dict["REF_ILLUMINANT"] = np.array(config_dict["REF_ILLUMINANT"])
        
        return cls(**config_dict)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid.
            TypeError: If parameter types are incorrect.
        """
        # Validate boolean flags
        for flag in ["do_ffc", "do_gc", "do_wb", "do_cc", "save", "check_saturation"]:
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a boolean")

        # Validate save_path if save is enabled
        if self.save and self.save_path is None:
            raise ValueError("save_path must be provided when save=True")
        if self.model_format not in ("pickle", "bundle"):
            raise ValueError("model_format must be pickle or bundle")

        # Validate REF_ILLUMINANT if provided
        if self.REF_ILLUMINANT is not None:
            if not isinstance(self.REF_ILLUMINANT, np.ndarray):
                raise TypeError("REF_ILLUMINANT must be a numpy array")
            white_xy(self.REF_ILLUMINANT)

        # Validate kwargs dictionaries
        for kwargs_name in ["FFC_kwargs", "GC_kwargs", "WB_kwargs", "CC_kwargs", "chart_detection", "image_options", "validation_options"]:
            kwargs = getattr(self, kwargs_name)
            if not isinstance(kwargs, dict):
                raise TypeError(f"{kwargs_name} must be a dictionary")

        from .detection import DetectionOptions
        DetectionOptions(**self.chart_detection)
        semantics = ColorSemantics.from_value(self.color_semantics)
        stages = StagePolicy.from_value(self.stage_policy)
        fitting = FittingPolicy.from_value(self.fitting_policy)
        if (stages.version != "legacy-v1" or fitting.version != "legacy-v1") and semantics.version != "explicit-v1":
            raise ValueError("modern stage and reproducible fitting policies require explicit-v1 color semantics")
        if stages.version == "modern-v1" and self.do_cc and fitting.version == "legacy-v1":
            raise ValueError("modern-v1 CC requires reproducible-v1 fitting to honor observation exclusion")
        if fitting.version != "legacy-v1" and stages.version != "modern-v1":
            raise ValueError("reproducible-v1 requires modern-v1 stages to avoid fit-only legacy pixel scaling")
        if set(self.validation_options) - {"enabled", "n_splits", "random_state"}:
            raise ValueError("unknown validation_options")
        if type(self.validation_options.get("enabled", True)) is not bool:
            raise TypeError("validation_options.enabled must be boolean")
        if type(self.validation_options.get("n_splits", 4)) is not int or self.validation_options.get("n_splits", 4) < 2:
            raise ValueError("validation_options.n_splits must be an integer >=2")
        if type(self.validation_options.get("random_state", 42)) is not int or not 0 <= self.validation_options.get("random_state", 42) < 2**32:
            raise ValueError("validation_options.random_state must be an integer in [0,2**32)")
        if stages.version == "modern-v1":
            if any(kwargs.get("show", False) for kwargs in (self.GC_kwargs, self.WB_kwargs, self.CC_kwargs)):
                raise ValueError("modern fitting has no interactive show hook; inspect observations and numerical_report")
            allowed_gc = {"show", "get_deltaE"}
            allowed_wb = {"show", "get_deltaE"}
            if set(self.GC_kwargs) - allowed_gc:
                raise ValueError("modern GC uses StagePolicy rather than legacy polynomial degree settings")
            if set(self.WB_kwargs) - allowed_wb:
                raise ValueError("modern WB settings belong in StagePolicy")
            if self.FFC_kwargs.get("safe_mode", True) is not True:
                raise ValueError("modern-v1 requires safe FFC estimation")
            allowed_ffc = {"model_path", "manual_crop", "show", "bins", "smooth_window", "crop_rect",
                           "degree", "interactions", "fit_method", "max_iter", "tol", "verbose", "random_seed",
                           "cache_multiplier", "get_deltaE", "safe_mode", "missing_plane", "min_signal",
                           "max_saturated_fraction", "min_crop_fraction", "min_gain", "max_gain",
                           "learning_rate", "n_components"}
            if set(self.FFC_kwargs) - allowed_ffc:
                raise ValueError("unsupported modern FFC settings: %s" % sorted(set(self.FFC_kwargs) - allowed_ffc))
            if "learning_rate" in self.FFC_kwargs and self.FFC_kwargs.get("fit_method", "linear") != "nn":
                raise ValueError("FFC learning_rate only affects fit_method='nn'")
            if "n_components" in self.FFC_kwargs and self.FFC_kwargs.get("fit_method", "linear") != "pls":
                raise ValueError("FFC n_components only affects fit_method='pls'")
        if semantics.version == "legacy-v1" and (semantics.input_encoding != "srgb" or semantics.output_encoding != "srgb" or semantics.adaptation != "CAT02" or semantics.evaluation_white != D65 or semantics.output_policy != "clip"):
            raise ValueError("legacy-v1 retains historical encoded sRGB/CAT02 behavior; choose explicit-v1 for new color semantics")
        if self.reference_profile is not None:
            if not isinstance(self.reference_profile, ReferenceProfile):
                raise TypeError("reference_profile must be ReferenceProfile or its validated dictionary")
            if self.reference_profile.layout != (4, 6) or self.reference_profile.template_id != "classic_4x6":
                raise ValueError("this calibration engine supports Classic24 reference layouts")
            if semantics.version != "explicit-v1":
                raise ValueError("custom reference profiles require explicit-v1 color semantics")
        if semantics.version == "explicit-v1":
            if semantics.output_policy != "clip":
                raise ValueError("this stage engine clips outputs; explicit-v1 pipeline requires output_policy='clip'")
            if self.REF_ILLUMINANT is not None and not np.allclose(self.REF_ILLUMINANT, D65, atol=1e-12, rtol=0):
                raise ValueError("explicit-v1 working RGB is D65; declare other reference whites in ReferenceProfile")
            if self.chart_detection.get("backend") == "mcc" or self.chart_detection.get("allow_mcc_fallback", False):
                raise ValueError("explicit-v1 observations require model geometry; MCC remains a legacy-v1 compatibility route")
        unknown_options = set(self.image_options) - {"apply_orientation", "icc_policy", "alpha_policy", "alpha_background", "grayscale_policy", "frame_index", "max_pixels", "max_decoded_bytes", "max_file_bytes"}
        if unknown_options:
            raise ValueError("unknown image_options: %s" % sorted(unknown_options))
        if semantics.version == "legacy-v1" and ((set(self.image_options) - {"apply_orientation", "icc_policy"}) or self.image_options.get("icc_policy") == "convert-srgb"):
            raise ValueError("new file decoding policies require explicit-v1 color semantics")
        if "apply_orientation" in self.image_options and not isinstance(self.image_options["apply_orientation"], bool):
            raise TypeError("image_options.apply_orientation must be boolean")
        if self.image_options.get("icc_policy", "reject") not in ("reject", "assume-declared", "convert-srgb"):
            raise ValueError("image_options.icc_policy must be reject, assume-declared or convert-srgb")
        if self.image_options.get("alpha_policy", "reject") not in ("reject", "drop", "composite"):
            raise ValueError("image_options.alpha_policy must be reject, drop or composite")
        if self.image_options.get("grayscale_policy", "reject") not in ("reject", "replicate"):
            raise ValueError("image_options.grayscale_policy must be reject or replicate")
        if "alpha_background" in self.image_options:
            background = np.asarray(self.image_options["alpha_background"])
            if background.shape != (3,) or background.dtype.kind not in "fiu" or not np.isfinite(background).all() or np.any(background < 0) or np.any(background > 1):
                raise ValueError("image_options.alpha_background must be three finite RGB values in [0,1]")
        if self.image_options.get("frame_index") is not None and (type(self.image_options["frame_index"]) is not int or self.image_options["frame_index"] < 0):
            raise ValueError("image_options.frame_index must be a nonnegative integer or None")
        for key in ("max_pixels", "max_decoded_bytes", "max_file_bytes"):
            if key in self.image_options and (type(self.image_options[key]) is not int or self.image_options[key] < 1):
                raise ValueError("image_options.%s must be a positive integer" % key)
        if semantics.version == "explicit-v1" and self.image_options.get("apply_orientation", True) is False:
            raise ValueError("explicit pipeline geometry requires applied orientation; use read_image_data separately to inspect raw coordinates")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(do_ffc={self.do_ffc}, do_gc={self.do_gc}, "
            f"do_wb={self.do_wb}, do_cc={self.do_cc}, "
            f"save={self.save}, save_path={self.save_path})"
        )
