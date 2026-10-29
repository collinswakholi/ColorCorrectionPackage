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

        # Validate REF_ILLUMINANT if provided
        if self.REF_ILLUMINANT is not None:
            if not isinstance(self.REF_ILLUMINANT, np.ndarray):
                raise TypeError("REF_ILLUMINANT must be a numpy array")
            white_xy(self.REF_ILLUMINANT)

        # Validate kwargs dictionaries
        for kwargs_name in ["FFC_kwargs", "GC_kwargs", "WB_kwargs", "CC_kwargs", "chart_detection", "image_options"]:
            kwargs = getattr(self, kwargs_name)
            if not isinstance(kwargs, dict):
                raise TypeError(f"{kwargs_name} must be a dictionary")

        from .detection import DetectionOptions
        DetectionOptions(**self.chart_detection)
        semantics = ColorSemantics.from_value(self.color_semantics)
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
        unknown_options = set(self.image_options) - {"apply_orientation", "icc_policy"}
        if unknown_options:
            raise ValueError("unknown image_options: %s" % sorted(unknown_options))
        if "apply_orientation" in self.image_options and not isinstance(self.image_options["apply_orientation"], bool):
            raise TypeError("image_options.apply_orientation must be boolean")
        if self.image_options.get("icc_policy", "reject") not in ("reject", "assume-declared"):
            raise ValueError("image_options.icc_policy must be reject or assume-declared")
        if semantics.version == "explicit-v1" and self.image_options.get("apply_orientation", True) is False:
            raise ValueError("explicit pipeline geometry requires applied orientation; use read_image_data separately to inspect raw coordinates")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(do_ffc={self.do_ffc}, do_gc={self.do_gc}, "
            f"do_wb={self.do_wb}, do_cc={self.do_cc}, "
            f"save={self.save}, save_path={self.save_path})"
        )
