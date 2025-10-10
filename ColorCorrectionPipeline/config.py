"""
Configuration module for color_correc_optim
============================================

Provides configuration classes for the color correction pipeline.
Supports TOML and INI file formats with validation.
"""

from typing import Any, Dict, Optional

import numpy as np

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
    ) -> None:
        """Initialize configuration with default or provided values."""
        self.do_ffc = do_ffc
        self.do_gc = do_gc
        self.do_wb = do_wb
        self.do_cc = do_cc
        self.save = save
        self.save_path = save_path
        self.check_saturation = check_saturation
        self.REF_ILLUMINANT = REF_ILLUMINANT
        
        # Stage-specific configuration
        self.FFC_kwargs = FFC_kwargs if FFC_kwargs is not None else {}
        self.GC_kwargs = GC_kwargs if GC_kwargs is not None else {}
        self.WB_kwargs = WB_kwargs if WB_kwargs is not None else {}
        self.CC_kwargs = CC_kwargs if CC_kwargs is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
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
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            Config instance.
        """
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
            if self.REF_ILLUMINANT.size not in [2, 3]:
                raise ValueError("REF_ILLUMINANT must have 2 (xy) or 3 (XYZ) values")

        # Validate kwargs dictionaries
        for kwargs_name in ["FFC_kwargs", "GC_kwargs", "WB_kwargs", "CC_kwargs"]:
            kwargs = getattr(self, kwargs_name)
            if not isinstance(kwargs, dict):
                raise TypeError(f"{kwargs_name} must be a dictionary")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(do_ffc={self.do_ffc}, do_gc={self.do_gc}, "
            f"do_wb={self.do_wb}, do_cc={self.do_cc}, "
            f"save={self.save}, save_path={self.save_path})"
        )
