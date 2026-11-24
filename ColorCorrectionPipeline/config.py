"""
Configuration for new calibrations (configuration API version 2).

3.0 removed the API-1 profile and the historical fitting policies. Requests that
would have selected them raise a ValueError naming the replacement; calibrations
saved by those recipes still load (see ColorCorrectionPipeline._compat).
"""

from copy import deepcopy
from typing import Any, Dict, Optional, cast

import numpy as np

from .colorimetry import ColorSemantics, D65, white_xy
from .contracts import ReferenceProfile
from .stage_policies import StagePolicy
from .fitting import FittingPolicy, require_fit_supported
from .config_groups import (
    FFCOptions,
    Steps,
    UNSET,
    GROUP_NAMES,
    GROUPED_LAYOUT,
    resolve_grouped,
    grouped_dict,
    group_view,
)

__all__ = ["Config", "PRESETS"]

#: Named fitting presets accepted by ``calibrate(model=...)`` and ``Config.from_preset``.
#: Each maps to keyword arguments of :class:`Config`.
PRESETS: Dict[str, Dict[str, Any]] = {
    "rpcc2": {"model": "rpcc2"},
    "rpcc3": {"model": "rpcc3"},
    "ccm": {"model": "ccm"},
    "3x3": {"model": "ccm"},
    "pls": {"model": "pls"},
    "high-precision": {"model": "rpcc2", "precision": "high"},
}

_SCALAR_DEFAULTS: Dict[str, Any] = {
    "do_ffc": True,
    "do_gc": True,
    "do_wb": True,
    "do_cc": True,
    "save": False,
    "model_format": "pickle",
}


_LEGACY_SELECTORS = ("mtd", "cc_method", "method", "root_polynomial_expansion")


def validate_validation_options(options):
    """Shared by fitting and by loading saved calibrations."""
    if set(options) - {"enabled", "n_splits", "random_state"}:
        raise ValueError("unknown validation_options")
    if type(options.get("enabled", True)) is not bool:
        raise TypeError("validation_options.enabled must be boolean")
    if type(options.get("n_splits", 4)) is not int or options.get("n_splits", 4) < 2:
        raise ValueError("validation_options.n_splits must be an integer >=2")
    if (
        type(options.get("random_state", 42)) is not int
        or not 0 <= options.get("random_state", 42) < 2**32
    ):
        raise ValueError("validation_options.random_state must be an integer in [0,2**32)")


def validate_image_options(image_options):
    """Shared by fitting and by loading saved calibrations."""
    unknown_options = set(image_options) - {
        "apply_orientation",
        "icc_policy",
        "alpha_policy",
        "alpha_background",
        "grayscale_policy",
        "frame_index",
        "max_pixels",
        "max_decoded_bytes",
        "max_file_bytes",
    }
    if unknown_options:
        raise ValueError("unknown image_options: %s" % sorted(unknown_options))
    if "apply_orientation" in image_options and not isinstance(
        image_options["apply_orientation"], bool
    ):
        raise TypeError("image_options.apply_orientation must be boolean")
    if image_options.get("icc_policy", "reject") not in (
        "reject",
        "assume-declared",
        "convert-srgb",
    ):
        raise ValueError("image_options.icc_policy must be reject, assume-declared or convert-srgb")
    if image_options.get("alpha_policy", "reject") not in ("reject", "drop", "composite"):
        raise ValueError("image_options.alpha_policy must be reject, drop or composite")
    if image_options.get("grayscale_policy", "reject") not in ("reject", "replicate"):
        raise ValueError("image_options.grayscale_policy must be reject or replicate")
    if "alpha_background" in image_options:
        background = np.asarray(image_options["alpha_background"])
        if (
            background.shape != (3,)
            or background.dtype.kind not in "fiu"
            or not np.isfinite(background).all()
            or np.any(background < 0)
            or np.any(background > 1)
        ):
            raise ValueError(
                "image_options.alpha_background must be three finite RGB values in [0,1]"
            )
    if image_options.get("frame_index") is not None and (
        type(image_options["frame_index"]) is not int or image_options["frame_index"] < 0
    ):
        raise ValueError("image_options.frame_index must be a nonnegative integer or None")
    for key in ("max_pixels", "max_decoded_bytes", "max_file_bytes"):
        if key in image_options and (type(image_options[key]) is not int or image_options[key] < 1):
            raise ValueError("image_options.%s must be a positive integer" % key)


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
        check_saturation: Report sampled chart clipping as its own stage.
            Default: False (the UI-compatible fit requires False).
        save_path: Directory path for saving outputs. Default: None
        REF_ILLUMINANT: Reference illuminant as numpy array. Default: None (uses D65)
        FFC_kwargs: Keyword arguments for flat-field correction. Default: None
        GC_kwargs: Keyword arguments for gamma correction. Default: None
        WB_kwargs: Keyword arguments for white balance. Default: None
        CC_kwargs: Keyword arguments for color correction. Default: None
        model: ccm/3x3, rpcc2, rpcc3, linear, pls or nn for new fits.
        basis: polynomial, root or hybrid for linear, PLS or NN.
        degree: CC expansion degree from one through four for new fits.
        precision: auto resolves once at fit, native-byte requires uint8 sRGB,
            and high uses the float64 image runtime.

    Grouped API-2 requests accept ``steps`` (Steps or a mapping), ``input``
    (image loader options), ``chart`` (detector/sampling options), ``reference``
    (ReferenceProfile), ``color`` (color semantics), ``numerics`` (precision),
    ``observations`` (saturation diagnostics), and ``persistence``. GC/WB quality,
    CC fitting/application and CC-only validation live with their owning step.
    See docs/configuration.md for recipes and every setting with its default.

    Example:
        >>> config = Config(steps={
        ...     "ffc": {"fit": {"method": "linear", "degree": 1}},
        ...     "cc": {"fit": {"model": "pls", "basis": "root", "degree": 3,
        ...                     "pls": {"components": 6}}},
        ... })

    ``from_dict`` requires a recorded ``api_version`` (or ``source_version=2``):
    unversioned dictionaries were API-1 configurations, which 3.0 removed.
    """

    @classmethod
    def from_preset(
        cls,
        name: str = "rpcc2",
        *,
        white: bool = True,
        frame_index: Optional[int] = 0,
        device: str = "auto",
        **overrides: Any,
    ) -> "Config":
        """Build a v2 configuration from a named preset.

        Presets: ``rpcc2`` (default), ``rpcc3``, ``ccm`` (alias ``3x3``), ``pls`` and
        ``high-precision``. ``white`` enables flat-field correction, ``frame_index`` picks the
        frame of multi-frame files (``None`` keeps the strict single-frame check) and
        ``overrides`` are further keyword arguments; ``input`` and ``chart`` mappings are merged.

        >>> Config.from_preset("pls", white=False).CC_kwargs["model"]
        'pls'
        """
        key = str(name).lower()
        if key not in PRESETS:
            raise ValueError(
                "unknown preset %r; choose one of %s" % (name, ", ".join(sorted(PRESETS)))
            )
        kwargs: Dict[str, Any] = dict(PRESETS[key])
        image_input: Dict[str, Any] = {} if frame_index is None else {"frame_index": frame_index}
        image_input.update(overrides.pop("input", None) or {})
        chart: Dict[str, Any] = {"device": device}
        chart.update(overrides.pop("chart", None) or {})
        if "steps" not in overrides:
            kwargs["steps"] = Steps(ffc=FFCOptions(enabled=bool(white)))
        kwargs.update(overrides)
        if image_input:
            kwargs["input"] = image_input
        kwargs["chart"] = chart
        return cls(**kwargs)

    def __init__(
        self,
        do_ffc: bool = UNSET,  # type: ignore[assignment]
        do_gc: bool = UNSET,  # type: ignore[assignment]
        do_wb: bool = UNSET,  # type: ignore[assignment]
        do_cc: bool = UNSET,  # type: ignore[assignment]
        save: bool = UNSET,  # type: ignore[assignment]
        check_saturation: Optional[bool] = None,
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
        model_format: str = UNSET,  # type: ignore[assignment]
        *,
        model=None,
        basis=None,
        degree=None,
        precision=None,
        api_version=None,
        steps=None,
        input=None,
        chart=None,
        reference=None,
        color=None,
        numerics=None,
        observations=None,
        persistence=None,
    ) -> None:
        """Initialize configuration with default or provided values."""
        arguments = locals().copy()
        arguments.pop("self")
        groups = {name: arguments.pop(name) for name in GROUP_NAMES}
        if any(value is not None for value in groups.values()):
            # One adapter resolves into the established flat state. Pipeline
            # snapshots and legacy mutable fields therefore retain one owner.
            resolved = resolve_grouped(Config, arguments, groups, required_keys=_SCALAR_DEFAULTS)
            self.__dict__.update(resolved.__dict__)
            return
        do_ffc = _SCALAR_DEFAULTS["do_ffc"] if do_ffc is UNSET else do_ffc
        do_gc = _SCALAR_DEFAULTS["do_gc"] if do_gc is UNSET else do_gc
        do_wb = _SCALAR_DEFAULTS["do_wb"] if do_wb is UNSET else do_wb
        do_cc = _SCALAR_DEFAULTS["do_cc"] if do_cc is UNSET else do_cc
        save = _SCALAR_DEFAULTS["save"] if save is UNSET else save
        model_format = _SCALAR_DEFAULTS["model_format"] if model_format is UNSET else model_format
        supplied = deepcopy(CC_kwargs) if CC_kwargs is not None else {}
        selectors = {"model": model, "basis": basis, "degree": degree}
        for key, value in selectors.items():
            if value is not None:
                if key in supplied and supplied[key] != value:
                    raise ValueError("Conflicting top-level and CC_kwargs " + key)
                supplied[key] = value
        selected_policy = (
            require_fit_supported(fitting_policy) if fitting_policy is not None else None
        )
        if (
            fitting_policy is None
            and "model" not in supplied
            and "basis" not in supplied
            and any(key in supplied for key in _LEGACY_SELECTORS)
        ):
            # In 2.x these keys silently selected the legacy-v1 fitter.
            raise ValueError(
                "CC_kwargs %s selected the legacy-v1 fitter, which was removed in "
                'ColorCorrectionPipeline 3.0. Choose a model instead, e.g. Config(model="3x3") '
                'for a matrix or Config(model="linear"|"pls"|"nn", basis=..., degree=...); '
                "see docs/migration-3.md. Models saved by 2.x still load."
                % sorted(key for key in _LEGACY_SELECTORS if key in supplied)
            )
        if api_version is not None and (type(api_version) is not int or api_version != 2):
            raise ValueError(
                "api_version must be 2: the API-1 configuration profile was removed in "
                "ColorCorrectionPipeline 3.0 (see docs/migration-3.md)"
            )
        self.api_version = 2
        self.precision = "auto" if precision is None else precision
        if self.precision not in ("auto", "native-byte", "high"):
            raise ValueError("precision must be auto, native-byte or high")
        if selected_policy is not None and selected_policy.ui_runtime in (
            "native-v1",
            "precision-f64-v1",
        ):
            expected_precision = (
                "native-byte" if selected_policy.ui_runtime == "native-v1" else "high"
            )
            if self.precision != "auto" and self.precision != expected_precision:
                raise ValueError("precision contradicts the explicit fitting runtime")
        if selected_policy is None or selected_policy.ui_runtime == "auto-v2":
            from dataclasses import replace

            runtime = {
                "auto": "auto-v2",
                "native-byte": "native-v1",
                "high": "precision-f64-v1",
            }[self.precision]
            fitting_policy = (
                FittingPolicy(version="ui-compatible-v1", ui_runtime=runtime)
                if selected_policy is None
                else replace(selected_policy, ui_runtime=runtime)
            )
        supplied.setdefault("model", "rpcc2")
        if supplied["model"] == "3x3":
            supplied["model"] = "ccm"
        if supplied["model"] in ("ccm", "rpcc2", "rpcc3"):
            supplied.setdefault("degree", {"ccm": 1, "rpcc2": 2, "rpcc3": 3}[supplied["model"]])
        if "degree" in supplied and (
            type(supplied["degree"]) is not int or not 1 <= supplied["degree"] <= 4
        ):
            raise ValueError("new CC fits require an integer degree from 1 to 4")
        if degree is not None and supplied["model"] in ("ccm", "rpcc2", "rpcc3"):
            expected = {"ccm": 1, "rpcc2": 2, "rpcc3": 3}[supplied["model"]]
            if type(degree) is not int or degree != expected:
                raise ValueError("degree contradicts the fixed model preset")
        if check_saturation is None:
            check_saturation = False
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
        self.CC_kwargs = supplied
        self.chart_detection = deepcopy(chart_detection) if chart_detection is not None else {}
        self.color_semantics = ColorSemantics.from_value(color_semantics).to_dict()
        self.reference_profile = (
            ReferenceProfile.from_dict(reference_profile)
            if isinstance(reference_profile, dict)
            else deepcopy(reference_profile)
        )
        self.image_options = deepcopy(image_options) if image_options is not None else {}
        self.stage_policy = StagePolicy.from_value(stage_policy).to_dict()
        policy = FittingPolicy.from_value(fitting_policy)
        self.fitting_policy = policy.to_dict()
        self.validation_options = (
            deepcopy(validation_options) if validation_options is not None else {}
        )
        self.model_format = model_format
        if isinstance(color_semantics, dict) and "fitting_domain" not in color_semantics:
            self.color_semantics = ColorSemantics.from_value(
                dict(color_semantics, fitting_domain="linear-rgb")
            ).to_dict()
        if color_semantics is None:
            self.color_semantics = ColorSemantics(
                version="explicit-v1", fitting_domain="linear-rgb"
            ).to_dict()
        if stage_policy is None:
            self.stage_policy = StagePolicy(version="modern-v1", saturation="report").to_dict()
        if reference_profile is None:
            from .ui_native import default_reference

            self.reference_profile = default_reference()
        self.chart_detection.setdefault("sampling_policy", policy.sampling_policy)
        for option in ("samples_per_patch", "sampling_seed"):
            if option in self.chart_detection and self.chart_detection[option] is None:
                raise ValueError(f"{option} must be an integer when supplied")
        if "samples_per_patch" in self.chart_detection:
            if "samples_per_axis" in self.chart_detection:
                raise ValueError("choose samples_per_patch or samples_per_axis, not both")
        else:
            self.chart_detection.setdefault("samples_per_axis", 9)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = deepcopy(
            {
                "do_ffc": self.do_ffc,
                "do_gc": self.do_gc,
                "do_wb": self.do_wb,
                "do_cc": self.do_cc,
                "save": self.save,
                "save_path": self.save_path,
                "check_saturation": self.check_saturation,
                "REF_ILLUMINANT": (
                    self.REF_ILLUMINANT.tolist() if self.REF_ILLUMINANT is not None else None
                ),
                "FFC_kwargs": self.FFC_kwargs,
                "GC_kwargs": self.GC_kwargs,
                "WB_kwargs": self.WB_kwargs,
                "CC_kwargs": self.CC_kwargs,
                "chart_detection": self.chart_detection,
                "color_semantics": self.color_semantics,
                "reference_profile": (
                    None if self.reference_profile is None else self.reference_profile.to_dict()
                ),
                "image_options": self.image_options,
                "stage_policy": self.stage_policy,
                "fitting_policy": self.fitting_policy,
                "validation_options": self.validation_options,
                "model_format": self.model_format,
            }
        )
        result.update(api_version=2, precision=self.precision)
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], *, source_version=None) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary from ``to_dict`` or ``to_grouped_dict``.
            source_version: 2 to read an unversioned mapping as an API-2 request.

        Returns:
            Config instance.

        Raises:
            ValueError: for unversioned (API-1) dictionaries or ``source_version=1``;
                3.0 removed that profile instead of reinterpreting it.
        """
        config_dict = deepcopy(config_dict)
        layout = config_dict.pop("layout", None)
        if layout is not None:
            if layout != GROUPED_LAYOUT:
                raise ValueError("unknown configuration layout: " + str(layout))
            if source_version not in (None, 2) or config_dict.get("api_version", 2) != 2:
                raise ValueError("grouped-v1 requires configuration API version 2")
            config_dict.setdefault("api_version", 2)
        if source_version is not None:
            if type(source_version) is not int or source_version not in (1, 2):
                raise ValueError("source_version must be 1 or 2")
            if "api_version" in config_dict and config_dict["api_version"] != source_version:
                raise ValueError("source_version contradicts the recorded api_version")
            config_dict["api_version"] = source_version
        if "api_version" not in config_dict:
            # 2.x read unversioned dictionaries as API-1 legacy-v1 requests.
            raise ValueError(
                "unversioned configuration dictionaries were API-1 requests, which "
                "ColorCorrectionPipeline 3.0 removed. Pass source_version=2 to read this "
                "mapping as a new API-2 configuration; see docs/migration-3.md."
            )
        # Convert REF_ILLUMINANT back to numpy array if present
        if "REF_ILLUMINANT" in config_dict and config_dict["REF_ILLUMINANT"] is not None:
            config_dict["REF_ILLUMINANT"] = np.array(config_dict["REF_ILLUMINANT"])

        return cls(**config_dict)

    def to_grouped_dict(self) -> Dict[str, Any]:
        """Return explicit grouped-v1 JSON data; flat to_dict remains unchanged.

        Fixed numerical policy identities and supported flat-only compatibility
        keys remain explicit.
        """
        self.validate()
        return cast(Dict[str, Any], grouped_dict(self))

    @property
    def steps(self):
        """Fresh immutable step snapshot; old flat mutations remain observable."""
        return group_view(self, "steps")

    @property
    def input(self):
        return group_view(self, "input")

    @property
    def chart(self):
        return group_view(self, "chart")

    @property
    def reference(self):
        return self.reference_profile

    @property
    def color(self):
        return group_view(self, "color")

    @property
    def numerics(self):
        return group_view(self, "numerics")

    @property
    def observations(self):
        return group_view(self, "observations")

    @property
    def persistence(self):
        return group_view(self, "persistence")

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid, including a fitting policy,
                stage policy or color semantics whose fitter was removed in 3.0.
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
        for kwargs_name in [
            "FFC_kwargs",
            "GC_kwargs",
            "WB_kwargs",
            "CC_kwargs",
            "chart_detection",
            "image_options",
            "validation_options",
        ]:
            kwargs = getattr(self, kwargs_name)
            if not isinstance(kwargs, dict):
                raise TypeError(f"{kwargs_name} must be a dictionary")

        from .detection import DetectionOptions

        DetectionOptions(**self.chart_detection)
        semantics = ColorSemantics.from_value(self.color_semantics)
        stages = StagePolicy.from_value(self.stage_policy)
        fitting = require_fit_supported(self.fitting_policy)
        if stages.version != "modern-v1":
            raise ValueError(
                "stage_policy %r was removed in ColorCorrectionPipeline 3.0; new fits use "
                "modern-v1 (the default). Models saved with it still load." % stages.version
            )
        if semantics.version != "explicit-v1":
            raise ValueError(
                "color_semantics %r was removed in ColorCorrectionPipeline 3.0; new fits use "
                "explicit-v1 (the default). Models saved with it still load." % semantics.version
            )
        if "domain" in self.FFC_kwargs:
            domain = self.FFC_kwargs["domain"]
            if not isinstance(domain, str) or domain not in ("lightness", "linear"):
                raise ValueError("FFC domain must be 'lightness' or 'linear'")
        if self.precision not in ("auto", "native-byte", "high"):
            raise ValueError("precision must be auto, native-byte or high")
        expected_runtime = {"native-byte": "native-v1", "high": "precision-f64-v1"}.get(
            self.precision
        )
        if expected_runtime is not None and fitting.ui_runtime not in (
            "auto-v2",
            expected_runtime,
        ):
            raise ValueError("precision contradicts the explicit fitting runtime")
        if (
            semantics.adaptation != "Bradford"
            or semantics.fitting_domain != "linear-rgb"
            or self.chart_detection.get("sampling_policy") != fitting.sampling_policy
        ):
            raise ValueError(
                "ui-compatible-v1 requires Bradford adaptation, linear-rgb fitting and UI sampling"
            )
        if fitting.ui_runtime == "native-v1" and semantics.input_encoding != "srgb":
            raise ValueError("native UI runtime requires sRGB input")
        if (
            self.check_saturation
            or stages.saturation != "report"
            or fitting.anchor_weight
            or fitting.random_state != 42
            or fitting.condition_limit != 1e12
        ):
            raise ValueError(
                "ui-compatible-v1 requires check_saturation=False, report saturation, no anchors, seed 42 and the default condition_limit"
            )
        from .ui_fitting import options

        degree = self.CC_kwargs.get("degree", 2)
        if type(degree) is not int or not 1 <= degree <= 4:
            raise ValueError("new CC fits require an integer degree from 1 to 4")
        fixed = {"ccm": 1, "rpcc2": 2, "rpcc3": 3}.get(self.CC_kwargs.get("model"))  # type: ignore[arg-type]
        if fixed is not None and degree != fixed:
            raise ValueError("degree contradicts the fixed model preset")
        options(self.CC_kwargs)
        validate_validation_options(self.validation_options)
        if any(
            kwargs.get("show", False) for kwargs in (self.GC_kwargs, self.WB_kwargs, self.CC_kwargs)
        ):
            raise ValueError(
                "modern fitting has no interactive show hook; inspect observations and numerical_report"
            )
        allowed_gc = {"show", "get_deltaE"}
        allowed_wb = {"show", "get_deltaE"}
        if set(self.GC_kwargs) - allowed_gc:
            raise ValueError(
                "modern GC uses StagePolicy rather than legacy polynomial degree settings"
            )
        if set(self.WB_kwargs) - allowed_wb:
            raise ValueError("modern WB settings belong in StagePolicy")
        if self.FFC_kwargs.get("safe_mode", True) is not True:
            raise ValueError("modern-v1 requires safe FFC estimation")
        allowed_ffc = {
            "model_path",
            "manual_crop",
            "show",
            "bins",
            "smooth_window",
            "crop_rect",
            "degree",
            "interactions",
            "fit_method",
            "max_iter",
            "tol",
            "verbose",
            "random_seed",
            "cache_multiplier",
            "get_deltaE",
            "safe_mode",
            "missing_plane",
            "min_signal",
            "max_saturated_fraction",
            "min_crop_fraction",
            "min_gain",
            "max_gain",
            "learning_rate",
            "n_components",
            "domain",
        }
        if set(self.FFC_kwargs) - allowed_ffc:
            raise ValueError(
                "unsupported modern FFC settings: %s" % sorted(set(self.FFC_kwargs) - allowed_ffc)
            )
        if (
            "learning_rate" in self.FFC_kwargs
            and self.FFC_kwargs.get("fit_method", "linear") != "nn"
        ):
            raise ValueError("FFC learning_rate only affects fit_method='nn'")
        if (
            "n_components" in self.FFC_kwargs
            and self.FFC_kwargs.get("fit_method", "linear") != "pls"
        ):
            raise ValueError("FFC n_components only affects fit_method='pls'")
        if self.reference_profile is not None:
            if not isinstance(self.reference_profile, ReferenceProfile):
                raise TypeError(
                    "reference_profile must be ReferenceProfile or its validated dictionary"
                )
            from .chart_references import TEMPLATES, is_builtin_reference, template_for

            if self.reference_profile.template_id not in TEMPLATES:
                raise ValueError(
                    "references must use a classic_4x6, spyder_4x6 or digital_sg_10x14 template"
                )
            requested = self.chart_detection.get("type", "auto")
            if (
                requested != "auto"
                and not is_builtin_reference(self.reference_profile)
                and template_for(requested) != self.reference_profile.template_id
            ):
                raise ValueError(
                    "chart type %r contradicts the custom reference's %s template"
                    % (requested, self.reference_profile.template_id)
                )
        if semantics.output_policy != "clip":
            raise ValueError(
                "this stage engine clips outputs; explicit-v1 pipeline requires output_policy='clip'"
            )
        if self.REF_ILLUMINANT is not None and not np.allclose(
            self.REF_ILLUMINANT, D65, atol=1e-12, rtol=0
        ):
            raise ValueError(
                "explicit-v1 working RGB is D65; declare other reference whites in ReferenceProfile"
            )
        validate_image_options(self.image_options)
        if self.image_options.get("apply_orientation", True) is False:
            raise ValueError(
                "explicit pipeline geometry requires applied orientation; use read_image_data separately to inspect raw coordinates"
            )

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(do_ffc={self.do_ffc}, do_gc={self.do_gc}, "
            f"do_wb={self.do_wb}, do_cc={self.do_cc}, "
            f"save={self.save}, save_path={self.save_path})"
        )


# Omitted values need a sentinel to distinguish explicit default-value aliases
# from grouped requests. Keep the established defaults visible to help/inspect.
import inspect as _inspect  # noqa: E402

_signature = _inspect.signature(Config.__init__)
Config.__init__.__signature__ = _signature.replace(  # type: ignore[attr-defined]
    parameters=[
        (
            parameter.replace(default=_SCALAR_DEFAULTS[parameter.name])
            if parameter.name in _SCALAR_DEFAULTS
            else parameter
        )
        for parameter in _signature.parameters.values()
    ]
)
del _signature, _inspect
