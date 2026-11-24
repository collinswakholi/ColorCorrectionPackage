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
    >>> from ColorCorrectionPipeline import ColorCorrection, Config, Steps, CCOptions
    >>> config = Config(
    ...     steps=Steps(cc=CCOptions(fit={"model": "rpcc3"})),
    ...     input={"frame_index": 0},
    ... )
    >>> cc = ColorCorrection()
    >>> fit = cc.fit_or_raise("chart.jpg", "white.jpg", config=config)
    >>> results = cc.predict_image("scene.jpg", stages="final")
"""

import gc
import copy
import logging
import threading
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ._lazy import pandas as pd  # deferred: only used inside functions

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from .config import Config
from .stage_policies import StagePolicy, MonotonicToneModel
from .fitting import FittingPolicy
from .contracts import ImageData, ReferenceProfile, ObservationSet
from .colorimetry import ColorSemantics, D65, srgb_to_linear, lab_to_xyz, adapt_xyz
from .io.image_data import read_image_data
from .observations import bind_stage_measurements, capture_observations
from .detection import (
    DetectionOptions,
    bind_detection_source,
    detection_context,
    get_detection_diagnostics,
)
from .core import predict_, to_float64, to_uint8
from .core.accel import apply_ffc_float
from .models import MyModels

__all__ = ["ColorCorrection", "FitResult", "CalibrationError"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class FitResult:
    """Outputs from one successful calibration. Later refits replace the pipeline's
    models without replacing this result's models. Arrays and models remain mutable.
    """

    name: str
    models: MyModels
    metrics: Dict[str, Any]
    images: Dict[str, np.ndarray]
    artifacts: Tuple[str, ...]


class CalibrationError(RuntimeError):
    """An enabled fit stage failed. Details belong to this attempt."""

    def __init__(self, errors):
        self.errors = copy.deepcopy(errors)
        super().__init__(self.errors)

    def __str__(self):
        details = [f"{item['stage']}: {item['message']}" for item in self.errors]
        return "Calibration failed" + (": " + "; ".join(details) if details else "")


@dataclass(eq=False)
class _FittedState:
    """State published together to prediction readers after a successful fit or load."""

    models: MyModels = field(default_factory=MyModels)
    Models_path: Optional[str] = None
    color_semantics: ColorSemantics = field(default_factory=ColorSemantics)
    stage_policy: StagePolicy = field(default_factory=StagePolicy)
    fitting_policy: FittingPolicy = field(default_factory=FittingPolicy)
    reference_profile: Optional[ReferenceProfile] = None
    observations: Optional[ObservationSet] = None
    stage_observations: dict = field(default_factory=dict)
    REF_ILLUMINANT: Optional[np.ndarray] = None
    validation_options: dict = field(default_factory=dict)
    numerical_report: dict = field(default_factory=dict)
    _chart_index: int = 0

    # Fit-only inputs, derived views and caches have no counterpart in a bundle.
    Image: Optional[np.ndarray] = None
    White_Image: Optional[np.ndarray] = None
    fitted_config: Optional[Config] = None
    image_data: Optional[ImageData] = None
    white_image_data: Optional[ImageData] = None
    _source_face: Any = None
    REFERENCE_CHART: Any = None
    REFERENCE_RGB_PD: Optional["pd.DataFrame"] = None
    REFERENCE_NEUTRAL_PATCHES_PD: Optional["pd.DataFrame"] = None
    REFERENCE_DISPLAY_RGB_PD: Optional["pd.DataFrame"] = None

    @classmethod
    def from_fit(cls, candidate):
        return cls(**{item.name: getattr(candidate, item.name) for item in fields(cls)})


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
        self._run_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self.__activate_fitted_state(_FittedState(), {})

    def __activate_fitted_state(self, state: _FittedState, diagnostics: Dict[str, Any]) -> None:
        with self._state_lock:
            for name, value in vars(state).items():
                setattr(self, name, value)
            self._stage_errors = {}
            self.last_run = diagnostics

    def get_reference_values(self, REF_ILLUMINANT: Optional[np.ndarray] = None) -> "pd.DataFrame":
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
        if self.color_semantics.version != "explicit-v1":
            raise ValueError(
                "legacy-v1 reference values served the fitters removed in 3.0; "
                "use explicit-v1 color semantics (the default for new fits)"
            )
        profile = self.reference_profile or ReferenceProfile.classic24()
        self.reference_profile = profile
        self.REF_ILLUMINANT = np.asarray(D65, dtype=np.float64)
        from .ui_native import enabled as native_enabled, reference as native_reference

        fitting_rgb = (
            native_reference(profile)[1]
            if native_enabled(self)
            else self.color_semantics.reference_to_working_rgb(
                profile.canonical_lab_d50, clip=False
            )
        )
        display_rgb = self.color_semantics.reference_to_working_rgb(
            profile.canonical_lab_d50, clip=True
        )
        self.REFERENCE_RGB_PD = pd.DataFrame(
            fitting_rgb, columns=["R", "G", "B"], index=profile.names
        )
        self.REFERENCE_DISPLAY_RGB_PD = pd.DataFrame(
            display_rgb, columns=["R", "G", "B"], index=profile.names
        )
        self.REFERENCE_NEUTRAL_PATCHES_PD = self.REFERENCE_RGB_PD.iloc[
            list(profile.neutral_indices)
        ].copy()
        xyz = adapt_xyz(
            lab_to_xyz(profile.canonical_lab_d50, white="D50"),
            "D50",
            "D65",
            transform=self.color_semantics.adaptation,
        )
        total = xyz.sum(axis=1)
        xy = np.divide(
            xyz[:, :2], total[:, None], out=np.zeros_like(xyz[:, :2]), where=total[:, None] != 0
        )
        xyY = np.column_stack((xy, xyz[:, 1]))
        from .checker_compat import make_colour_checker

        self.REFERENCE_CHART = make_colour_checker(
            name=profile.name,
            data=dict(zip(profile.cell_ids, xyY)),
            illuminant=self.REF_ILLUMINANT,
            rows=4,
            columns=6,
        )
        return self.REFERENCE_RGB_PD

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
            >>> ffc_kwargs = {"bins": 50, "degree": 5, "fit_method": "poly"}
            >>> img_ffc, metrics, err = cc.do_flat_field_correction(img, True, ffc_kwargs)
        """
        if not do_ffc:
            return Image, None, False
        if self.White_Image is None:
            self._record_stage_error(
                "FFC", ValueError("FFC is enabled but White_Image was not provided")
            )
            return Image, None, True

        try:
            from .ui_stages import fit_ffc

            return fit_ffc(self, Image, ffc_kwargs or {})
        except Exception as e:
            self._record_stage_error("FFC", e)
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
            logger.info("Skipping saturation check")
            return Image, None, None

        try:
            from .modern import report_saturation

            return report_saturation(self, Image)
        except Exception as e:
            self._record_stage_error("saturation", e)
            raise

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
            >>> img_gc, metrics, err = cc.do_gamma_correction(img, True, {})
        """
        if not do_gc:
            logger.info("Skipping gamma correction")
            return Image, None, False

        try:
            from .modern import fit_gamma

            return fit_gamma(self, Image)
        except Exception as e:
            self._record_stage_error("GC", e)
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
            logger.info("Skipping white balance")
            return Image, None, False

        try:
            from .modern import fit_wb

            return fit_wb(self, Image)
        except Exception as e:
            self._record_stage_error("WB", e)
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
            cc_method: "conv" (Finlayson-style matrix vocabulary) or "ours" (default)
            cc_kwargs: Color correction parameters dict

        Returns:
            tuple: (corrected_image, metrics, error_flag)
                - corrected_image: RGB float64 0-1
                - metrics: Dict of metrics if requested
                - error_flag: True if exception raised

        Example:
            >>> img_cc, metrics, err = cc.do_color_correction(img, True, "ours", {"model": "rpcc2"})
        """
        if not do_cc:
            logger.info("Skipping color correction")
            return Image, None, False

        try:
            from .ui_fitting import fit_stage

            return fit_stage(self, Image, cc_method, cc_kwargs)
        except Exception as e:
            self._record_stage_error("CC", e)
            return Image, None, True

    @staticmethod
    def _normalized_image(image):
        if not image.orientation_applied and image.source_orientation != 1:
            raise ValueError(
                "pipeline geometry requires applied orientation; call ImageData.with_orientation_applied()"
            )
        return image.with_orientation_applied()

    @classmethod
    def _prepare_input(cls, image, semantics, image_options=None, *, _borrow_readonly=False):
        def _working_rgb(data):
            if _borrow_readonly and data.color_space == "srgb":
                return data.pixels.view()
            return data.to_srgb()

        if isinstance(image, ImageData):
            data = cls._normalized_image(image)
            return data, _working_rgb(data)
        if isinstance(image, (str, os.PathLike)):
            if semantics.version == "explicit-v1":
                data = read_image_data(
                    image, color_space=semantics.input_encoding, **(image_options or {})
                )
                data = cls._normalized_image(data)
                return data, _working_rgb(data)
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Cannot read Image from '{image}'")
            bgr = cv2.imread(os.fspath(image))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read Image from '{image}'")
            legacy = cls._validate_rgb(to_float64(bgr[:, :, ::-1]))
            data = ImageData.from_array(
                legacy,
                provenance={
                    "source_kind": "legacy-file",
                    "decoder": "OpenCV default imread",
                    "original_precision": "unknown after legacy decode",
                    "original_orientation": "legacy decoder behavior",
                },
            )
            return data, _working_rgb(data) if _borrow_readonly else legacy
        if isinstance(image, np.ndarray):
            if semantics.version == "legacy-v1":
                legacy = cls._validate_rgb(image)
                data = ImageData.from_array(legacy)
                # Legacy float32 Lab/FFC helpers retain their recorded input
                # precision; only the float64 route can borrow canonical data.
                return data, (
                    _working_rgb(data)
                    if _borrow_readonly and legacy.dtype == np.float64
                    else legacy
                )
            data = ImageData.from_array(image, color_space=semantics.input_encoding)
            return data, _working_rgb(data)
        raise TypeError("Image must be a path, ImageData, or numpy array")

    def _active_semantics(self):
        return ColorSemantics.from_value(
            getattr(self.models, "metadata", {}).get("color_semantics")
        )

    def _convert_outputs(self, images):
        semantics = self._active_semantics()
        if semantics.output_encoding == "linear-srgb":
            return {key: srgb_to_linear(value) for key, value in images.items()}
        return images

    def _explicit_metrics(self, before, after, selected_index):
        if not before.observations or not after.observations:
            reason = before.unavailable_reason or after.unavailable_reason
            self.last_run.setdefault("warnings", []).append(
                {"metrics": "unavailable", "reason": reason}
            )
            return {"metrics_available": False}
        first, last = before.observations[selected_index], after.observations[selected_index]
        first.validate_reference(self.reference_profile)
        last.validate_reference(self.reference_profile)
        a = self.color_semantics.delta_e00(
            first.measured_rgb, self.reference_profile.canonical_lab_d50, input_encoding="srgb"
        )
        b = self.color_semantics.delta_e00(
            last.measured_rgb, self.reference_profile.canonical_lab_d50, input_encoding="srgb"
        )
        return {
            "metrics_available": True,
            "scope": "training-chart fixed-region mean RGB; not independent validation",
            "delta_e00_before_mean": float(a.mean()),
            "delta_e00_after_mean": float(b.mean()),
            "delta_e00_after_median": float(np.median(b)),
            "delta_e00_after_max": float(b.max()),
            "cells": len(b),
            "reference_digest": self.reference_profile.digest,
            "evaluation_white": str(self.color_semantics.evaluation_white),
            "adaptation": self.color_semantics.adaptation,
            "selected_face_id": last.face_id,
        }

    def _record_stage_error(self, stage: str, error: Exception) -> None:
        self._stage_errors[stage] = {"type": type(error).__name__, "message": str(error)}
        logging.getLogger(__name__).error("%s: %s", stage, error)

    @staticmethod
    def _validate_rgb(image: np.ndarray, label: str = "Image") -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
            raise ValueError(f"{label} must have nonempty shape (height, width, 3)")
        if not np.issubdtype(image.dtype, np.floating):
            raise TypeError(
                f"{label} ndarray must be floating RGB in [0, 1]; use read_image for integer images"
            )
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError(f"{label} must contain finite RGB values in [0, 1]")
        return image.copy()

    def run(
        self,
        Image: Union[str, np.ndarray],
        White_Image: Optional[Union[str, np.ndarray]] = None,
        name_: str = "",
        config: Optional[Config] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], bool]:
        """Fit a candidate calibration and commit it only on complete success.

        The historical ``(metrics, images, error)`` return is retained. Inspect
        ``last_run`` for stage status, failure reasons and detection diagnostics.
        Enabled stages are required: missing FFC input or missing calibration
        charts fail the run. A failed run leaves the previous calibration intact.
        Input/configuration and filesystem exceptions still raise, after recording
        failure and preserving active state. Direct ``do_*`` calls remain low-level
        mutating operations; the transaction guarantee applies to ``run``.
        """
        with self._run_lock:
            return self._run_transaction(Image, White_Image, name_, config)

    def fit_or_raise(
        self,
        Image: Union[str, os.PathLike, np.ndarray, ImageData],
        White_Image: Optional[Union[str, os.PathLike, np.ndarray, ImageData]] = None,
        name_: str = "",
        config: Optional[Config] = None,
    ) -> FitResult:
        """Return this fit's outputs, or raise CalibrationError for a failed stage.

        Input, configuration and persistence exceptions propagate as in run().
        A failed attempt leaves the previously active calibration available.
        """
        # Capture outputs and diagnostics before a competing fit or load can replace them.
        with self._run_lock:
            metrics, images, failed = self.run(Image, White_Image, name_, config)
            if failed:
                raise CalibrationError(self.last_run["errors"])
            return FitResult(name_, self.models, metrics, images, tuple(self.last_run["artifacts"]))

    def fit(
        self, chart, white=None, *, name: str = "", config: Optional[Config] = None
    ) -> FitResult:
        """Fit a calibration; same as :meth:`fit_or_raise` with lowercase argument names.

        Raises :class:`CalibrationError` if an enabled stage fails; a failed attempt leaves
        the previously active calibration in place.
        """
        return self.fit_or_raise(chart, white, name_=name, config=config)

    def predict(self, image, *, stages=None, chunk_pixels=None) -> Dict[str, np.ndarray]:
        """Apply the active calibration; same as :meth:`predict_image` without plotting."""
        return self.predict_image(image, stages=stages, chunk_pixels=chunk_pixels)

    def _run_transaction(self, Image, White_Image, name_, config):
        candidate = copy.copy(self)
        candidate.models = MyModels()
        # Preserve instance-bound extension hooks, rebinding them to the candidate.
        for attr, value in tuple(candidate.__dict__.items()):
            if getattr(value, "__self__", None) is self and hasattr(value, "__func__"):
                setattr(candidate, attr, value.__func__.__get__(candidate, type(candidate)))
        candidate._stage_errors = {}
        candidate.last_run = {"success": False, "stages": {}, "errors": [], "artifacts": []}
        try:
            selected = copy.deepcopy(config) if config is not None else Config()
            if not isinstance(selected, Config):
                raise TypeError("config must be a Config instance")
            selected.validate()
            candidate.color_semantics = ColorSemantics.from_value(selected.color_semantics)
            candidate.stage_policy = StagePolicy.from_value(selected.stage_policy)
            candidate.fitting_policy = FittingPolicy.from_value(selected.fitting_policy)
            candidate.validation_options = copy.deepcopy(selected.validation_options)
            candidate.numerical_report = {}
            candidate._source_face = None
            candidate._chart_index = selected.chart_detection.get("chart_index", 0)
            candidate.reference_profile = selected.reference_profile
            candidate.image_data = candidate.white_image_data = candidate.observations = None
            candidate.stage_observations = {}
            candidate.last_run["color_semantics"] = candidate.color_semantics.to_dict()
            for stage, enabled in (
                ("FFC", selected.do_ffc),
                ("saturation", selected.check_saturation),
                ("GC", selected.do_gc),
                ("WB", selected.do_wb),
                ("CC", selected.do_cc),
            ):
                candidate.last_run["stages"][stage] = {
                    "status": "not_run" if enabled else "disabled"
                }
            if not isinstance(name_, str) or "/" in name_ or "\\" in name_ or name_ in (".", ".."):
                raise ValueError("name_ must be a filename stem, without path separators")
            # A drive-relative name can escape staging; ':' also selects NTFS
            # alternate streams. POSIX colon filenames retain their behavior.
            if os.name == "nt" and ":" in name_:
                raise ValueError("name_ must not contain Windows drive or stream qualifiers")
            with detection_context(DetectionOptions(**selected.chart_detection)):
                try:
                    metrics, images, failed = candidate._run_stages(
                        Image, White_Image, name_, selected
                    )
                finally:
                    candidate.last_run["detection"] = copy.deepcopy(get_detection_diagnostics())
                    candidate.last_run["numerical_report"] = copy.deepcopy(
                        candidate.numerical_report
                    )
            if candidate.models.metadata:
                execution_keys = (
                    "backend",
                    "status",
                    "sha256",
                    "provider",
                    "requested_device",
                    "selected_device",
                    "device_id",
                    "runtime_version",
                    "graphics_backend",
                    "cpu_fallback",
                    "execution_verified",
                    "execution_evidence",
                    "node_provider_counts",
                    "attempts",
                    "geometry_source",
                    "charts",
                    "selected_chart_index",
                    "ignored_layouts",
                )
                candidate.models.metadata["detector_execution"] = [
                    {key: row[key] for key in execution_keys if key in row}
                    for row in candidate.last_run["detection"]
                ]
                candidate.models.metadata["chart_selection"] = {
                    "chart_index": selected.chart_detection.get("chart_index", 0),
                    "identity_scope": "fixed faces within one source capture",
                }
                candidate.last_run["contracts"] = copy.deepcopy(candidate.models.metadata)
            if failed:
                self.last_run = candidate.last_run
                return metrics, images, True
            candidate.fitted_config = copy.deepcopy(selected)
            converted_images = candidate._convert_outputs(images)
            # Persistence is part of a requested run. Prepare every output before
            # replacing destinations, and preserve the active calibration on error.
            if selected.save:
                candidate._save_run_outputs(metrics, name_, selected)
            state = _FittedState.from_fit(candidate)
            self.__activate_fitted_state(state, {**candidate.last_run, "success": True})
            return metrics, converted_images, False
        except Exception as exc:
            candidate.last_run["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            self.last_run = candidate.last_run
            raise

    def _run_stages(self, Image, White_Image, name_, config):
        self.image_data, self.Image = self._prepare_input(
            Image, self.color_semantics, config.image_options
        )
        self.White_Image = None
        self.white_image_data = None
        if config.do_ffc:
            try:
                if isinstance(White_Image, ImageData):
                    self.white_image_data = self._normalized_image(White_Image)
                elif isinstance(White_Image, (str, os.PathLike)):
                    self.white_image_data = read_image_data(
                        White_Image,
                        color_space=self.color_semantics.input_encoding,
                        **config.image_options,
                    )
                elif isinstance(White_Image, np.ndarray):
                    if White_Image.dtype == np.uint8:
                        self.white_image_data = ImageData.from_array(
                            White_Image,
                            channel_order="BGR",
                            color_space=self.color_semantics.input_encoding,
                        )
                    else:
                        self.white_image_data = ImageData.from_array(
                            White_Image, color_space=self.color_semantics.input_encoding
                        )
                elif White_Image is not None:
                    raise TypeError("White_Image must be a path, ImageData, or numpy array")
                if self.white_image_data is None:
                    raise ValueError("FFC is enabled but White_Image was not provided")
                self.white_image_data = self._normalized_image(self.white_image_data)
                self.White_Image = to_uint8(self.white_image_data.to_srgb()[:, :, ::-1])
            except (FileNotFoundError, ValueError, TypeError) as exc:
                self._record_stage_error("FFC", exc)
                self.last_run["stages"]["FFC"] = {"status": "failed", **self._stage_errors["FFC"]}
                self.last_run["errors"].append({"stage": "FFC", **self._stage_errors["FFC"]})
                return {}, {}, True

        # Config.validate admits only ui-compatible-v1 with modern-v1 stages and
        # explicit-v1 color semantics; auto-v2 resolves its image runtime here.
        from dataclasses import replace
        from .ui_native import high_precision, working_input

        if self.fitting_policy.ui_runtime == "auto-v2":
            sources = [self.image_data]
            if self.white_image_data is not None:
                sources.append(self.white_image_data)
            byte_sources = all(
                data.source_dtype == "uint8"
                and tuple(data.source_range) == (0.0, 255.0)
                and data.color_space == "srgb"
                for data in sources
            )
            requested_precision = getattr(config, "precision", None) or "auto"
            if requested_precision == "native-byte" and not byte_sources:
                raise ValueError(
                    "native-byte precision requires full-range uint8 sRGB chart and white inputs"
                )
            selected_runtime = (
                "precision-f64-v1"
                if requested_precision == "high"
                else (
                    "native-v1"
                    if requested_precision == "native-byte" or byte_sources
                    else "precision-f64-v1"
                )
            )
            self.fitting_policy = replace(self.fitting_policy, ui_runtime=selected_runtime)
            config.fitting_policy = self.fitting_policy.to_dict()
        high = high_precision(self)
        if high:
            # The first high-precision evaluator uses direct CC; a stored
            # float32 LUT would discard distinctions the source retains.
            if config.CC_kwargs.get("use_lut", False):
                raise ValueError(
                    "high precision requires use_lut=False; its approximate LUT has not been qualified"
                )
            config.CC_kwargs.setdefault("use_lut", False)
        self.Image = working_input(self.image_data, self.Image, high=high)
        from .detection.charts import _scope

        _scope.get().measurement_runtime = (
            "precision-f64-v1" if high else config.chart_detection.get("sampling_policy")
        )

        self.get_reference_values(config.REF_ILLUMINANT)
        # Geometry always comes from the original quantized source, even if the
        # first extraction occurs after FFC; a float32 round trip could truncate an
        # exact code value one unit low. Optional metric switches cannot change it.
        bind_detection_source(np.rint(self.image_data.to_srgb()[:, :, ::-1] * 255).astype(np.uint8))
        _scope.get().reference_profile = self.reference_profile
        if config.do_wb and len(self.reference_profile.neutral_indices) < 2:
            raise ValueError("WB requires at least two declared neutral cell IDs")
        self.Models_path = None
        self._pending_saturation = None
        from .modern import prepare_source

        try:
            prepare_source(
                self,
                required=any((config.do_gc, config.do_wb, config.do_cc, config.check_saturation)),
            )
        except Exception as exc:
            failed_stage = next(
                stage
                for stage, enabled in (
                    ("saturation", config.check_saturation),
                    ("GC", config.do_gc),
                    ("WB", config.do_wb),
                    ("CC", config.do_cc),
                )
                if enabled
            )
            self._record_stage_error(failed_stage, exc)
            self.last_run["stages"][failed_stage] = {
                "status": "failed",
                **self._stage_errors[failed_stage],
            }
            self.last_run["errors"].append(
                {"stage": failed_stage, **self._stage_errors[failed_stage]}
            )
            return {}, {}, True
        metrics, images = {}, {}
        current = self.Image
        # Fit-time FFC gain domain; committed to metadata only when FFC is fitted.
        self._ffc_domain = "lightness"

        def fit_color(image):
            method = config.CC_kwargs.get("cc_method", "ours")
            return self.do_color_correction(image, True, method, config.CC_kwargs)

        stages = (
            (
                "FFC",
                config.do_ffc,
                lambda x: self.do_flat_field_correction(x, True, config.FFC_kwargs),
            ),
            ("saturation", config.check_saturation, None),
            ("GC", config.do_gc, lambda x: self.do_gamma_correction(x, True, config.GC_kwargs)),
            ("WB", config.do_wb, lambda x: self.do_white_balance(x, True, config.WB_kwargs)),
            ("CC", config.do_cc, fit_color),
        )
        for stage, enabled, operation in stages:
            if not enabled:
                continue
            started = time.perf_counter()
            try:
                explicit = self.color_semantics.version == "explicit-v1"
                stage_kwargs = getattr(config, stage + "_kwargs", {})
                wants_metrics = stage != "saturation" and stage_kwargs.get("get_deltaE", True)
                bind_stage_measurements(current if explicit else None)
                before = capture_observations(
                    current,
                    self.image_data,
                    self.reference_profile,
                    stage + "_before",
                    explicit=explicit,
                    force=explicit and wants_metrics,
                )
                if stage == "saturation":
                    output, values, ids = self._check_saturation(current, True)
                    self._pending_saturation = (values, ids) if ids is not None else None
                    stage_metrics, failed = None, False
                else:
                    output, stage_metrics, failed = operation(current)
                if failed:
                    if stage not in self._stage_errors:
                        self._record_stage_error(stage, RuntimeError(f"{stage} reported failure"))
                else:
                    if (
                        not isinstance(output, np.ndarray)
                        or output.shape != self.Image.shape
                        or not np.isfinite(output).all()
                    ):
                        raise ValueError(f"{stage} returned invalid or nonfinite image data")
                    if (
                        not np.issubdtype(output.dtype, np.floating)
                        or np.any(output < 0)
                        or np.any(output > 1)
                    ):
                        raise ValueError(f"{stage} must return floating RGB in [0, 1]")
                    if stage != "saturation":
                        fitted = getattr(self.models, "model_" + stage.lower())
                        if fitted is None:
                            raise ValueError(f"{stage} returned without a fitted model")
                        if isinstance(fitted, np.ndarray) and (
                            not fitted.size or not np.isfinite(fitted).all()
                        ):
                            raise ValueError(
                                f"{stage} returned invalid or nonfinite model parameters"
                            )
            except Exception as exc:
                self._record_stage_error(stage, exc)
                failed = True
            elapsed = time.perf_counter() - started
            if failed:
                detail = {"stage": stage, **self._stage_errors[stage]}
                self.last_run["stages"][stage] = {"status": "failed", "seconds": elapsed, **detail}
                self.last_run["errors"].append(detail)
                return metrics, images, True
            self.last_run["stages"][stage] = {"status": "succeeded", "seconds": elapsed}
            after = capture_observations(
                output, self.image_data, self.reference_profile, stage, explicit=explicit
            )
            self.stage_observations[stage] = after
            if explicit and wants_metrics:
                if not before.observations and after.observations:
                    before = capture_observations(
                        current,
                        self.image_data,
                        self.reference_profile,
                        stage + "_before",
                        explicit=True,
                    )
                stage_metrics = self._explicit_metrics(
                    before, after, config.chart_detection.get("chart_index", 0)
                )
            current = output
            if stage != "saturation":
                metrics[f"{name_}_{stage}"] = stage_metrics
                images[f"{name_}_{stage}"] = output
        self.observations = capture_observations(
            self.Image,
            self.image_data,
            self.reference_profile,
            "source",
            explicit=self.color_semantics.version == "explicit-v1",
        )
        self.models.metadata = {
            "schema": 1,
            "color_semantics": self.color_semantics.to_dict(),
            "reference_profile": self.reference_profile.to_dict(),
            "input_image": self.image_data.to_metadata(),
            "white_image": (
                None if self.white_image_data is None else self.white_image_data.to_metadata()
            ),
            "observations": self.observations.to_dict(),
            "stage_observations": {
                key: value.to_dict() for key, value in self.stage_observations.items()
            },
            "enabled_stages": {
                key: getattr(config, key)
                for key in ("do_ffc", "do_gc", "do_wb", "do_cc", "check_saturation")
            },
            "image_options": copy.deepcopy(config.image_options),
            "stage_policy": self.stage_policy.to_dict(),
            "fitting_policy": self.fitting_policy.to_dict(),
            "validation_options": copy.deepcopy(self.validation_options),
            "numerical_report": copy.deepcopy(self.numerical_report),
            "legacy_reference_illuminant": np.asarray(self.REF_ILLUMINANT).tolist(),
            # Key order matches the records 2.x wrote for the same recipe.
            "algorithms": {
                "GC": "neutral-pchip-v1; assumed endpoints 0/100",
                "WB": "neutral-diagonal-v1; recorded headroom",
                "CC": "ui-compatible-v1; linear-light fit; float32 stage boundaries",
                "FFC": "ui-compatible-v1; float32 Lab-D65 gain; plane/poly surface",
                "saturation": "sampled-pixel-report-or-exclude-v1; identity image transform",
                "ui_runtime": self.fitting_policy.ui_runtime,
                "pipeline": "native uint8 decode; joint FFC/GC; ordered sampling",
            },
        }
        if self.fitting_policy.ui_runtime == "precision-f64-v1":
            self.models.metadata["algorithms"].update(
                {
                    "pipeline": "float64 source measurements; joint FFC/GC; ordered sampling",
                    "CC": "ui-compatible-v1; linear-light fit; float64 stages",
                    "FFC": "precision-f64-v1; float64 Lab-D65 gain; plane/poly surface",
                }
            )
        if config.do_ffc and config.FFC_kwargs.get("domain", "lightness") == "linear":
            # Absent records replay the historical L* gain, so only the opt-in
            # linear-light domain is written.
            self.models.metadata["ffc_domain"] = "linear"
            self.models.metadata["algorithms"]["FFC"] += "; linear-light RGB gain"
        self.models.metadata["config_version"] = 2
        self.models.metadata["requested_precision"] = config.precision
        self.last_run["contracts"] = copy.deepcopy(self.models.metadata)
        return metrics, images, False

    def _save_run_outputs(self, metrics, name_, config):
        import tempfile
        import shutil
        from pathlib import Path

        directory = Path(config.save_path)
        directory.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=directory, prefix=".ccp-run-"))
        retain_for_recovery = False
        try:
            frames = []
            if metrics:
                frames.append((f"{name_}_metrics.csv", pd.DataFrame.from_dict(metrics)))
            if self._pending_saturation is not None:
                values, ids = self._pending_saturation
                frames.append(
                    (
                        f"{name_}_Sat_data.csv",
                        pd.DataFrame(
                            {
                                "Image": [name_] * len(ids),
                                "ID": ids,
                                "Value_R": values[:, 0],
                                "Value_G": values[:, 1],
                                "Value_B": values[:, 2],
                            }
                        ),
                    )
                )
            for filename, frame in frames:
                frame.to_csv(work / filename, float_format="%.9f", encoding="utf-8-sig")
            model_name = f"{name_}_models." + ("ccpb" if config.model_format == "bundle" else "pkl")
            if config.model_format == "bundle":
                self.models.save_bundle(work / model_name)
            else:
                self.models.save(str(work), f"{name_}_models")
            names = [filename for filename, _ in frames] + [model_name]
            backups = {}
            for index, filename in enumerate(names):
                destination = directory / filename
                if destination.exists():
                    backup = work / f"backup-{index}"
                    shutil.copy2(destination, backup)
                    backups[filename] = backup
            replaced = []
            try:
                for filename in names:
                    os.replace(work / filename, directory / filename)
                    replaced.append(filename)
            except BaseException as publish_error:
                # Per-file replacement is atomic. This restores earlier replacements
                # after an ordinary publish failure; it is not a crash-atomic bundle.
                # Protect backups before attempting recovery: a second filesystem
                # error or interruption must not delete the remaining good copies.
                retain_for_recovery = True
                recovery = {
                    "directory": str(work),
                    "publish_error": {
                        "type": type(publish_error).__name__,
                        "message": str(publish_error),
                    },
                    "restored": [],
                    "failed": [],
                    "backups": {filename: str(backup) for filename, backup in backups.items()},
                }
                self.last_run["recovery"] = recovery
                for filename in reversed(replaced):
                    destination = directory / filename
                    try:
                        if filename in backups:
                            os.replace(backups[filename], destination)
                        else:
                            destination.unlink(missing_ok=True)
                        recovery["restored"].append(filename)
                    except Exception as restore_error:
                        recovery["failed"].append(
                            {
                                "file": filename,
                                "type": type(restore_error).__name__,
                                "message": str(restore_error),
                            }
                        )
                if recovery["failed"]:
                    raise OSError(
                        "Failed to publish run outputs and restore every prior file; "
                        f"recovery files retained in '{work}'"
                    ) from publish_error
                retain_for_recovery = False
                self.last_run.pop("recovery", None)
                raise
            self.last_run["artifacts"] = [str(directory / filename) for filename in names]
            model_path = str(directory / model_name)
            self.Models_path = model_path
        finally:
            if not retain_for_recovery:
                try:
                    shutil.rmtree(work)
                except OSError as cleanup_error:
                    # Cleanup cannot turn a fully published calibration into a
                    # failed fit. Report the leftover directory for inspection.
                    self.last_run.setdefault("cleanup_warnings", []).append(
                        {
                            "directory": str(work),
                            "type": type(cleanup_error).__name__,
                            "message": str(cleanup_error),
                        }
                    )

    def predict_image(
        self, Image: Union[str, np.ndarray], show: bool = False, *, stages=None, chunk_pixels=None
    ) -> Dict[str, np.ndarray]:
        """Apply one consistent active calibration snapshot (identity if unfitted)."""
        with self._state_lock:
            snapshot = copy.copy(self)
            snapshot.models = copy.copy(self.models)
        if stages is None and chunk_pixels is None:
            return snapshot._predict_image(Image, show)
        return snapshot._predict_image(Image, show, stages=stages, chunk_pixels=chunk_pixels)

    def _predict_image(
        self,
        Image: Union[str, np.ndarray],
        show: bool = False,
        *,
        stages=None,
        chunk_pixels=None,
        _return_context=False,
        _quiet=False,
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
        from .prediction import normalize_stages

        selected = normalize_stages(stages)
        if chunk_pixels is not None and (type(chunk_pixels) is not int or chunk_pixels < 1):
            raise ValueError("chunk_pixels must be a positive integer or None")
        semantics = self._active_semantics()
        metadata = getattr(self.models, "metadata", {})
        started = time.perf_counter()
        from .ui_fitting import is_ui

        data, current = self._prepare_input(
            Image, semantics, metadata.get("image_options", {}), _borrow_readonly=is_ui(self.models)
        )
        if metadata:
            self.REF_ILLUMINANT = np.asarray(
                metadata.get("legacy_reference_illuminant", D65), dtype=np.float64
            )
        from .ui_native import (
            enabled as native_enabled,
            high_precision,
            runtime,
            validate_input,
            predict as native_predict,
        )

        if native_enabled(self.models):
            if not high_precision(self.models):
                validate_input(data)
            decoded = time.perf_counter()
            outputs = self._convert_outputs(
                native_predict(current, self.models, selected, chunk_pixels, native_codes=True)
            )
            elapsed = time.perf_counter() - started
            if not _quiet:
                logger.info("Prediction done in %.2fs", elapsed)
            if _return_context:
                return outputs, {
                    "input_image": data.to_metadata(),
                    "color_semantics": semantics.to_dict(),
                    "selected_stages": list(selected),
                    "timings": {
                        "decode_seconds": decoded - started,
                        "native_execution_seconds": time.perf_counter() - decoded,
                        "total_seconds": elapsed,
                    },
                    "execution": (
                        runtime(self.models) if high_precision(self.models) else "ui-native-v1"
                    ),
                    "fusion": "FFC+GC when both enabled",
                }
            return outputs
        from .ui_fitting import is_ui, round_image

        ui_policy = is_ui(self.models)
        if ui_policy and chunk_pixels is None and not _return_context:
            from .ui_execution import try_execute

            accelerated = try_execute(current, self.models, selected)
            if accelerated is not None:
                outputs = self._convert_outputs(accelerated)
                elapsed = time.perf_counter() - started
                if not _quiet:
                    logger.info("Prediction done in %.2fs", elapsed)
                return outputs
        if ui_policy:
            current = round_image(current)
        expected_shape = current.shape
        input_metadata = data.to_metadata() if _return_context else None
        del data
        decoded = time.perf_counter()
        out_images = {}
        timings = {"decode_seconds": decoded - started}
        order = ("FFC", "GC", "WB", "CC")
        for stage in order[: max(order.index(key) for key in selected) + 1]:
            stage_started = time.perf_counter()
            if stage == "FFC" and self.models.model_ffc is not None:
                if (
                    semantics.version == "explicit-v1"
                    and np.shape(self.models.model_ffc) != current.shape[:2]
                ):
                    raise ValueError(
                        "explicit-v1 FFC requires the calibrated image dimensions; spatial gain maps are not implicitly resized"
                    )
                if ui_policy:
                    from .ui_stages import apply_ffc
                    from .ui_native import ffc_domain

                    current = apply_ffc(current, self.models.model_ffc, ffc_domain(self.models))
                else:
                    current = apply_ffc_float(current, self.models.model_ffc)
            elif stage == "GC" and self.models.model_gc is not None:
                if isinstance(self.models.model_gc, MonotonicToneModel):
                    from .modern import apply_tone

                    current = apply_tone(current, self.models.model_gc)
                else:
                    from ._compat.legacy_gc import predict_image

                    current = predict_image(
                        img=current,
                        coeffs=self.models.model_gc,
                        ref_illuminant=self.REF_ILLUMINANT,
                        explicit_color=semantics.version == "explicit-v1",
                    )
                current = np.clip(current, 0.0, 1.0)
            elif stage == "WB" and self.models.model_wb is not None:
                current = np.clip(current @ self.models.model_wb, 0.0, 1.0)
            elif stage == "CC" and self.models.model_cc is not None:
                model, params, method = self.models.model_cc
                if chunk_pixels is not None and current.shape[0] * current.shape[1] > chunk_pixels:
                    from .prediction import chunked_cc

                    current = chunked_cc(current, self.models.model_cc, chunk_pixels)
                elif method == "conv":
                    from .conventional import apply_conventional

                    current = apply_conventional(RGB=current, CCM=model, **params)
                elif method == "ours":
                    current = predict_(RGB=current, M=model)
                else:
                    raise ValueError("unsupported saved CC method: %s" % method)
                current = np.clip(current, 0.0, 1.0)
            if ui_policy:
                current = round_image(current)
            if _return_context and (
                not isinstance(current, np.ndarray)
                or current.shape != expected_shape
                or not np.isfinite(current).all()
                or np.any(current < 0)
                or np.any(current > 1)
            ):
                raise FloatingPointError(
                    "%s produced invalid/nonfinite image data during batch prediction" % stage
                )
            timings[stage + "_seconds"] = time.perf_counter() - stage_started
            if stage in selected:
                out_images[stage] = current
        outputs = self._convert_outputs(out_images)
        timings["total_seconds"] = time.perf_counter() - started
        if not _quiet:
            logger.info("Prediction done in %.2fs", timings["total_seconds"])
        if _return_context:
            return outputs, {
                "input_image": input_metadata,
                "timings": timings,
                "color_semantics": semantics.to_dict(),
                "selected_stages": list(selected),
            }
        return outputs

    def save_bundle(self, path, *, limits=None):
        """Save an owned calibration snapshot without executable serialization."""
        with self._state_lock:
            models = copy.deepcopy(self.models)
        return models.save_bundle(path, limits=limits)

    def load_bundle(self, path, *, limits=None):
        """Activate a completely validated portable calibration atomically."""
        from .bundle import load_bundle

        with self._run_lock:
            candidate = load_bundle(path, limits=limits)
            metadata = candidate.metadata
            state = _FittedState(
                models=candidate,
                Models_path=os.fspath(path),
                color_semantics=ColorSemantics.from_value(metadata.get("color_semantics")),
                reference_profile=(
                    ReferenceProfile.from_dict(metadata["reference_profile"]) if metadata else None
                ),
                observations=(
                    ObservationSet.from_dict(metadata["observations"]) if metadata else None
                ),
                stage_observations={
                    key: ObservationSet.from_dict(value)
                    for key, value in metadata.get("stage_observations", {}).items()
                },
                REF_ILLUMINANT=np.asarray(metadata.get("legacy_reference_illuminant", D65)),
                stage_policy=StagePolicy.from_value(metadata.get("stage_policy")),
                fitting_policy=FittingPolicy.from_value(metadata.get("fitting_policy")),
                numerical_report=copy.deepcopy(metadata.get("numerical_report", {})),
                validation_options=copy.deepcopy(metadata.get("validation_options", {})),
                _chart_index=metadata.get("chart_selection", {}).get("chart_index", 0),
            )
            loaded_run = {
                "success": True,
                "operation": "load_bundle",
                "source": os.fspath(path),
                "contracts": copy.deepcopy(metadata),
                "numerical_report": copy.deepcopy(state.numerical_report),
                "calibration_id": candidate.bundle_digest,
                "interpretation": candidate.bundle_interpretation,
                "stages": {},
                "errors": [],
                "artifacts": [os.fspath(path)],
            }
            self.__activate_fitted_state(state, loaded_run)
        return self

    def prepare_execution(self, options=None, *, cache=None):
        """Own a calibration snapshot and qualify an opt-in reusable CPU execution path.

        Preparation and sampled error diagnostics are available on the returned
        context manager. Existing prediction APIs retain their numerical policy.
        """
        from .acceleration import PreparedExecution

        return PreparedExecution(self, options, cache=cache)

    def iter_predict(self, images, **options):
        """Stream bounded per-image outcomes from one owned calibration snapshot."""
        from .prediction import iter_predict

        return iter_predict(self, images, **options)

    def predict_batch(self, images, **options):
        """Collect structured outcomes; use iter_predict for bounded result retention."""
        from .results import BatchResult

        with self.iter_predict(images, **options) as stream:
            items = tuple(stream)
        return BatchResult(items=items, summary=stream.summary)

    def save_report(self, path):
        """Atomically save the current run diagnostics without image pixels."""
        from .reporting import save_report

        with self._state_lock:
            report = copy.deepcopy(self.last_run)
        return save_report(path, report)

    # ── Batch / parallel prediction ──────────────────────────────────────

    def predict_images(
        self,
        images: List[Union[str, np.ndarray]],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Apply one calibration snapshot to the entire batch, in input order."""
        with self._state_lock:
            snapshot = copy.copy(self)
            snapshot.models = copy.copy(self.models)
        return snapshot._predict_images(images, on_progress, max_workers)

    def _predict_images(
        self,
        images: List[Union[str, np.ndarray]],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Apply saved models to multiple images in parallel.

        Uses a thread pool so that Numba-compiled kernels (which release
        the GIL) and I/O (cv2.imread) can overlap.

        Parameters
        ----------
        images : list of str | np.ndarray
            Paths or RGB float64 arrays to correct.
        on_progress : callable, optional
            ``on_progress(completed, total, image_name)`` called from the
            main thread after each image finishes.
        max_workers : int, optional
            Number of threads.  Defaults to ``min(os.cpu_count(), 4)``.

        Returns
        -------
        results : list of dict
            Same order as *images*.  Each dict has keys
            ``'FFC', 'GC', 'WB', 'CC'`` → float64 RGB arrays.

        Example
        -------
        >>> cc = ColorCorrection()
        >>> metrics, imgs, err = cc.run(Image=chart, White_Image=white,
        ...                             name_="cal", config=cfg)
        >>> results = cc.predict_images(
        ...     ["img1.jpg", "img2.jpg", "img3.jpg"],
        ...     on_progress=lambda cur, tot, name: print(f"{cur}/{tot} {name}"),
        ... )
        """
        n = len(images)
        if n == 0:
            return []

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 4)
        max_workers = max(1, min(max_workers, n))

        results: List[Optional[Dict[str, np.ndarray]]] = [None] * n

        def _name(img, idx):
            if isinstance(img, str):
                return os.path.basename(img)
            return f"image_{idx}"

        def _worker(idx):
            return idx, self.predict_image(images[idx], show=False)

        completed = 0
        t0 = time.time()

        limit_context = (
            threadpool_limits(limits=1)
            if threadpool_limits is not None and max_workers > 1
            else nullcontext()
        )
        with limit_context:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_worker, i): i for i in range(n)}

                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    completed += 1
                    name = _name(images[idx], idx)
                    if on_progress is not None:
                        on_progress(completed, n, name)

                    # Periodic GC instead of per-image
                    if completed % max(1, max_workers) == 0:
                        gc.collect()

        elapsed = time.time() - t0
        logger.info(
            "Batch prediction done — %d images in %.2fs (%.2fs/img, %d workers)",
            n,
            elapsed,
            elapsed / max(n, 1),
            max_workers,
        )
        gc.collect()

        return results
