"""
Model storage and persistence module
=====================================

Provides a container for storing and loading trained correction models.
Models include flat-field correction multipliers, gamma correction coefficients,
white balance matrices, and color correction models/matrices.
"""

import os
import pickle
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

__all__ = ["MyModels"]


class MyModels:
    """
    Container for storing and persisting color correction models.

    This class stores trained models from each stage of the pipeline:
    - Flat-field correction (FFC) multiplier
    - Gamma correction (GC) coefficients
    - White balance (WB) diagonal matrix
    - Color correction (CC) model or matrix

    Attributes:
        model_ffc: Flat-field correction multiplier (numpy array).
        model_gc: Gamma correction coefficients (numpy array).
        model_wb: White balance diagonal matrix (numpy array).
        model_cc: Color correction model (matrix or trained model object).

    Example:
        >>> models = MyModels()
        >>> # After training...
        >>> models.model_ffc = ffc_multiplier
        >>> models.model_gc = gc_coeffs
        >>> models.model_wb = wb_matrix
        >>> models.model_cc = (ccm, params, "ours")
        >>> 
        >>> # Save models
        >>> models.save("results", "my_models")
        >>> 
        >>> # Load models later
        >>> loaded_models = MyModels()
        >>> loaded_models.load("results", "my_models")
    """

    def __init__(self) -> None:
        """Initialize empty model container."""
        self.model_ffc: Optional[Any] = None
        self.model_gc: Optional[Any] = None
        self.model_wb: Optional[Any] = None
        self.model_cc: Optional[Any] = None
        self.metadata = {}

    @staticmethod
    def _validated_state(loaded: Any) -> dict:
        """Validate the legacy container before changing an active model set.

        This checks the container contract, not pickle security. Unpickling can
        execute code; callers must only load files from a trusted source.
        """
        if not isinstance(loaded, MyModels):
            raise ValueError("Expected a MyModels model container")
        names = ("model_ffc", "model_gc", "model_wb", "model_cc")
        if any(name not in vars(loaded) for name in names):
            raise ValueError("Model container is missing required stage attributes")
        state = {name: vars(loaded)[name] for name in names}
        metadata = deepcopy(vars(loaded).get("metadata", {}))
        if not isinstance(metadata, dict):
            raise ValueError("model metadata must be a dictionary")
        cc = state["model_cc"]
        if not metadata and isinstance(cc, tuple) and len(cc) == 3 and getattr(cc[0], "fitting_version", None) == "ui-compatible-v1":
            raise ValueError("UI-compatible calibration requires its fitting and color metadata")
        if metadata:
            from .colorimetry import ColorSemantics, D65, white_xy
            from .contracts import ReferenceProfile, ObservationSet
            required = {"schema", "color_semantics", "reference_profile", "enabled_stages", "input_image", "observations"}
            if not required.issubset(metadata) or type(metadata["schema"]) is not int or metadata["schema"] != 1:
                raise ValueError("unsupported or incomplete calibration metadata")
            semantics = ColorSemantics.from_value(metadata["color_semantics"])
            reference_illuminant = metadata.get("legacy_reference_illuminant", D65)
            if isinstance(reference_illuminant, str):
                raise ValueError("saved reference illuminant must be a numeric xy pair")
            reference_illuminant = white_xy(reference_illuminant)
            if semantics.version == "explicit-v1" and any(abs(a - b) > 1e-12 for a, b in zip(reference_illuminant, D65)):
                raise ValueError("explicit saved calibration requires a D65 reference illuminant")
            profile = ReferenceProfile.from_dict(metadata["reference_profile"])
            if profile.template_id != "classic_4x6" or profile.layout != (4, 6):
                raise ValueError("saved calibration requires a supported Classic24 reference layout")
            switches = metadata["enabled_stages"]
            expected_switches = {"do_ffc", "do_gc", "do_wb", "do_cc", "check_saturation"}
            if not isinstance(switches, dict) or set(switches) != expected_switches or any(type(value) is not bool for value in switches.values()):
                raise ValueError("model stage switches must contain exactly the five known boolean flags")
            image_options = metadata.get("image_options", {})
            if not isinstance(image_options, dict):
                raise ValueError("saved image_options must be a dictionary")
            # Loading is another entry into the current engine. Pure conversion
            # semantics can describe policies that this stage engine cannot honor;
            # enforce the same capability gates used before fitting.
            from .config import Config
            validation = Config(
                **switches, color_semantics=semantics,
                reference_profile=profile if semantics.version == "explicit-v1" else None,
                image_options=image_options,
                stage_policy=metadata.get("stage_policy"),
                fitting_policy=metadata.get("fitting_policy"),
                validation_options=metadata.get("validation_options"),
            )
            validation.validate()
            from .stage_policies import StagePolicy, MonotonicToneModel
            from .fitting import FittingPolicy
            stages = StagePolicy.from_value(metadata.get("stage_policy"))
            fitting = FittingPolicy.from_value(metadata.get("fitting_policy"))
            if stages.version == "modern-v1":
                import numpy as np
                if state["model_gc"] is not None:
                    curve = state["model_gc"]
                    if not isinstance(curve, MonotonicToneModel):
                        raise ValueError("modern GC requires a neutral-pchip-v1 tone model")
                    MonotonicToneModel(curve.measured_knots, curve.target_knots, curve.version)
                if state["model_ffc"] is not None:
                    gain = np.asarray(state["model_ffc"])
                    if gain.ndim != 2 or not gain.size or not np.isfinite(gain).all() or np.any(gain <= 0):
                        raise ValueError("modern FFC requires positive finite spatial gains")
                    quality = metadata.get("numerical_report", {}).get("FFC", {})
                    bounds = quality.get("gain_limits")
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bounds) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1] <= 100:
                        raise ValueError("saved modern FFC requires finite ordered gain limits")
                    shape = metadata["input_image"].get("shape")
                    if not isinstance(shape, list) or len(shape) != 3 or tuple(shape[:2]) != gain.shape or quality.get("source_shape") != shape[:2]:
                        raise ValueError("saved modern FFC gain dimensions must match calibration geometry")
                    if np.min(gain) < bounds[0] or np.max(gain) > bounds[1]:
                        raise ValueError("saved FFC gains violate recorded quality limits")
                if state["model_wb"] is not None:
                    matrix = np.asarray(state["model_wb"])
                    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.array_equal(matrix, np.diag(np.diag(matrix))) or np.any(np.diag(matrix) <= 0) or np.any(np.diag(matrix) > stages.wb_max_gain):
                        raise ValueError("modern WB requires a bounded positive diagonal matrix")
            elif isinstance(state["model_gc"], MonotonicToneModel):
                raise ValueError("neutral-pchip-v1 requires modern stage metadata")
            cc = state["model_cc"]
            if fitting.version == "ui-compatible-v1" and cc is not None:
                from .core.correction import Regressor_Model, _predict_direct
                from .ui_fitting import UIPredictor, options
                if not isinstance(cc, tuple) or len(cc) != 3 or cc[2] != "ours" or type(cc[0]) is not Regressor_Model or type(cc[0].model) is not UIPredictor:
                    raise ValueError("UI CC requires its known numerical predictor")
                model = cc[0]
                recorded = metadata.get("numerical_report", {}).get("CC", {}).get("fit", {})
                settings = {k:list(v) if isinstance(v,tuple) else v for k,v in model.model.settings.items()}
                if (options(cc[1]) != cc[1] or settings != cc[1] or recorded.get("effective_parameters") != cc[1]
                        or recorded.get("version") != fitting.version or getattr(model,"fitting_version",None) != fitting.version
                        or getattr(model,"fit_report",None) != recorded or model.degree != cc[1]["degree"] or model.mtd != cc[1]["mtd"]
                        or semantics.fitting_domain != "linear-rgb" or recorded.get("fitting_domain") != "linear-rgb"
                        or recorded.get("target_domain") != "unclipped-linear-srgb"):
                    raise ValueError("UI estimator contradicts recorded settings or fitting semantics")
                for key in ("use_lut", "lazy_lut", "lut_min_pixels"):
                    if getattr(model,key) != cc[1][key]:
                        raise ValueError("UI estimator contradicts recorded LUT settings")
                if model._lut_grid_size != cc[1]["lut_grid_size"]:
                    raise ValueError("UI estimator contradicts recorded LUT grid")
                predictor = model.model
                UIPredictor(settings,predictor.coefficients,predictor.intercept,predictor.weights,predictor.biases)
                prediction = _predict_direct(np.array([[.1,.2,.3],[.4,.5,.6]]),model)
                if prediction.shape != (2,3) or not np.isfinite(prediction).all():
                    raise ValueError("invalid UI prediction")
            elif fitting.version == "reproducible-v1" and cc is not None:
                import numpy as np
                if not isinstance(cc, tuple) or len(cc) != 3 or cc[2] not in ("ours", "conv") or not isinstance(cc[1], dict):
                    raise ValueError("reproducible CC requires a known model/options/method tuple")
                recorded = metadata.get("numerical_report", {}).get("CC", {}).get("fit", {})
                if recorded.get("version") != fitting.version or recorded.get("effective_parameters") != cc[1]:
                    raise ValueError("reproducible CC model contradicts recorded effective parameters")
                if cc[2] == "ours":
                    from .core.correction import Regressor_Model, _predict_direct
                    model = cc[0]
                    if not isinstance(model, Regressor_Model) or getattr(model, "fitting_version", None) != fitting.version or model.degree != cc[1].get("degree") or model.mtd != cc[1].get("mtd") or getattr(model, "fit_report", {}) != recorded:
                        raise ValueError("reproducible CC estimator contradicts fitted version or effective degree/method")
                    prediction = _predict_direct(np.array([[.1, .2, .3], [.4, .5, .6]]), model)
                    if prediction.shape != (2, 3) or not np.isfinite(prediction).all():
                        raise ValueError("reproducible CC estimator has invalid inference parameters")
                else:
                    import colour
                    options = cc[1]
                    if set(options) != {"method", "degree", "root_polynomial_expansion"} or options["method"] != "Finlayson 2015" or type(options["degree"]) is not int or not 1 <= options["degree"] <= 4 or type(options["root_polynomial_expansion"]) is not bool:
                        raise ValueError("unsupported reproducible conventional CC options")
                    features = colour.characterisation.polynomial_expansion_Finlayson2015(np.array([[.1, .2, .3]]), degree=options["degree"], root_polynomial_expansion=options["root_polynomial_expansion"])
                    matrix = np.asarray(cc[0])
                    if matrix.shape != (3, features.shape[1]) or not np.isfinite(matrix).all():
                        raise ValueError("reproducible conventional CC matrix contradicts feature dimensions")
            elif cc is not None and isinstance(cc, tuple) and len(cc) == 3 and getattr(cc[0], "fitting_version", None) in ("reproducible-v1", "ui-compatible-v1"):
                raise ValueError("versioned CC requires matching fitting policy metadata")
            if semantics.version == "explicit-v1" and switches["do_wb"] and len(profile.neutral_indices) < 2:
                raise ValueError("saved WB calibration requires at least two declared neutral cells")
            for switch, name in (("do_ffc", "model_ffc"), ("do_gc", "model_gc"),
                                 ("do_wb", "model_wb"), ("do_cc", "model_cc")):
                if switches[switch] != (state[name] is not None):
                    raise ValueError("saved stage switches contradict fitted model presence: %s" % switch)
            observations = ObservationSet.from_dict(metadata["observations"])
            observations.validate_reference(profile)
            for stage_observations in metadata.get("stage_observations", {}).values():
                recorded = ObservationSet.from_dict(stage_observations)
                recorded.validate_reference(profile)
                if recorded.source_digest != observations.source_digest:
                    raise ValueError("stage observations have a different source identity")
            if observations.source_digest != metadata["input_image"].get("digest"):
                raise ValueError("model observation source does not match input-image contract")
        state["metadata"] = metadata
        return state

    def save(self, directory: str, name: str = "models") -> None:
        """
        Save all models to a pickle file with atomic destination replacement.

        Serialization, flushing, and a read-back check finish in a temporary
        file in the destination directory before replacing an existing file.
        Failed writes therefore preserve the previous destination. This is the
        legacy pickle format: only load files from trusted sources.

        Args:
            directory: Directory path where models will be saved.
            name: Base filename for the pickle file (without extension).

        Raises:
            OSError: If directory cannot be created or file cannot be written.

        Example:
            >>> models.save("results", "experiment_01")
            # Creates: results/experiment_01.pkl
        """
        directory_path = Path(directory)
        filepath = directory_path / f"{name}.pkl"
        temporary_path = None

        try:
            self._validated_state(self)
            directory_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory_path, prefix=f".{filepath.name}.",
                suffix=".tmp", delete=False,
            ) as f:
                temporary_path = Path(f.name)
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            with temporary_path.open("rb") as f:
                self._validated_state(pickle.load(f))
            os.replace(temporary_path, filepath)
        except Exception as e:
            raise OSError(f"Failed to save models to {filepath}: {e}") from e
        finally:
            if temporary_path is not None:
                # A failed cleanup must not hide the write/replace error.
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, directory: str, name: str = "models") -> None:
        """
        Load models from a pickle file.

        This method loads a previously saved MyModels instance and copies
        all model attributes to the current instance only after validation.
        A failed read or incomplete container leaves the active models intact.

        Warning:
            Pickle can execute arbitrary code during loading. Only load files
            you trust; validation after unpickling is not a security boundary.

        Args:
            directory: Directory path where models are stored.
            name: Base filename of the pickle file (without extension).

        Raises:
            FileNotFoundError: If the pickle file does not exist.
            OSError: If file cannot be read.
            pickle.UnpicklingError: If pickle file is corrupted.

        Example:
            >>> models = MyModels()
            >>> models.load("results", "experiment_01")
            # Loads from: results/experiment_01.pkl
        """
        filepath = Path(directory) / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, "rb") as f:
                loaded: "MyModels" = pickle.load(f)
            state = self._validated_state(loaded)
        except Exception as e:
            raise OSError(f"Failed to load models from {filepath}: {e}") from e

        self.__dict__.update(state)

        # A trusted pickle does not establish an identity for a portable artifact.
        self.__dict__.pop("bundle_digest", None)
        self.__dict__.pop("bundle_interpretation", None)

    def has_models(self) -> bool:
        """
        Check if any models have been trained/loaded.

        Returns:
            True if at least one model is not None, False otherwise.
        """
        return any(
            [
                self.model_ffc is not None,
                self.model_gc is not None,
                self.model_wb is not None,
                self.model_cc is not None,
            ]
        )

    def save_bundle(self, path, *, limits=None):
        """Save supported inference parameters as a non-executable bundle.

        The existing save/load methods remain the trusted-file pickle path.
        Unsupported estimators fail explicitly; no pickle fallback is used.
        """
        from .bundle import save_bundle
        return save_bundle(self, path, limits=limits)

    def load_bundle(self, path, *, limits=None):
        """Validate a complete bundle before replacing active model state."""
        from .bundle import load_bundle
        candidate = load_bundle(path, limits=limits)
        state = self._validated_state(candidate)
        state["bundle_digest"] = candidate.bundle_digest
        state["bundle_interpretation"] = candidate.bundle_interpretation
        self.__dict__.update(state)
        return self

    def clear(self) -> None:
        """Clear all stored models."""
        self.model_ffc = None
        self.model_gc = None
        self.model_wb = None
        self.model_cc = None
        self.metadata = {}
        self.__dict__.pop("bundle_digest", None)
        self.__dict__.pop("bundle_interpretation", None)

    def __repr__(self) -> str:
        """String representation showing which models are loaded."""
        models_status = {
            "FFC": self.model_ffc is not None,
            "GC": self.model_gc is not None,
            "WB": self.model_wb is not None,
            "CC": self.model_cc is not None,
        }
        loaded = [name for name, status in models_status.items() if status]
        return f"MyModels(loaded=[{', '.join(loaded)}])"
