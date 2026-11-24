"""
Color-correction inference classes
==================================

``Regressor_Model`` holds a fitted color-correction estimator and owns its 3D LUT
cache; ``predict_`` applies it. Fitting lives in
:mod:`ColorCorrectionPipeline.ui_fitting`. ``CustomNN`` (the 2.x ``custom``
estimator) and the legacy polynomial gamma application ``predict_image`` are
load-only and resolved from :mod:`ColorCorrectionPipeline._compat` so saved
calibrations keep applying.
"""

import hashlib
import json
import logging
import sys
import threading
from typing import Any, Optional

import numpy as np

from .accel import build_3d_lut, trilinear_lut_apply
from ..ui_fitting import VERSION as _UI_VERSION, UIPredictor

__all__ = [
    "Regressor_Model",
    "CustomNN",  # noqa: F822 (module __getattr__)
    "predict_",
    "predict_image",  # noqa: F822 (module __getattr__)
]

logger = logging.getLogger(__name__)

_FEATURE_BACKEND = ("is_cuda", "device_", "get_poly_features")


def _load_feature_backend():
    # Historical callers can replace these hooks. Only fill unbound names.
    from . import transforms

    for name in _FEATURE_BACKEND:
        globals().setdefault(name, getattr(transforms, name))


def __getattr__(name):
    if name in _FEATURE_BACKEND:
        _load_feature_backend()
        return globals()[name]
    if name == "CustomNN":
        # Pickles from 1.3-2.x name this path; deferred so importing this module
        # does not import scikit-learn or torch.
        from .._compat.estimators import CustomNN

        globals()["CustomNN"] = CustomNN
        return CustomNN
    if name in ("predict_image", "gc_lightness"):
        from .._compat import legacy_gc

        value = getattr(legacy_gc, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


# ----------------------------------------------------------------------------
# Content identity of an estimator, used to key its derived LUT. Executable
# objects are never serialized and prediction callbacks are never invoked to
# fingerprint them; unknown estimator state is deliberately uncacheable.
# ----------------------------------------------------------------------------

_LOCK_CREATION = threading.RLock()


def _update(digest, value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not inference content")
        array = np.ascontiguousarray(value)
        digest.update(b"array\0" + str(array.dtype).encode() + repr(array.shape).encode() + b"\0")
        digest.update(memoryview(array).cast("B"))
    elif isinstance(value, np.generic):
        _update(digest, value.item())
    elif value is None or type(value) in (str, bool, int, float):
        digest.update(json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\0")
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\0" + str(len(value)).encode() + b"\0")
        for item in value:
            _update(digest, item)
    elif isinstance(value, dict):
        digest.update(b"mapping\0" + str(len(value)).encode() + b"\0")
        for key in sorted(value):
            _update(digest, key)
            _update(digest, value[key])
    else:
        raise TypeError("unsupported inference state: " + type(value).__name__)


def _torch_state(network):
    import torch.nn as nn

    if type(network) is not nn.Sequential or "forward" in vars(network):
        raise TypeError("unrecognized Torch network")
    from torch.nn.modules import module

    for name in (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    ):
        if getattr(module, name, {}):
            raise TypeError("global Torch hooks make inference state opaque")
    layers = []
    for layer in (network, *tuple(network.children())):
        if any(
            getattr(layer, name, {})
            for name in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ) or "forward" in vars(layer):
            raise TypeError("Torch hooks/overrides make inference state opaque")
    for layer in network:
        if type(layer) is nn.Linear:
            description = ["linear", layer.in_features, layer.out_features]
        elif type(layer) is nn.BatchNorm1d:
            description = [
                "batchnorm",
                layer.num_features,
                layer.eps,
                layer.affine,
                layer.track_running_stats,
            ]
        elif type(layer) in (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Identity):
            description = [type(layer).__name__]
        elif type(layer) is nn.Dropout:
            description = ["dropout", layer.p]
        else:
            raise TypeError("unrecognized Torch layer")
        layers.append(description)
    # Read whitelisted tensors directly; state_dict can execute user hooks.
    parameters = {}
    for index, layer in enumerate(network):
        for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                parameters[str(index) + "." + name] = [
                    str(tensor.device),
                    tensor.detach().cpu().numpy(),
                ]
    return layers, parameters


def _loaded_class(module, name):
    """Return ``module.name`` if that module is already imported, else None.

    An estimator of a class can only exist once its module has been imported, so exact
    ``type(x) is cls`` checks do not need to import scikit-learn (~1 s) or CustomNN for
    calibrations that use neither.
    """
    loaded = sys.modules.get(module)
    return getattr(loaded, name, None) if loaded is not None else None


def _valid_table(table, grid):
    values = np.asarray(table)
    if (
        type(grid) is not int
        or not 2 <= grid <= 129
        or values.shape != (grid, grid, grid, 3)
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise ValueError("LUT must contain finite floating values with matching grid size 2..129")
    return values


def _owned(table):
    values = np.ascontiguousarray(table)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


# ============================================================================
# ML Model Classes
# ============================================================================


class Regressor_Model:
    """
    Container for regression model parameters and fitted model.

    The 3D LUT is a derived cache owned by this class: one table per model,
    synchronized and keyed by the estimator's inference content, rebuilt when that
    content changes and never pickled with an unproven producer.

    Attributes:
        mtd: Model type ("linear", "nn", "pls", "custom")
        model: Fitted estimator
        degree: Polynomial degree for feature expansion
        max_iterations: Maximum training iterations
        random_state: Random seed
        tol: Convergence tolerance
        verbose: Whether to print training progress
        ncomp: Number of PLS components
        param_search: Whether to use parameter search
        hidden_layers: Hidden layer sizes (used by both nn/MLPRegressor and custom/CustomNN)
        learning_rate: Learning rate for NN training
        batch_size: Batch size for NN training
        use_batch_norm: Whether to use batch normalization
        patience: Early stopping patience
        dropout_rate: Dropout rate for NN training
        optim_type: Optimizer type for CustomNN
        use_lut: Whether to use a 3D LUT for image-sized predictions
        lazy_lut: Whether to build the 3D LUT only on first large prediction
        lut_min_pixels: Minimum RGB rows before using/building the LUT
    """

    def __init__(self):
        self.mtd = "linear"
        self.model: Optional[Any] = None
        self.degree = 3
        self.max_iterations = 100
        self.random_state = 42
        self.tol = 1e-6
        self.verbose = False
        self.ncomp = 1
        self.param_search = False

        # NN parameters (shared by MLPRegressor and CustomNN)
        self.hidden_layers = [64]
        self.learning_rate = 0.001
        self.batch_size = 16
        self.use_batch_norm = False
        self.patience = 10
        self.dropout_rate = 0.2
        self.optim_type = "Adam"

        # 3D LUT acceleration. _lut_grid_size is a saved attribute; the table and
        # its builder callback are derived state.
        self._lut = None  # (G, G, G, 3) ndarray or None
        self._lut_grid_size = 33
        self._lut_predict_fn: Optional[Any] = None
        self.use_lut = True
        self.lazy_lut = True
        self.lut_min_pixels = 4096

    def __getstate__(self):
        """Callbacks are derived runtime state, never persisted with the model."""
        state = self.__dict__.copy()
        state["_lut"] = self.saved_lut()
        state.pop("_lut_predict_fn", None)
        for name in (
            "_lut_lock",
            "_lut_identity",
            "_lut_failed_identity",
            "_lut_cache_info",
            "_lut_table_ref",
        ):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        # Supply acceleration defaults for older trusted pickle containers.
        self.__init__()
        self.__dict__.update(state)
        # Pre-LUT artifacts used exact estimator prediction; do not silently opt
        # them into interpolation merely because new acceleration defaults exist.
        if "use_lut" not in state:
            self.use_lut = False
        self.attach_lut_builder()

    def _predict_for_lut(self, rgb_n3):
        result = _predict_direct(rgb_n3, self)
        return (
            result.astype(np.float32)
            if getattr(self, "fitting_version", None) == _UI_VERSION
            else result
        )

    # -- LUT lifecycle ------------------------------------------------------

    def _lock(self):
        with _LOCK_CREATION:
            lock = getattr(self, "_lut_lock", None)
            if lock is None:
                lock = threading.RLock()
                self._lut_lock = lock
            return lock

    def attach_lut_builder(self):
        """Allow the LUT to be built from this estimator and adopt any held table.

        A table restored from a saved artifact is kept as saved; the producer
        content current at this point is pinned, and later estimator mutations
        invalidate it.
        """
        self._lut_predict_fn = (
            self._predict_for_lut if self.use_lut and self.model is not None else None
        )
        if self._lut is not None:
            self._prime_lut_identity()

    def detach_lut_builder(self):
        """Stop building LUTs from the estimator and drop the bound self-reference.

        An already current table may still be used; otherwise prediction is direct.
        """
        self._lut_predict_fn = None

    def hold_lut(self, table):
        """Keep a saved table without building or trusting it until attach_lut_builder."""
        self._lut = table
        self._lut_predict_fn = None

    def clear_lut_cache(self):
        """Drop the derived table and failure memo; a later prediction can rebuild."""
        with self._lock():
            self._lut = None
            self._lut_identity = None
            self._lut_failed_identity = None
            self._lut_table_ref = None
            self._lut_cache_info = {"status": "cleared", "retained_bytes": 0, "entry_limit": 1}

    def ensure_lut(self) -> bool:
        """Make the LUT current for this estimator; False means predict directly."""
        with self._lock():
            if not self.use_lut:
                self._lut = None
                self._lut_identity = None
                self._lut_failed_identity = None
                self._lut_table_ref = None
                self._lut_cache_info = {"status": "disabled", "retained_bytes": 0, "entry_limit": 1}
                return False
            if self.model is None and self._lut is not None and self._lut_predict_fn is None:
                try:
                    self._lut = _owned(_valid_table(self._lut, self._lut_grid_size))
                    self._lut_table_ref = self._lut
                    self._lut_cache_info = {
                        "status": "static-table",
                        "retained_bytes": int(self._lut.nbytes),
                        "entry_limit": 1,
                    }
                    return True
                except ValueError as exc:
                    self._lut = self._lut_table_ref = None
                    self._lut_cache_info = {
                        "status": "failed",
                        "reason": str(exc),
                        "retained_bytes": 0,
                        "entry_limit": 1,
                    }
                    return False
            identity, reason = self._inference_identity()
            if self._lut is not None and not hasattr(self, "_lut_identity"):
                self._prime_lut_identity()
            if (
                self._lut is not None
                and identity is not None
                and identity == getattr(self, "_lut_identity", None)
                and self._lut is getattr(self, "_lut_table_ref", None)
            ):
                # A cache cannot be mutated through a retained caller-owned array.
                self._lut_cache_info.update(status="hit", retained_bytes=int(self._lut.nbytes))
                return True
            if identity is not None and identity == getattr(self, "_lut_failed_identity", None):
                return False
            self._lut = None
            self._lut_table_ref = None
            if self._lut_predict_fn is None:
                self._lut_cache_info = {
                    "status": "unavailable",
                    "reason": "no prediction callback",
                    "retained_bytes": 0,
                    "entry_limit": 1,
                }
                return False
            try:
                grid = self._lut_grid_size
                if type(grid) is not int or not 2 <= grid <= 129:
                    raise ValueError("LUT grid size must be an integer in 2..129")
                table = _owned(
                    _valid_table(build_3d_lut(self._lut_predict_fn, grid_size=grid), grid)
                )
                # Do not publish a table if caller mutation changed its producer
                # while native inference released the GIL. It can retry next call.
                after, _ = self._inference_identity()
                if identity is not None and identity != after:
                    raise RuntimeError("estimator changed during LUT construction")
                self._lut = table
                self._lut_table_ref = table
                self._lut_identity = identity
                self._lut_failed_identity = None
                self._lut_cache_info = {
                    "status": "built",
                    "producer_identity": identity,
                    "identity_unavailable_reason": reason,
                    "retained_bytes": int(table.nbytes),
                    "entry_limit": 1,
                }
                return True
            except Exception as exc:
                self._lut = None
                self._lut_failed_identity = identity
                self._lut_cache_info = {
                    "status": "failed",
                    "reason": type(exc).__name__ + ": " + str(exc),
                    "producer_identity": identity,
                    "retained_bytes": 0,
                    "entry_limit": 1,
                }
                logger.warning("Failed to build 3D LUT: %s", exc)
                return False

    def lut_table(self, *, identified=False):
        """Return the current LUT table, or None when prediction must be direct.

        ``identified=True`` also requires a known producer identity, which fused
        kernels need before they replace the reference equations.
        """
        if identified and self._inference_identity()[0] is None:
            return None
        return self._lut if self.ensure_lut() else None

    def apply_lut(self, rgb):
        """Interpolate ``rgb`` through the table made current by ``ensure_lut``."""
        return trilinear_lut_apply(rgb, self._lut, self._lut_grid_size)

    def saved_lut(self):
        """Return only a current derived table for persistence, without rebuilding.

        Old tables without an identity retain historical adoption semantics. A
        versioned cache must prove its complete producer is unchanged. Never mutate
        the model or execute its prediction callback while preparing saved state.
        """
        with self._lock():
            table = getattr(self, "_lut", None)
            if table is None or not getattr(self, "use_lut", False):
                return None
            try:
                _valid_table(table, self._lut_grid_size)
            except (TypeError, ValueError):
                return None
            if self.model is None and self._lut_predict_fn is None:
                return table
            if not hasattr(self, "_lut_identity"):
                return table
            identity, _ = self._inference_identity()
            if (
                identity is not None
                and identity == self._lut_identity
                and table is getattr(self, "_lut_table_ref", None)
            ):
                return table
            return None

    def _prime_lut_identity(self):
        """Adopt restored values without rebuilding; called after complete loading.

        Pre-version-9 artifacts cannot prove which producer generated their saved
        LUT. Exact saved values are retained and current producer content is pinned
        at adoption. Mutations after this point invalidate that derived cache.
        """
        with self._lock():
            identity, reason = self._inference_identity()
            if self._lut is not None:
                self._lut = _owned(_valid_table(self._lut, self._lut_grid_size))
            self._lut_table_ref = self._lut
            self._lut_identity = identity
            self._lut_failed_identity = None
            self._lut_cache_info = {
                "status": "restored" if self._lut is not None else "empty",
                "producer_identity": identity,
                "identity_unavailable_reason": reason,
                "retained_bytes": 0 if self._lut is None else int(self._lut.nbytes),
                "entry_limit": 1,
            }

    def _inference_identity(self):
        """Return full inference/settings hash, or (None, reason) when opaque."""
        from ..portable_inference import AffinePredictor, MLPPredictor, TorchPredictor

        # Load-only 2.x estimators exist only once their module has been imported.
        CustomNN = _loaded_class("ColorCorrectionPipeline._compat.estimators", "CustomNN")
        DeterministicTorchRegressor = _loaded_class(
            "ColorCorrectionPipeline._compat.estimators", "DeterministicTorchRegressor"
        )
        LinearRegression = _loaded_class("sklearn.linear_model", "LinearRegression")
        PLSRegression = _loaded_class("sklearn.cross_decomposition", "PLSRegression")
        MLPRegressor = _loaded_class("sklearn.neural_network", "MLPRegressor")

        callback = getattr(self, "_lut_predict_fn", None)
        if callback is not None and not (
            getattr(callback, "__self__", None) is self
            and getattr(callback, "__func__", None) is Regressor_Model._predict_for_lut
        ):
            return None, "custom prediction callback state is opaque; reuse disabled"
        estimator = self.model
        try:
            if estimator is None:
                return None, "no estimator; producer identity unavailable"
            if any(
                name in vars(estimator)
                for name in ("predict", "_decision_function", "_forward_pass_fast")
            ) or (type(estimator) is UIPredictor and "predict_rgb" in vars(estimator)):
                return None, "instance prediction override; reuse disabled"
            if type(estimator) is UIPredictor:
                state = {
                    "settings": dict(estimator.settings),
                    "coefficients": estimator.coefficients,
                    "intercept": estimator.intercept,
                    "weights": estimator.weights,
                    "biases": estimator.biases,
                }
            elif type(estimator) in (AffinePredictor, MLPPredictor, TorchPredictor):
                state = vars(estimator)
            elif type(estimator) is LinearRegression:
                state = {"coef": estimator.coef_, "intercept": estimator.intercept_}
            elif type(estimator) is PLSRegression:
                state = {
                    name: getattr(estimator, name)
                    for name in ("coef_", "intercept_", "_x_mean", "_x_std", "_y_mean", "_y_std")
                    if hasattr(estimator, name)
                }
            elif type(estimator) is MLPRegressor:
                state = {
                    "coefs": estimator.coefs_,
                    "biases": estimator.intercepts_,
                    "activation": estimator.activation,
                    "output_activation": estimator.out_activation_,
                    "n_layers": estimator.n_layers_,
                    "n_outputs": estimator.n_outputs_,
                }
            elif type(estimator) in (CustomNN, DeterministicTorchRegressor):
                state = [_torch_state(estimator.model), str(estimator.device)]  # type: ignore[assignment]
            else:
                return None, "unknown estimator state; reuse disabled"
            settings = {
                "degree": self.degree,
                "grid_size": self._lut_grid_size,
                "use_lut": self.use_lut,
                "feature_policy": getattr(self, "fitting_version", "legacy-v1"),
                "portable": getattr(self, "portable_inference", False),
                "reference_implementation": id(_predict_direct),
                "prediction_implementation": id(type(estimator).predict),
                "helper_implementations": [
                    id(getattr(type(estimator), name, None))
                    for name in ("_decision_function", "_forward_pass_fast")
                ],
                "estimator_type": type(estimator).__module__ + "." + type(estimator).__name__,
            }
            # Only the historical polynomial feature path depends on Torch's device.
            if (
                settings["feature_policy"] not in ("reproducible-v1", _UI_VERSION)
                and not settings["portable"]
            ):
                from . import transforms

                settings.update(
                    legacy_cuda=bool(transforms.is_cuda), legacy_device=str(transforms.device_)
                )
            digest = hashlib.sha256()
            _update(digest, settings)
            _update(digest, state)
            return digest.hexdigest(), None
        except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
            return None, type(exc).__name__ + ": " + str(exc)


# CustomNN is load-only; it lives in _compat/estimators.py (see __getattr__ above).


def _predict_direct(RGB: np.ndarray, M: Regressor_Model) -> np.ndarray:
    """Predict RGB values without 3D LUT interpolation."""
    if getattr(M, "fitting_version", None) == _UI_VERSION:
        if type(M.model) is not UIPredictor:
            raise ValueError("UI fitting requires its validated numerical predictor")
        return M.model.predict_rgb(RGB)
    orig_shape = RGB.shape
    if len(RGB.shape) == 3:
        X = RGB.reshape(-1, 3)
    else:
        X = RGB

    if (
        getattr(M, "portable_inference", False)
        or getattr(M, "fitting_version", "legacy-v1") == "reproducible-v1"
    ):
        # Match the recorded CPU feature precision/order rather than switching
        # large prediction arrays to the legacy float32 CUDA feature generator.
        from sklearn.preprocessing import PolynomialFeatures

        X = PolynomialFeatures(degree=M.degree, include_bias=True).fit_transform(
            np.asarray(X, dtype=np.float64)
        )
    else:
        _load_feature_backend()
        X, _ = get_poly_features(X, degree=M.degree)  # noqa: F821 (lazy global)
    pred = M.model.predict(X)
    return pred.reshape(orig_shape)


def predict_(RGB: np.ndarray, M: Regressor_Model) -> np.ndarray:
    """
    Predict RGB values using fitted model.

    Uses the 3D LUT with trilinear interpolation for image-sized inputs when it
    is enabled and current; otherwise evaluates the estimator directly.

    Args:
        RGB: Input RGB image (H, W, 3) or (N, 3)
        M: Fitted Regressor_Model

    Returns:
        np.ndarray: Predicted RGB values (same shape as input)
    """
    orig_shape = RGB.shape
    n_pixels = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else orig_shape[0]
    if n_pixels >= M.lut_min_pixels and M.ensure_lut():
        return M.apply_lut(RGB).reshape(orig_shape)
    return _predict_direct(RGB, M)
