"""
Color-correction inference classes
==================================

``Regressor_Model`` holds a fitted color-correction estimator and its 3D LUT;
``predict_`` applies it. Fitting lives in :mod:`ColorCorrectionPipeline.ui_fitting`.
``CustomNN`` (the 2.x ``custom`` estimator) and the legacy polynomial gamma
application ``predict_image`` are load-only and resolved from
:mod:`ColorCorrectionPipeline._compat` so saved calibrations keep applying.
"""

import logging
from typing import Any, Optional

import numpy as np

from .accel import build_3d_lut, trilinear_lut_apply

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


# ============================================================================
# ML Model Classes
# ============================================================================


class Regressor_Model:
    """
    Container for regression model parameters and fitted model.

    Attributes:
        mtd: Model type ("linear", "nn", "pls", "custom")
        model: Fitted sklearn/torch model
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
        lut_grid_size: 3D LUT grid resolution
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

        # 3D LUT acceleration (filled after fit_model)
        self._lut = None  # (G, G, G, 3) ndarray or None
        self._lut_grid_size = 33  # default grid resolution
        self._lut_predict_fn: Optional[Any] = None
        self.use_lut = True
        self.lazy_lut = True
        self.lut_min_pixels = 4096

    def __getstate__(self):
        """Callbacks are derived runtime state, never persisted with the model."""
        from .lut_cache import persistence_lut

        state = self.__dict__.copy()
        state["_lut"] = persistence_lut(self)
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
        self._lut_predict_fn = None
        if self.use_lut and self.model is not None:
            self._lut_predict_fn = self._predict_for_lut
        if self._lut is not None:
            from .lut_cache import prime_lut_identity

            prime_lut_identity(self)

    def _predict_for_lut(self, rgb_n3):
        result = _predict_direct(rgb_n3, self)
        return (
            result.astype(np.float32)
            if getattr(self, "fitting_version", None) == "ui-compatible-v1"
            else result
        )

    def clear_lut_cache(self):
        from .lut_cache import clear_lut_cache

        clear_lut_cache(self)


# CustomNN is load-only; it lives in _compat/estimators.py (see __getattr__ above).


def _predict_direct(RGB: np.ndarray, M: Regressor_Model) -> np.ndarray:
    """Predict RGB values without 3D LUT interpolation."""
    if getattr(M, "fitting_version", None) == "ui-compatible-v1":
        from ..ui_fitting import UIPredictor

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


def _ensure_lut(M: Regressor_Model) -> bool:
    """Synchronize one content-keyed owned LUT; transient state is not pickled."""
    from .lut_cache import ensure_lut

    return ensure_lut(M, build_3d_lut)


def predict_(RGB: np.ndarray, M: Regressor_Model) -> np.ndarray:
    """
    Predict RGB values using fitted model.

    Uses 3D LUT with trilinear interpolation when available (much faster
    than polynomial expansion on the full image).  Falls back to the
    original poly-expansion path otherwise.

    Args:
        RGB: Input RGB image (H, W, 3) or (N, 3)
        M: Fitted Regressor_Model

    Returns:
        np.ndarray: Predicted RGB values (same shape as input)
    """
    orig_shape = RGB.shape
    n_pixels = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else orig_shape[0]

    # ── Fast path: 3D LUT trilinear interpolation ──
    if n_pixels >= M.lut_min_pixels and _ensure_lut(M):
        result = trilinear_lut_apply(RGB, M._lut, M._lut_grid_size)
        return result.reshape(orig_shape)

    # ── Slow path: polynomial expansion + model.predict ──
    return _predict_direct(RGB, M)
