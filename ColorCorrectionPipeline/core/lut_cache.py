"""One owned LUT per regressor, synchronized and keyed by inference content.

This module never serializes executable objects or invokes a prediction callback
to fingerprint it. Unknown callback/estimator state is deliberately uncacheable.
"""
import hashlib
import json
import threading

import numpy as np

_LOCK_CREATION = threading.RLock()


def model_lock(model):
    with _LOCK_CREATION:
        lock = getattr(model, "_lut_lock", None)
        if lock is None:
            lock = threading.RLock()
            model._lut_lock = lock
        return lock


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
    import torch
    import torch.nn as nn
    if type(network) is not nn.Sequential or "forward" in vars(network):
        raise TypeError("unrecognized Torch network")
    from torch.nn.modules import module
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks", "_global_backward_pre_hooks"):
        if getattr(module, name, {}):
            raise TypeError("global Torch hooks make inference state opaque")
    layers = []
    for layer in (network, *tuple(network.children())):
        if any(getattr(layer,name,{}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")) or "forward" in vars(layer):
            raise TypeError("Torch hooks/overrides make inference state opaque")
    for layer in network:
        if type(layer) is nn.Linear:
            description = ["linear",layer.in_features,layer.out_features]
        elif type(layer) is nn.BatchNorm1d:
            description = ["batchnorm",layer.num_features,layer.eps,layer.affine,layer.track_running_stats]
        elif type(layer) in (nn.ReLU,nn.Tanh,nn.Sigmoid,nn.Identity):
            description = [type(layer).__name__]
        elif type(layer) is nn.Dropout:
            description = ["dropout",layer.p]
        else:
            raise TypeError("unrecognized Torch layer")
        layers.append(description)
    # Read whitelisted tensors directly; state_dict can execute user hooks.
    parameters = {}
    for index, layer in enumerate(network):
        for name in ("weight","bias","running_mean","running_var","num_batches_tracked"):
            tensor = getattr(layer,name,None)
            if tensor is not None:
                parameters[str(index)+"."+name] = [str(tensor.device),tensor.detach().cpu().numpy()]
    return layers, parameters


def inference_identity(model):
    """Return full inference/settings hash, or (None, reason) when opaque."""
    from .correction import Regressor_Model, CustomNN, _predict_direct
    from . import transforms
    from ..portable_inference import AffinePredictor, MLPPredictor, TorchPredictor
    from ..fitting import DeterministicTorchRegressor
    from ..ui_fitting import UIPredictor
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neural_network import MLPRegressor
    callback = getattr(model,"_lut_predict_fn",None)
    if callback is not None and not (getattr(callback,"__self__",None) is model and getattr(callback,"__func__",None) is Regressor_Model._predict_for_lut):
        return None,"custom prediction callback state is opaque; reuse disabled"
    estimator = model.model
    try:
        if estimator is None:
            return None,"no estimator; producer identity unavailable"
        if any(name in vars(estimator) for name in ("predict", "_decision_function", "_forward_pass_fast")) or (type(estimator) is UIPredictor and "predict_rgb" in vars(estimator)):
            return None,"instance prediction override; reuse disabled"
        if type(estimator) is UIPredictor:
            state = {"settings": dict(estimator.settings), "coefficients": estimator.coefficients,
                     "intercept": estimator.intercept, "weights": estimator.weights, "biases": estimator.biases}
        elif type(estimator) in (AffinePredictor,MLPPredictor,TorchPredictor):
            state = vars(estimator)
        elif type(estimator) is LinearRegression:
            state = {"coef":estimator.coef_,"intercept":estimator.intercept_}
        elif type(estimator) is PLSRegression:
            state = {name:getattr(estimator,name) for name in ("coef_","intercept_","_x_mean","_x_std","_y_mean","_y_std") if hasattr(estimator,name)}
        elif type(estimator) is MLPRegressor:
            state = {"coefs":estimator.coefs_,"biases":estimator.intercepts_,"activation":estimator.activation,
                     "output_activation":estimator.out_activation_,
                     "n_layers":estimator.n_layers_, "n_outputs":estimator.n_outputs_}
        elif type(estimator) in (CustomNN,DeterministicTorchRegressor):
            state = [_torch_state(estimator.model),str(estimator.device)]
        else:
            return None,"unknown estimator state; reuse disabled"
        settings = {"degree":model.degree,"grid_size":model._lut_grid_size,"use_lut":model.use_lut,
                    "feature_policy":getattr(model,"fitting_version","legacy-v1"),
                    "portable":getattr(model,"portable_inference",False),
                    "legacy_cuda":bool(transforms.is_cuda), "legacy_device":str(transforms.device_),
                    "reference_implementation":id(_predict_direct),"prediction_implementation":id(type(estimator).predict),
                    "helper_implementations":[id(getattr(type(estimator),name,None)) for name in ("_decision_function", "_forward_pass_fast")],
                    "estimator_type":type(estimator).__module__+"."+type(estimator).__name__}
        digest = hashlib.sha256()
        _update(digest,settings)
        _update(digest,state)
        return digest.hexdigest(),None
    except (TypeError,ValueError,AttributeError,RuntimeError) as exc:
        return None,type(exc).__name__+": "+str(exc)


def _valid_table(table, grid):
    values = np.asarray(table)
    if type(grid) is not int or not 2 <= grid <= 129 or values.shape != (grid,grid,grid,3) or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("LUT must contain finite floating values with matching grid size 2..129")
    return values


def _owned(table):
    values = np.ascontiguousarray(table)
    return np.frombuffer(values.tobytes(),dtype=values.dtype).reshape(values.shape)


def prime_lut_identity(model):
    """Adopt restored values without rebuilding; called after complete loading.

    Pre-version-9 artifacts cannot prove which producer generated their saved
    LUT. Exact saved values are retained and current producer content is pinned
    at adoption. Mutations after this point invalidate that derived cache.
    """
    with model_lock(model):
        identity, reason = inference_identity(model)
        if model._lut is not None:
            model._lut = _owned(_valid_table(model._lut,model._lut_grid_size))
        model._lut_table_ref = model._lut
        model._lut_identity = identity
        model._lut_failed_identity = None
        model._lut_cache_info = {"status":"restored" if model._lut is not None else "empty",
                                 "producer_identity":identity,"identity_unavailable_reason":reason,
                                 "retained_bytes":0 if model._lut is None else int(model._lut.nbytes),
                                 "entry_limit":1}


def persistence_lut(model):
    """Return only a current derived table for persistence, without rebuilding.

    Old tables without an identity retain historical adoption semantics. A
    versioned cache must prove its complete producer is unchanged. Never mutate
    the caller or execute its prediction callback while preparing saved state.
    """
    with model_lock(model):
        table = getattr(model,"_lut",None)
        if table is None or not getattr(model,"use_lut",False):
            return None
        try:
            _valid_table(table,model._lut_grid_size)
        except (TypeError,ValueError):
            return None
        if model.model is None and model._lut_predict_fn is None:
            return table
        if not hasattr(model,"_lut_identity"):
            return table
        identity,_ = inference_identity(model)
        if identity is not None and identity == model._lut_identity and table is getattr(model,"_lut_table_ref",None):
            return table
        return None


def ensure_lut(model, builder):
    with model_lock(model):
        if not model.use_lut:
            model._lut = None
            model._lut_identity = None
            model._lut_failed_identity = None
            model._lut_table_ref = None
            model._lut_cache_info = {"status":"disabled","retained_bytes":0,"entry_limit":1}
            return False
        if model.model is None and model._lut is not None and model._lut_predict_fn is None:
            try:
                model._lut = _owned(_valid_table(model._lut,model._lut_grid_size))
                model._lut_table_ref = model._lut
                model._lut_cache_info = {"status":"static-table","retained_bytes":int(model._lut.nbytes),"entry_limit":1}
                return True
            except ValueError as exc:
                model._lut = model._lut_table_ref = None
                model._lut_cache_info = {"status":"failed","reason":str(exc),"retained_bytes":0,"entry_limit":1}
                return False
        identity,reason = inference_identity(model)
        if model._lut is not None and not hasattr(model,"_lut_identity"):
            prime_lut_identity(model)
        if model._lut is not None and identity is not None and identity == getattr(model,"_lut_identity",None) and model._lut is getattr(model,"_lut_table_ref",None):
            # A cache cannot be mutated through a retained caller-owned array.
            model._lut_cache_info.update(status="hit",retained_bytes=int(model._lut.nbytes))
            return True
        if identity is not None and identity == getattr(model,"_lut_failed_identity",None):
            return False
        model._lut = None
        model._lut_table_ref = None
        if model._lut_predict_fn is None:
            model._lut_cache_info = {"status":"unavailable","reason":"no prediction callback","retained_bytes":0,"entry_limit":1}
            return False
        try:
            grid = model._lut_grid_size
            if type(grid) is not int or not 2 <= grid <= 129:
                raise ValueError("LUT grid size must be an integer in 2..129")
            table = _owned(_valid_table(builder(model._lut_predict_fn,grid_size=grid),grid))
            # Do not publish a table if caller mutation changed its producer
            # while native inference released the GIL. It can retry next call.
            after,_ = inference_identity(model)
            if identity is not None and identity != after:
                raise RuntimeError("estimator changed during LUT construction")
            model._lut = table
            model._lut_table_ref = table
            model._lut_identity = identity
            model._lut_failed_identity = None
            model._lut_cache_info = {"status":"built","producer_identity":identity,
                "identity_unavailable_reason":reason,"retained_bytes":int(table.nbytes),"entry_limit":1}
            return True
        except Exception as exc:
            model._lut = None
            model._lut_failed_identity = identity
            model._lut_cache_info = {"status":"failed","reason":type(exc).__name__+": "+str(exc),
                                     "producer_identity":identity,"retained_bytes":0,"entry_limit":1}
            import logging
            logging.getLogger(__name__).warning("Failed to build 3D LUT: %s",exc)
            return False


def clear_lut_cache(model):
    """Drop one derived table/failure memo; a subsequent prediction can rebuild."""
    with model_lock(model):
        model._lut = None
        model._lut_identity = None
        model._lut_failed_identity = None
        model._lut_table_ref = None
        model._lut_cache_info = {"status":"cleared","retained_bytes":0,"entry_limit":1}
