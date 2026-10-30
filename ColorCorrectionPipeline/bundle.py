"""Versioned, non-executable calibration bundles with bounded validation.

The format is a ZIP_STORED container of strict JSON and individual numeric NPY
arrays. No pickle, arbitrary imports, executable graph or estimator deserializer
is involved. Hashes detect integrity errors; they are not authenticity signatures.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
import zipfile

import numpy as np

from .portable_inference import AffinePredictor, MLPPredictor, TorchPredictor
from .ui_fitting import UIPredictor

FORMAT = "ColorCorrectionPipeline.calibration"
VERSION = 1
_ARRAY_NAME = re.compile(r"arrays/a[0-9]{4}\.npy\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FLOATS = {"<f4", "<f8", ">f4", ">f8"}
_ACTIVATIONS = {"identity", "relu", "tanh", "logistic"}
_MODEL_ATTRIBUTES = {"mtd", "degree", "max_iterations", "random_state", "tol", "verbose", "ncomp",
                     "param_search", "hidden_layers", "learning_rate", "batch_size", "use_batch_norm",
                     "patience", "dropout_rate", "optim_type", "use_lut", "lazy_lut", "lut_min_pixels",
                     "_lut_grid_size", "fitting_version", "fit_report"}


class BundleError(ValueError):
    """Unsupported, corrupt or resource-exceeding calibration bundle."""


@dataclass(frozen=True)
class BundleLimits:
    max_archive_bytes: int = 512 * 1024 * 1024
    max_total_array_bytes: int = 512 * 1024 * 1024
    max_array_bytes: int = 256 * 1024 * 1024
    max_manifest_bytes: int = 4 * 1024 * 1024
    max_arrays: int = 256
    max_header_bytes: int = 10000
    max_dimension: int = 32768
    max_network_layers: int = 256
    max_network_width: int = 4096
    max_network_parameters: int = 32 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("bundle resource limits must be positive integers")
        if self.max_arrays > 9999:
            raise ValueError("bundle format permits at most 9999 arrays")


def _limits(value):
    if value is None:
        return BundleLimits()
    if not isinstance(value, BundleLimits):
        raise TypeError("limits must be BundleLimits")
    return value


def _integer(value, label, low=1, high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        raise BundleError("%s must be an integer in [%d,%d]" % (label, low, high))
    return value


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise BundleError("invalid %s fields" % label)
    return value


def _plain(value, depth=0):
    if depth > 64:
        raise BundleError("JSON nesting limit exceeded")
    if value is None or type(value) in (str, bool, int):
        if isinstance(value, str) and len(value) > 65536:
            raise BundleError("JSON string exceeds supported length")
        return value
    if isinstance(value, np.generic):
        return _plain(value.item(), depth + 1)
    if type(value) is float:
        if not math.isfinite(value):
            raise BundleError("JSON metadata must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str or len(key) > 1024 for key in value):
            raise BundleError("JSON metadata keys must be bounded strings")
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise BundleError("non-JSON metadata is unsupported: " + type(value).__name__)


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(raw):
    try:
        value = json.loads(raw.decode("utf8"), object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(BundleError("nonfinite JSON literal")))
        return _plain(value)
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise BundleError("invalid strict JSON manifest: %s" % exc) from exc


class _Arrays:
    def __init__(self, limits):
        self.limits = limits
        self.records = []
        self.data = {}
        self.total = 0

    def add(self, value):
        values = np.asarray(value)
        if values.dtype.kind in "iu":
            if np.any(values > 2**53) or (values.dtype.kind == "i" and np.any(values < -(2**53))):
                raise BundleError("integer calibration values exceed exact float64 representation")
            values = values.astype(np.float64)
        if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
            raise BundleError("calibration arrays must be float32 or float64")
        values = np.ascontiguousarray(values, dtype="<f4" if values.dtype.itemsize == 4 else "<f8")
        _shape(list(values.shape), values.dtype.itemsize, self.limits)
        if not np.isfinite(values).all():
            raise BundleError("calibration arrays must be finite")
        if len(self.records) >= self.limits.max_arrays:
            raise BundleError("array count exceeds limit")
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, values, allow_pickle=False)
        raw = buffer.getvalue()
        self.total += len(raw)
        if len(raw) > self.limits.max_array_bytes or self.total > self.limits.max_total_array_bytes:
            raise BundleError("array resource limit exceeded")
        name = "arrays/a%04d.npy" % len(self.records)
        self.records.append({"path": name, "dtype": values.dtype.str, "shape": list(values.shape),
                             "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.data[name] = raw
        return name


def _shape(shape, itemsize, limits):
    if not isinstance(shape, list) or not 1 <= len(shape) <= 4:
        raise BundleError("array rank must be 1..4")
    count = 1
    for extent in shape:
        count *= _integer(extent, "array dimension", high=limits.max_dimension)
        if count * itemsize > limits.max_array_bytes:
            raise BundleError("declared array shape exceeds limit")
    return count


def _array(values, shape, label):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise BundleError("invalid %s shape/dtype/finiteness" % label)
    return values


def _torch_layers(model):
    import torch
    from .core.correction import CustomNN
    from .fitting import DeterministicTorchRegressor
    if type(model) not in (CustomNN, DeterministicTorchRegressor) or type(model.model) is not torch.nn.Sequential:
        raise BundleError("unsupported custom neural estimator; only known package Sequential models export")
    hook_names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    if any(getattr(model.model, name, {}) for name in hook_names) or "forward" in vars(model.model):
        raise BundleError("root neural graph hooks/overrides cannot be represented in a non-executable bundle")
    module_state = torch.nn.modules.module
    if any(getattr(module_state, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks", "_global_backward_pre_hooks")):
        raise BundleError("global Torch hooks are unsupported for portable export")
    result = []
    for layer in model.model:
        if any(getattr(layer, name, {}) for name in hook_names) or "forward" in vars(layer):
            raise BundleError("neural layer hooks cannot be represented in a non-executable bundle")
        if type(layer) is torch.nn.Linear:
            result.append({"type": "linear", "weight": layer.weight.detach().cpu().numpy(),
                           "bias": None if layer.bias is None else layer.bias.detach().cpu().numpy()})
        elif type(layer) is torch.nn.BatchNorm1d:
            if not layer.track_running_stats or layer.running_mean is None or layer.running_var is None:
                raise BundleError("portable BatchNorm requires recorded running statistics")
            result.append({"type": "batchnorm", "eps": float(layer.eps),
                           "running_mean": layer.running_mean.detach().cpu().numpy(),
                           "running_var": layer.running_var.detach().cpu().numpy(),
                           "weight": None if layer.weight is None else layer.weight.detach().cpu().numpy(),
                           "bias": None if layer.bias is None else layer.bias.detach().cpu().numpy()})
        elif type(layer) in (torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid, torch.nn.Identity):
            result.append({"type": {torch.nn.ReLU: "relu", torch.nn.Tanh: "tanh",
                                    torch.nn.Sigmoid: "sigmoid", torch.nn.Identity: "identity"}[type(layer)]})
        elif type(layer) is torch.nn.Dropout:
            result.append({"type": "dropout", "p": float(layer.p)})
        else:
            raise BundleError("unsupported neural layer: " + type(layer).__name__)
    return result


def _estimator_export(model, arrays):
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neural_network import MLPRegressor
    if any(name in vars(model) for name in ("predict", "_decision_function", "_forward_pass_fast")) or (type(model) is UIPredictor and "predict_rgb" in vars(model)):
        raise BundleError("custom estimator inference overrides cannot be represented in a non-executable bundle")
    if type(model) is UIPredictor:
        return {"kind": "ui-compatible-v1", "settings": _plain(dict(model.settings)),
                "coefficients": None if model.coefficients is None else arrays.add(model.coefficients),
                "intercept": None if model.intercept is None else arrays.add(model.intercept),
                "weights": [arrays.add(w) for w in model.weights], "biases": [arrays.add(b) for b in model.biases]}
    if type(model) is AffinePredictor:
        return {"kind": model.kind, "coefficients": arrays.add(model.coefficients), "intercept": arrays.add(model.intercept),
                "center": None if model.center is None else arrays.add(model.center),
                "scale": None if model.scale is None else arrays.add(model.scale)}
    if type(model) is LinearRegression:
        coef = np.asarray(model.coef_)
        intercept = np.asarray(model.intercept_)
        if intercept.ndim == 0:
            intercept = np.repeat(intercept.reshape(1), 3)
        return {"kind": "linear", "coefficients": arrays.add(coef), "intercept": arrays.add(intercept), "center": None, "scale": None}
    if type(model) is PLSRegression:
        center, std = np.asarray(model._x_mean), np.asarray(model._x_std)
        coef = np.asarray(model.coef_)
        if coef.shape == (len(center), 3):
            coef = coef.T
        offset = np.asarray(getattr(model, "intercept_", model._y_mean))
        # PLS coef orientation/scaling changed across sklearn releases. Detect the
        # supported numerical form at export, then persist its explicit equation.
        probe = np.random.default_rng(713).normal(size=(9, len(center)))
        expected = model.predict(probe)
        options = [(None, offset), (std, offset), (std, np.asarray(model._y_mean))]
        for scale, intercept in options:
            candidate = AffinePredictor(coef, intercept, center=center, scale=scale, kind="pls")
            if np.allclose(candidate.predict(probe), expected, rtol=2e-13, atol=2e-13):
                return {"kind": "pls", "coefficients": arrays.add(coef), "intercept": arrays.add(intercept),
                        "center": arrays.add(center), "scale": None if scale is None else arrays.add(scale)}
        raise BundleError("unsupported PLS coefficient/scaling convention")
    if type(model) in (MLPRegressor, MLPPredictor):
        weights = model.coefs_ if type(model) is MLPRegressor else model.weights
        biases = model.intercepts_ if type(model) is MLPRegressor else model.biases
        activation = model.activation
        output = model.out_activation_ if type(model) is MLPRegressor else model.output_activation
        return {"kind": "mlp", "weights": [arrays.add(value) for value in weights],
                "biases": [arrays.add(value) for value in biases], "activation": activation, "output_activation": output}
    layers = model.layers if type(model) is TorchPredictor else _torch_layers(model)
    return {"kind": "torch-eval", "layers": [{key: arrays.add(value) if isinstance(value, np.ndarray) else value
                                                for key, value in layer.items()} for layer in layers]}


def _attrs(model):
    result = _plain({name: deepcopy(value) for name, value in vars(model).items() if name in _MODEL_ATTRIBUTES})
    _validate_attrs(result)
    return result


def _validate_attrs(attrs):
    if not isinstance(attrs, dict) or set(attrs) - _MODEL_ATTRIBUTES or not {"degree", "mtd", "use_lut", "lazy_lut", "lut_min_pixels", "_lut_grid_size"}.issubset(attrs):
        raise BundleError("invalid regressor attributes")
    _integer(attrs["degree"], "polynomial degree", low=0 if attrs.get("fitting_version", "legacy-v1") == "legacy-v1" else 1, high=8)
    if attrs["mtd"].lower() not in ("linear", "linearcc", "pls", "nn", "custom"):
        raise BundleError("unsupported regressor method")
    for key in ("use_lut", "lazy_lut"):
        if type(attrs[key]) is not bool:
            raise BundleError("LUT controls must be boolean")
    _integer(attrs["lut_min_pixels"], "LUT threshold")
    _integer(attrs["_lut_grid_size"], "LUT grid", low=2, high=129)
    if attrs.get("fitting_version", "legacy-v1") not in ("legacy-v1", "reproducible-v1", "ui-compatible-v1"):
        raise BundleError("unsupported fitting version")


def _encode(models, limits):
    from importlib.metadata import version as package_version
    from .__version__ import __version__
    from .models import MyModels
    from .stage_policies import MonotonicToneModel
    from .core.correction import Regressor_Model
    if type(models) is not MyModels:
        raise TypeError("expected MyModels")
    arrays = _Arrays(limits)
    stages = {"FFC": None, "GC": None, "WB": None, "CC": None}
    if models.model_ffc is not None:
        stages["FFC"] = {"kind": "spatial-Lab-gain-v1", "array": arrays.add(models.model_ffc)}
    if models.model_gc is not None:
        if type(models.model_gc) is MonotonicToneModel:
            stages["GC"] = {"kind": "neutral-pchip-v1", "measured": arrays.add(models.model_gc.measured_knots), "target": arrays.add(models.model_gc.target_knots)}
        elif isinstance(models.model_gc, np.ndarray):
            stages["GC"] = {"kind": "legacy-polynomial-v1", "coefficients": arrays.add(models.model_gc)}
        else:
            raise BundleError("unsupported gamma model")
    if models.model_wb is not None:
        stages["WB"] = {"kind": "RGB-matrix-v1", "matrix": arrays.add(models.model_wb)}
    cc = models.model_cc
    if cc is not None:
        if not isinstance(cc, tuple) or len(cc) != 3 or not isinstance(cc[1], dict):
            raise BundleError("CC must be a known model/options/method tuple")
        if cc[2] == "conv":
            stages["CC"] = {"kind": "conventional-v1", "matrix": arrays.add(cc[0]), "options": _plain(cc[1])}
        elif cc[2] == "ours" and type(cc[0]) is Regressor_Model:
            fitted = cc[0]
            from .core.lut_cache import persistence_lut
            table = persistence_lut(fitted)
            stages["CC"] = {"kind": "regressor-v1", "options": _plain(cc[1]), "attributes": _attrs(fitted),
                            "feature_order": "ui-compatible-v1/linear-rgb/cpu-f64" if getattr(fitted, "fitting_version", None) == "ui-compatible-v1" else "sklearn-polynomial-features/include-bias/c-order/cpu-f64/v1",
                            "estimator": _estimator_export(fitted.model, arrays),
                            "lut": None if table is None else arrays.add(table)}
        else:
            raise BundleError("unsupported CC family; no opaque serialization fallback exists")
    # Validate only after whitelisting estimator export, avoiding an unsupported
    # user object's prediction callback in the modern metadata consistency check.
    MyModels._validated_state(models)
    manifest = {"format": FORMAT, "version": VERSION, "metadata": _plain(models.metadata),
                "interpretation": "recorded" if models.metadata else "legacy-v1; capture/reference provenance unknown",
                "exporter": {"package_version": __version__, "numpy": str(np.__version__),
                             "scikit-learn": package_version("scikit-learn"), "torch": package_version("torch")},
                "stages": stages, "arrays": arrays.records}
    manifest["content_sha256"] = hashlib.sha256(_json(manifest)).hexdigest()
    raw = _json(manifest)
    if len(raw) > limits.max_manifest_bytes:
        raise BundleError("manifest exceeds limit")
    return manifest, raw, arrays.data


def _reference(value, arrays):
    if type(value) is not str or value not in arrays:
        raise BundleError("unknown array reference")
    return arrays[value]


def _estimator_load(record, arrays, features, limits):
    if not isinstance(record, dict):
        raise BundleError("invalid estimator record")
    kind = record.get("kind")
    if kind == "ui-compatible-v1":
        _keys(record, {"kind", "settings", "coefficients", "intercept", "weights", "biases"}, "UI estimator")
        weights, biases = record["weights"], record["biases"]
        if not isinstance(weights, list) or not isinstance(biases, list) or len(weights) != len(biases) or len(weights) > min(5, limits.max_network_layers):
            raise BundleError("invalid UI network layer count")
        ws = [_reference(w, arrays) for w in weights]
        bs = [_reference(b, arrays) for b in biases]
        if any(w.ndim != 2 or max(w.shape) > limits.max_network_width for w in ws) or sum(w.size+b.size for w,b in zip(ws,bs)) > limits.max_network_parameters:
            raise BundleError("UI network exceeds resource limits")
        try:
            return UIPredictor(record["settings"],
                None if record["coefficients"] is None else _reference(record["coefficients"], arrays),
                None if record["intercept"] is None else _reference(record["intercept"], arrays), ws, bs)
        except (TypeError, ValueError, KeyError) as exc:
            raise BundleError("invalid UI estimator: %s" % exc) from exc
    if kind in ("linear", "pls"):
        _keys(record, {"kind", "coefficients", "intercept", "center", "scale"}, "affine estimator")
        coef = _array(_reference(record["coefficients"], arrays), (3, features), "coefficients")
        intercept = _array(_reference(record["intercept"], arrays), (3,), "intercept")
        center = None if record["center"] is None else _array(_reference(record["center"], arrays), (features,), "center")
        scale = None if record["scale"] is None else _array(_reference(record["scale"], arrays), (features,), "scale")
        if (kind == "linear" and (center is not None or scale is not None)) or (kind == "pls" and center is None) or (scale is not None and np.any(scale <= 0)):
            raise BundleError("invalid affine normalization convention")
        return AffinePredictor(coef, intercept, center=center, scale=scale, kind=kind)
    if kind == "mlp":
        _keys(record, {"kind", "weights", "biases", "activation", "output_activation"}, "MLP")
        if record["activation"] not in _ACTIVATIONS or record["output_activation"] not in (_ACTIVATIONS | {"exponential"}):
            raise BundleError("unsupported MLP activation")
        weights, biases = record["weights"], record["biases"]
        if not isinstance(weights, list) or not isinstance(biases, list) or not 1 <= len(weights) == len(biases) <= limits.max_network_layers:
            raise BundleError("invalid MLP layer count")
        current, ws, bs, parameters = features, [], [], 0
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            weight, bias = _reference(weight, arrays), _reference(bias, arrays)
            if weight.ndim != 2 or not 1 <= weight.shape[1] <= limits.max_network_width:
                raise BundleError("invalid MLP width")
            _array(weight, (current, weight.shape[1]), "MLP weights")
            _array(bias, (weight.shape[1],), "MLP bias")
            parameters += weight.size + bias.size
            if parameters > limits.max_network_parameters:
                raise BundleError("expanded network parameter count exceeds limit")
            current = weight.shape[1]
            ws.append(weight); bs.append(bias)
        if current != 3:
            raise BundleError("MLP must predict three RGB channels")
        return MLPPredictor(ws, bs, activation=record["activation"], output_activation=record["output_activation"])
    if kind == "torch-eval":
        _keys(record, {"kind", "layers"}, "Torch eval graph")
        layers = record["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= limits.max_network_layers:
            raise BundleError("invalid Torch layer count")
        current, loaded, parameters = features, [], 0
        for layer in layers:
            if not isinstance(layer, dict):
                raise BundleError("invalid Torch layer")
            typ = layer.get("type")
            item = {"type": typ}
            if typ == "linear":
                _keys(layer, {"type", "weight", "bias"}, "Torch Linear")
                weight = _reference(layer["weight"], arrays)
                if weight.ndim != 2 or not 1 <= weight.shape[0] <= limits.max_network_width:
                    raise BundleError("invalid Torch Linear width")
                _array(weight, (weight.shape[0], current), "Torch Linear weights")
                bias = None if layer["bias"] is None else _array(_reference(layer["bias"], arrays), (weight.shape[0],), "Torch bias")
                if weight.dtype.itemsize != 4 or (bias is not None and bias.dtype.itemsize != 4):
                    raise BundleError("package Torch inference requires float32 parameters")
                current = weight.shape[0]
                item.update(weight=weight, bias=bias)
            elif typ == "batchnorm":
                _keys(layer, {"type", "eps", "running_mean", "running_var", "weight", "bias"}, "Torch BatchNorm")
                eps = layer["eps"]
                if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0:
                    raise BundleError("invalid BatchNorm epsilon")
                item["eps"] = eps
                for key in ("running_mean", "running_var", "weight", "bias"):
                    item[key] = None if layer[key] is None else _array(_reference(layer[key], arrays), (current,), "BatchNorm " + key)
                    if item[key] is not None and item[key].dtype.itemsize != 4:
                        raise BundleError("BatchNorm parameters require float32")
                if item["running_mean"] is None or item["running_var"] is None or np.any(item["running_var"] < 0) or np.any(item["running_var"] + eps <= 0) or (item["weight"] is None) != (item["bias"] is None):
                    raise BundleError("invalid BatchNorm eval statistics")
            elif typ == "dropout":
                _keys(layer, {"type", "p"}, "Dropout")
                p = layer["p"]
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise BundleError("invalid dropout probability")
                item["p"] = p
            elif typ in ("identity", "relu", "tanh", "sigmoid"):
                _keys(layer, {"type"}, "activation")
            else:
                raise BundleError("unsupported Torch operation")
            parameters += sum(value.size for value in item.values() if isinstance(value, np.ndarray))
            if parameters > limits.max_network_parameters:
                raise BundleError("expanded network parameter count exceeds limit")
            loaded.append(item)
        if current != 3 or loaded[-1]["type"] != "linear":
            raise BundleError("package Torch graph must end with three-channel Linear output")
        return TorchPredictor(loaded)
    raise BundleError("unsupported estimator kind")


def _conventional_options(options):
    import colour
    if not isinstance(options, dict) or set(options) - {"method", "degree", "root_polynomial_expansion", "terms"}:
        raise BundleError("unsupported conventional CC options")
    method = options.get("method", "Finlayson 2015")
    if method not in ("Finlayson 2015", "Cheung 2004", "Vandermonde"):
        raise BundleError("unsupported conventional feature basis")
    if "degree" in options:
        _integer(options["degree"], "conventional degree", high=8)
    root = options.get("root_polynomial_expansion")
    if root is not None and type(root) is not bool:
        raise BundleError("root polynomial expansion must be boolean or legacy None")
    if options.get("terms") is not None:
        _integer(options["terms"], "Cheung terms", high=35)
    probe = np.array([[.17, .31, .53]])
    # The known library feature expander defines each basis; artifact values
    # cannot name/import a function. Identity application below validates width.
    from .conventional import conventional_basis
    basis = conventional_basis(probe, **options)
    return basis.shape[1]


def _decode(manifest, arrays, limits):
    from .models import MyModels
    from .stage_policies import MonotonicToneModel
    from .core.correction import Regressor_Model
    result = MyModels()
    result.metadata = deepcopy(manifest["metadata"])
    stages = _keys(manifest["stages"], {"FFC", "GC", "WB", "CC"}, "stage set")
    ffc, gc, wb, cc = (stages[name] for name in ("FFC", "GC", "WB", "CC"))
    if ffc is not None:
        _keys(ffc, {"kind", "array"}, "FFC")
        values = _reference(ffc["array"], arrays)
        if ffc["kind"] != "spatial-Lab-gain-v1" or values.ndim != 2 or np.any(values <= 0):
            raise BundleError("invalid spatial FFC gain")
        result.model_ffc = values
    if gc is not None:
        if not isinstance(gc, dict):
            raise BundleError("invalid GC")
        if gc.get("kind") == "legacy-polynomial-v1":
            _keys(gc, {"kind", "coefficients"}, "polynomial GC")
            values = _reference(gc["coefficients"], arrays)
            if values.ndim != 1 or not 1 <= len(values) <= 33:
                raise BundleError("invalid polynomial GC coefficient count")
            result.model_gc = values
        elif gc.get("kind") == "neutral-pchip-v1":
            _keys(gc, {"kind", "measured", "target"}, "neutral GC")
            result.model_gc = MonotonicToneModel(_reference(gc["measured"], arrays), _reference(gc["target"], arrays))
        else:
            raise BundleError("unsupported GC kind")
    if wb is not None:
        _keys(wb, {"kind", "matrix"}, "WB")
        if wb["kind"] != "RGB-matrix-v1":
            raise BundleError("unsupported WB kind")
        result.model_wb = _array(_reference(wb["matrix"], arrays), (3, 3), "WB")
    if cc is not None:
        if not isinstance(cc, dict):
            raise BundleError("invalid CC")
        if cc.get("kind") == "conventional-v1":
            _keys(cc, {"kind", "matrix", "options"}, "conventional CC")
            features = _conventional_options(cc["options"])
            matrix = _array(_reference(cc["matrix"], arrays), (3, features), "conventional CCM")
            result.model_cc = (matrix, deepcopy(cc["options"]), "conv")
        elif cc.get("kind") == "regressor-v1":
            _keys(cc, {"kind", "options", "attributes", "feature_order", "estimator", "lut"}, "regressor CC")
            ui_policy = cc["attributes"].get("fitting_version") == "ui-compatible-v1" if isinstance(cc["attributes"], dict) else False
            expected_order = "ui-compatible-v1/linear-rgb/cpu-f64" if ui_policy else "sklearn-polynomial-features/include-bias/c-order/cpu-f64/v1"
            if cc["feature_order"] != expected_order or not isinstance(cc["options"], dict):
                raise BundleError("unsupported polynomial feature order/options")
            attrs = cc["attributes"]
            _validate_attrs(attrs)
            model = Regressor_Model()
            model.__dict__.update(deepcopy(attrs))
            model.portable_inference = True
            features = math.comb(model.degree + 3, 3)
            model.model = _estimator_load(cc["estimator"], arrays, features, limits)
            expected_kind = "ui-compatible-v1" if ui_policy else {"linear": "linear", "linearcc": "linear", "pls": "pls", "nn": "mlp", "custom": "torch-eval"}[model.mtd.lower()]
            if cc["estimator"].get("kind") != expected_kind:
                raise BundleError("estimator contradicts recorded method")
            model._lut = None
            if cc["lut"] is not None:
                model._lut = _array(_reference(cc["lut"], arrays), (model._lut_grid_size,) * 3 + (3,), "LUT")
            model._lut_predict_fn = None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                probe = model.model.predict_rgb(np.array([[0.,0.,0.],[1.,1.,1.],[.5,.5,.5]])) if ui_policy else model.model.predict(np.vstack((np.zeros(features), np.ones(features), np.full(features, .5))))
            if np.shape(probe) != (3, 3) or not np.isfinite(probe).all():
                raise BundleError("estimator has invalid inference shape or nonfinite probe output")
            result.model_cc = (model, deepcopy(cc["options"]), "ours")
        else:
            raise BundleError("unsupported CC kind")
    MyModels._validated_state(result)
    if result.model_cc is not None and result.model_cc[2] == "ours":
        model = result.model_cc[0]
        model._lut_predict_fn = model._predict_for_lut if model.use_lut else None
        if model._lut is not None:
            from .core.lut_cache import prime_lut_identity
            prime_lut_identity(model)
    result.bundle_digest = manifest["content_sha256"]
    result.bundle_interpretation = manifest["interpretation"]
    return result


def _preflight(raw, limits):
    if len(raw) > limits.max_archive_bytes or len(raw) < 22:
        raise BundleError("archive size exceeds limit or is truncated")
    # Reject ZIP64/multi-disk/comments before ZipFile allocates its central index.
    signature, disk, cd_disk, count_disk, count, size, offset, comment = struct.unpack("<4s4H2IH", raw[-22:])
    if signature != b"PK\x05\x06" or disk or cd_disk or count_disk != count or comment or not 1 <= count <= limits.max_arrays + 1 or size > (limits.max_arrays + 1) * 512 or offset + size != len(raw) - 22:
        raise BundleError("unsupported ZIP directory structure or entry limit")
    return offset


def _read_array_payload(raw, record, limits):
    dtype = np.dtype(record["dtype"])
    count = _shape(record["shape"], dtype.itemsize, limits)
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise BundleError("array content hash mismatch")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise BundleError("unsupported NPY version")
    position = stream.tell()
    length_size = 2 if version == (1, 0) else 4
    field = stream.read(length_size)
    if len(field) != length_size:
        raise BundleError("truncated NPY header length")
    header_length = struct.unpack("<H" if length_size == 2 else "<I", field)[0]
    if not 1 <= header_length <= limits.max_header_bytes or position + length_size + header_length > len(raw):
        raise BundleError("NPY header length exceeds bounded payload")
    stream.seek(position)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, stored_dtype = reader(stream, max_header_size=limits.max_header_bytes)
    if stored_dtype.fields is not None or stored_dtype.subdtype is not None or stored_dtype.hasobject:
        raise BundleError("NPY dtype must be an unstructured primitive float")
    if list(shape) != record["shape"] or stored_dtype.str != dtype.str or fortran or len(raw) - stream.tell() != count * dtype.itemsize:
        raise BundleError("NPY payload contradicts declared shape/dtype/size")
    values = np.frombuffer(raw, dtype=dtype, count=count, offset=stream.tell()).reshape(shape)
    if not np.isfinite(values).all():
        raise BundleError("nonfinite array values")
    return values


def _drop_transient_callbacks(models):
    # Read-back validation candidates have no future caller. Disconnect the bound
    # self callback so large LUT/NN buffers do not await cyclic-GC reclamation.
    cc = models.model_cc
    if cc is not None and cc[2] == "ours":
        cc[0]._lut_predict_fn = None


def loads_models(data, *, limits=None):
    """Validate a bounded bundle byte string and return a fresh model candidate."""
    limits = _limits(limits)
    if not isinstance(data, bytes):
        raise TypeError("bundle data must be bytes")
    central = _preflight(data, limits)
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(set(names)) != len(names) or "manifest.json" not in names or len(names) > limits.max_arrays + 1:
                raise BundleError("duplicate/missing ZIP entries")
            position = 0
            for info in sorted(infos, key=lambda item: item.header_offset):
                if info.filename != "manifest.json" and not _ARRAY_NAME.fullmatch(info.filename):
                    raise BundleError("unexpected ZIP entry/path")
                mode = info.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG) or info.flag_bits != 0 or info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size or info.extra or info.comment or info.extract_version > 20:
                    raise BundleError("unsupported ZIP entry type/compression/encryption")
                maximum = limits.max_manifest_bytes if info.filename == "manifest.json" else limits.max_array_bytes
                if not 1 <= info.file_size <= maximum or info.header_offset != position:
                    raise BundleError("ZIP member size/overlap/prefix is invalid")
                header = data[position:position + 30]
                if len(header) != 30:
                    raise BundleError("truncated ZIP local header")
                fields = struct.unpack("<4s5H3I2H", header)
                name_length, extra_length = fields[-2:]
                name = info.filename.encode("ascii")
                if fields[0] != b"PK\x03\x04" or fields[2] != 0 or fields[3] != 0 or fields[6] != info.CRC or fields[7] != info.file_size or fields[8] != info.file_size or extra_length or name_length != len(name) or data[position+30:position+30+name_length] != name:
                    raise BundleError("ZIP local/directory metadata disagree")
                position += 30 + name_length + info.file_size
            if position != central:
                raise BundleError("ZIP has unclaimed/overlapping member bytes")
            manifest = _strict_json(archive.read("manifest.json"))
            _keys(manifest, {"format", "version", "metadata", "interpretation", "exporter", "stages", "arrays", "content_sha256"}, "manifest")
            if manifest["format"] != FORMAT or type(manifest["version"]) is not int or manifest["version"] != VERSION:
                raise BundleError("unsupported calibration bundle format/version")
            if not isinstance(manifest["metadata"], dict) or manifest["interpretation"] != ("recorded" if manifest["metadata"] else "legacy-v1; capture/reference provenance unknown"):
                raise BundleError("invalid calibration interpretation")
            exporter = _keys(manifest["exporter"], {"package_version", "numpy", "scikit-learn", "torch"}, "exporter provenance")
            if any(type(value) is not str or not 1 <= len(value) <= 128 for value in exporter.values()):
                raise BundleError("invalid exporter version labels")
            identity = manifest["content_sha256"]
            unclaimed = dict(manifest); unclaimed.pop("content_sha256")
            if type(identity) is not str or not _HASH.fullmatch(identity) or hashlib.sha256(_json(unclaimed)).hexdigest() != identity:
                raise BundleError("manifest content identity mismatch")
            records = manifest["arrays"]
            if not isinstance(records, list) or len(records) > limits.max_arrays:
                raise BundleError("invalid array table")
            expected = {"manifest.json"}
            arrays, total = {}, 0
            for index, record in enumerate(records):
                _keys(record, {"path", "dtype", "shape", "bytes", "sha256"}, "array record")
                name = record["path"]
                if name != "arrays/a%04d.npy" % index or name in expected:
                    raise BundleError("invalid canonical array identity")
                expected.add(name)
                if record["dtype"] not in _FLOATS:
                    raise BundleError("unsupported array dtype")
                dtype = np.dtype(record["dtype"])
                _shape(record["shape"], dtype.itemsize, limits)
                size = _integer(record["bytes"], "array bytes", high=limits.max_array_bytes)
                total += size
                if total > limits.max_total_array_bytes:
                    raise BundleError("total array bytes exceed limit")
                if name not in names or archive.getinfo(name).file_size != size or type(record["sha256"]) is not str or not _HASH.fullmatch(record["sha256"]):
                    raise BundleError("array record/header mismatch")
                raw = archive.read(name)
                arrays[name] = _read_array_payload(raw, record, limits)
            if set(names) != expected:
                raise BundleError("archive contains undeclared array members")
            return _decode(manifest, arrays, limits)
    except BundleError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, EOFError, OverflowError, struct.error, zipfile.BadZipFile) as exc:
        raise BundleError("invalid calibration bundle: %s" % exc) from exc


def dumps_models(models, *, limits=None):
    """Serialize supported inference parameters, never executable training state."""
    limits = _limits(limits)
    try:
        _, manifest, arrays = _encode(models, limits)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            for name, raw in [("manifest.json", manifest)] + list(arrays.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, raw)
        result = buffer.getvalue()
        validation = loads_models(result, limits=limits)
        _drop_transient_callbacks(validation)
        return result
    except BundleError:
        raise
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise BundleError("unsupported calibration export: %s" % exc) from exc


def digest_models(models, *, limits=None):
    """Canonical content identity; hashes are integrity checks, not signatures."""
    limits = _limits(limits)
    try:
        manifest, _, data = _encode(models, limits)
        arrays = {record["path"]: _read_array_payload(data[record["path"]], record, limits) for record in manifest["arrays"]}
        validation = _decode(manifest, arrays, limits)
        _drop_transient_callbacks(validation)
        return manifest["content_sha256"]
    except BundleError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
        raise BundleError("invalid calibration identity: %s" % exc) from exc


def snapshot_models(models, *, limits=None):
    """Owned, independently reconstructed inference snapshot for a batch/run."""
    return loads_models(dumps_models(models, limits=limits), limits=limits)


def load_bundle(path, *, limits=None):
    limits = _limits(limits)
    path = Path(path)
    details = path.stat()
    if not stat.S_ISREG(details.st_mode):
        raise BundleError("bundle input must be a regular file")
    if details.st_size > limits.max_archive_bytes:
        raise BundleError("archive file exceeds limit")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > limits.max_archive_bytes:
            raise BundleError("opened bundle is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(limits.max_archive_bytes + 1)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return loads_models(raw, limits=limits)


def save_bundle(models, path, *, limits=None):
    """Read-back-validated atomic replacement; failure preserves old destination."""
    limits = _limits(limits)
    destination = Path(path)
    raw = dumps_models(models, limits=limits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix="." + destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        validation = load_bundle(temporary, limits=limits)
        _drop_transient_callbacks(validation)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
