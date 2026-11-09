"""Grouped request adapters over Config's existing flat numerical recipe.

Groups are owned immutable snapshots, never a second mutable configuration.
The leaf map below owns both normalization and the inverse JSON presentation.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
import numbers
import os
from types import MappingProxyType

import numpy as np


class _Unset:
    def __repr__(self):
        return "UNSET"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_unset, ())


def _unset():
    return UNSET


UNSET = _Unset()
GROUP_NAMES = ("steps", "input", "chart", "reference", "color", "numerics", "observations", "persistence")
GROUPED_LAYOUT = "grouped-v1"


def _freeze(value):
    if isinstance(value, Mapping):
        return _FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("grouped options cannot contain object arrays")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported grouped option value: " + type(value).__name__)


def _thaw(value, *, json_ready=False):
    if isinstance(value, Mapping):
        return {key: _thaw(item, json_ready=json_ready) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item, json_ready=json_ready) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if json_ready else value.copy()
    if json_ready and isinstance(value, os.PathLike):
        return os.fspath(value)
    return deepcopy(value)


class _FrozenMap(Mapping):
    """A deep-owned mapping that also survives Config deepcopy and pickle."""
    __slots__ = ("_data",)

    def __init__(self, value=()):
        if not isinstance(value, Mapping):
            value = dict(value)
        if any(not isinstance(key, str) for key in value):
            raise TypeError("grouped option keys must be strings")
        object.__setattr__(self, "_data", MappingProxyType({key: _freeze(item) for key, item in value.items()}))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __setattr__(self, name, value):
        raise TypeError("grouped option snapshots are immutable")

    def __deepcopy__(self, memo):
        return type(self)(_thaw(self))

    def __reduce__(self):
        return (type(self), (_thaw(self),))

    def __repr__(self):
        return repr(dict(self))


class _Step:
    def __post_init__(self):
        if self.enabled is not UNSET and type(self.enabled) is not bool:
            raise TypeError(type(self).__name__ + ".enabled must be boolean")
        for item in fields(self):
            if item.name == "enabled":
                continue
            value = getattr(self, item.name)
            if not isinstance(value, Mapping):
                raise TypeError(type(self).__name__ + "." + item.name + " must be a mapping")
            object.__setattr__(self, item.name, _FrozenMap(value))

    def to_dict(self):
        return {item.name: _thaw(getattr(self, item.name)) for item in fields(self)
                if getattr(self, item.name) is not UNSET and
                (item.name == "enabled" or bool(getattr(self, item.name)))}


@dataclass(frozen=True)
class FFCOptions(_Step):
    """FFC enable switch, surface fit, white plane, quality and reporting."""
    enabled: object = UNSET
    fit: Mapping = field(default_factory=dict)
    white_plane: Mapping = field(default_factory=dict)
    quality: Mapping = field(default_factory=dict)
    reporting: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class GCOptions(_Step):
    """Monotonic neutral-lightness fitting guards and reporting."""
    enabled: object = UNSET
    quality: Mapping = field(default_factory=dict)
    reporting: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class WBOptions(_Step):
    """Neutral WB headroom, eligibility/gain guards and reporting."""
    enabled: object = UNSET
    fit: Mapping = field(default_factory=dict)
    quality: Mapping = field(default_factory=dict)
    reporting: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class CCOptions(_Step):
    """CC solver/features, conditional NN/PLS controls, LUT and validation."""
    enabled: object = UNSET
    fit: Mapping = field(default_factory=dict)
    apply: Mapping = field(default_factory=dict)
    validation: Mapping = field(default_factory=dict)
    reporting: Mapping = field(default_factory=dict)


_STEP_TYPES = {"ffc": FFCOptions, "gc": GCOptions, "wb": WBOptions, "cc": CCOptions}


@dataclass(frozen=True)
class Steps:
    """Sparse immutable requests, or a fresh snapshot of Config's flat state."""
    ffc: FFCOptions = field(default_factory=FFCOptions)
    gc: GCOptions = field(default_factory=GCOptions)
    wb: WBOptions = field(default_factory=WBOptions)
    cc: CCOptions = field(default_factory=CCOptions)

    def __post_init__(self):
        for name, kind in _STEP_TYPES.items():
            value = getattr(self, name)
            if isinstance(value, Mapping):
                try:
                    value = kind(**dict(value))
                except TypeError as error:
                    raise TypeError("steps." + name + ": " + str(error)) from error
                object.__setattr__(self, name, value)
            if not isinstance(value, kind):
                raise TypeError("steps." + name + " must be " + kind.__name__ + " or a mapping")

    def to_dict(self):
        return {name: value for name in _STEP_TYPES if (value := getattr(self, name).to_dict())}


# Paths are the sole translation registry. Reverse presentation consumes the
# same map, so a renamed control cannot silently diverge from serialization.
_LEAVES = {}


def _map(group, names, target, aliases=None):
    aliases = aliases or {}
    for name in names.split():
        _LEAVES[tuple(group.split(".")) + (name,)] = (target, aliases.get(name, name)) if target else (aliases[name],)


for _name in _STEP_TYPES:
    _LEAVES[("steps", _name, "enabled")] = ("do_" + _name,)
    _map("steps." + _name + ".reporting", "show get_deltaE", _name.upper() + "_kwargs")
_map("steps.ffc.fit", "method degree bins smooth_window", "FFC_kwargs", {"method": "fit_method"})
_map("steps.ffc.white_plane", "model_path manual_crop crop_rect", "FFC_kwargs")
_map("steps.ffc.quality", "safe_mode missing_plane min_signal max_saturated_fraction min_crop_fraction min_gain max_gain", "FFC_kwargs")
_map("steps.gc.quality", "min_neutrals min_span duplicate_tolerance target_tolerance", "stage_policy",
     {key: "gc_" + key for key in ("min_neutrals", "min_span", "duplicate_tolerance", "target_tolerance")})
_map("steps.wb.fit", "headroom", "stage_policy", {"headroom": "wb_headroom"})
_map("steps.wb.quality", "min_signal max_gain", "stage_policy", {"min_signal": "wb_min_signal", "max_gain": "wb_max_gain"})
_map("steps.cc.fit", "model basis degree include_combinations device random_state", "CC_kwargs")
_map("steps.cc.fit.nn", "hidden_layers max_iterations patience learning_rate tol batch_size optim_type dropout_rate use_batch_norm verbose", "CC_kwargs")
_map("steps.cc.fit.pls", "components tol max_iterations", "CC_kwargs", {"components": "ncomp"})
_map("steps.cc.apply", "use_lut lazy_lut grid_size min_pixels", "CC_kwargs", {"grid_size": "lut_grid_size", "min_pixels": "lut_min_pixels"})
_map("steps.cc.validation", "enabled n_splits random_state", "validation_options")
_map("numerics", "precision", None, {"precision": "precision"})
_map("observations", "saturation max_clipped_fraction", "stage_policy")
_map("persistence", "enabled directory format", None, {"enabled": "save", "directory": "save_path", "format": "model_format"})
_DIRECT = {"input": "image_options", "chart": "chart_detection", "reference": "reference_profile", "color": "color_semantics"}


def _mapping(value, path):
    if path == "stage_policy" and isinstance(value, str):
        # A version alias specifies only its identity, not all policy defaults.
        value = {"version": value}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(path + " must be a mapping")
    return deepcopy(dict(value))


def _equal(first, second):
    if isinstance(first, os.PathLike) or isinstance(second, os.PathLike):
        try:
            return os.fspath(first) == os.fspath(second)
        except TypeError:
            return False
    if hasattr(first, "to_dict"):
        first = first.to_dict()
    if hasattr(second, "to_dict"):
        second = second.to_dict()
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(_equal(a, b) for a, b in zip(first, second))
    # Equal numeric values are not interchangeable validation requests: True
    # is not integer 1, and integer-only controls must not accept float 2.0.
    if isinstance(first, numbers.Number) or isinstance(second, numbers.Number):
        if type(first) is not type(second):
            return False
    if isinstance(first, np.ndarray) != isinstance(second, np.ndarray):
        return False
    try:
        return bool(np.array_equal(first, second))
    except (TypeError, ValueError):
        return False


def _put(request, target, value, path):
    if len(target) == 1:
        owner, key = request, target[0]
    else:
        root, key = target
        if request.get(root) is None:
            request[root] = {}
        if not isinstance(request[root], dict):
            request[root] = _mapping(request[root], root)
        owner = request[root]
    if key in owner:
        if not _equal(owner[key], value):
            raise ValueError(path + " conflicts with " + ".".join(target))
        return  # Preserve the existing explicit value and its validation type.
    owner[key] = deepcopy(value)


def _walk(value, prefix):
    if prefix in _LEAVES:
        yield prefix, value
        return
    if not isinstance(value, Mapping):
        raise TypeError(".".join(prefix) + " must be a mapping")
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("grouped option keys must be strings")
        path = prefix + (key,)
        if not any(leaf[:len(path)] == path for leaf in _LEAVES):
            raise ValueError("unknown grouped option: " + ".".join(path))
        yield from _walk(item, path)


def resolve_grouped(config_type, arguments, groups, required_keys=()):
    """Resolve an additive request through the unchanged numerical constructor."""
    request = {key: deepcopy(value) for key, value in arguments.items()
               if value is not UNSET and (value is not None or key in required_keys)}
    policies = {}
    steps = groups.get("steps")
    if steps is not None:
        if isinstance(steps, Mapping):
            unknown = set(steps) - set(_STEP_TYPES)
            if unknown:
                raise ValueError("unknown steps: " + repr(sorted(unknown)))
            steps = Steps(**dict(steps))
        if not isinstance(steps, Steps):
            raise TypeError("steps must be Steps or a mapping")
        groups["steps"] = steps.to_dict()
    for name, value in groups.items():
        if value is None:
            continue
        if name == "reference":
            _put(request, ("reference_profile",), value, name)
        elif name in ("input", "chart"):
            for key, item in _mapping(value, name).items():
                _put(request, (_DIRECT[name], key), item, name + "." + key)
        elif name == "color":
            policies["color_semantics"] = _mapping(value, name)
        else:
            for path, item in _walk(_mapping(value, name), (name,)):
                target = _LEAVES[path]
                if target[0] == "stage_policy":
                    policies.setdefault("stage_policy", {})[target[1]] = item
                    # Check old partial policy fields, without materializing defaults.
                    if arguments.get("stage_policy") is not None:
                        check = {"stage_policy": _mapping(arguments["stage_policy"], "stage_policy")}
                        _put(check, target, item, ".".join(path))
                else:
                    _put(request, target, item, ".".join(path))
    baseline = config_type(**request)
    if baseline.api_version != 2 or baseline.fitting_policy["version"] != "ui-compatible-v1":
        raise ValueError("grouped configuration requires modern API 2; retain flat forms for historical training")
    for root, patch in policies.items():
        original = arguments.get(root)
        if original is not None:
            check = {root: _mapping(original, root)}
            for key, value in patch.items():
                _put(check, (root, key), value, ("color" if root == "color_semantics" else "steps/observations") + "." + key)
        request[root] = {**getattr(baseline, root), **patch}
    result = config_type(**request)
    cc = (groups.get("steps") or {}).get("cc", {}).get("fit", {})
    kind = result.CC_kwargs.get("model", "rpcc2")
    if "nn" in cc and kind != "nn":
        raise ValueError("steps.cc.fit.nn requires model='nn'")
    if "pls" in cc and kind != "pls":
        raise ValueError("steps.cc.fit.pls requires model='pls'")
    method = result.FFC_kwargs.get("fit_method", "poly")
    if method not in ("linear", "poly"):
        raise ValueError("steps.ffc.fit.method supports linear/poly in the modern UI profiles")
    if result.precision == "high" and result.CC_kwargs.get("use_lut") is True:
        raise ValueError("steps.cc.apply.use_lut=True is not qualified for high precision")
    try:
        result.validate()
    except (TypeError, ValueError) as error:
        raise type(error)("grouped configuration: " + str(error)) from error
    return result


def grouped_dict(config):
    """Lossless categorized presentation of the existing API-2 flat state."""
    if config.api_version != 2 or config.fitting_policy["version"] != "ui-compatible-v1":
        raise ValueError("grouped presentation requires modern API 2; use to_dict() for historical configurations")
    remaining = config.to_dict()
    result = {"layout": GROUPED_LAYOUT, "api_version": 2}
    remaining.pop("api_version", None)
    for name, target in _DIRECT.items():
        result[name] = remaining.pop(target)
    kind = config.CC_kwargs.get("model", "rpcc2")
    for path, target in _LEAVES.items():
        if path[:4] == ("steps", "cc", "fit", "nn") and kind != "nn":
            continue
        if path[:4] == ("steps", "cc", "fit", "pls") and kind != "pls":
            continue
        owner = remaining if len(target) == 1 else remaining.get(target[0], {})
        key = target[-1]
        if key not in owner:
            continue
        value = owner.pop(key)
        branch = result
        for name in path[:-1]:
            branch = branch.setdefault(name, {})
        branch[path[-1]] = value
    # Fixed policy identities and any supported flat compatibility controls
    # remain explicit; no old key is discarded by the presentation adapter.
    result.update({key: value for key, value in remaining.items() if value is not None and value != {}})
    return _thaw(_FrozenMap(result), json_ready=True)


def group_view(config, name):
    value = grouped_dict(config).get(name, {})
    return Steps(**value) if name == "steps" else _FrozenMap(value)
