"""Frozen patch-9 image contract oracle (MIT, same project).
Extracted helper/class definitions retain exact prior source.
Original contracts.py SHA256: d9835e1442e31ef46d2758e0706953964a9ba7cf797d3fe7ee74eee410c72aa6
Do not update alongside an optimization without deliberate baseline review.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple
import numpy as np
CONTRACT_VERSION = 1
_RGB_SPACES = ("srgb", "linear-srgb")

def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _metadata(value):
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a JSON-compatible mapping")
    def validate_keys(item):
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("provenance keys must be strings")
            for nested in item.values():
                validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate_keys(nested)
    validate_keys(value)
    try:
        serialized = json.dumps(_plain(value), sort_keys=True, allow_nan=False)
        cloned = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must contain finite JSON-compatible values") from exc
    if not all(isinstance(key, str) for key in cloned):
        raise ValueError("provenance keys must be strings")
    return _freeze_json(cloned)


def _version(value):
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError("unsupported contract version")


def _readonly_array(value, shape=None, label="array"):
    array = np.asarray(value, dtype=np.dtype("<f8"))
    if shape is not None and array.shape != shape:
        raise ValueError("%s must have shape %s, received %s" % (label, shape, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("%s must contain finite values" % label)
    # A non-owning ndarray backed by immutable bytes cannot be made writable by
    # setting its flag back to True, unlike an ordinary writeable=False array.
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype).reshape(array.shape)


def _image_dtype(dtype):
    dtype = np.dtype(dtype)
    return (dtype.kind == "u" and dtype.itemsize in (1, 2)) or (dtype.kind == "f" and dtype.itemsize in (2, 4, 8))


def _integer(value, label, low=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError("%s must be an integer >= %d" % (label, low))
    return int(value)


def _hash_parts(metadata, arrays=()):
    digest = hashlib.sha256(json.dumps(_plain(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8"))
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)))
    return digest.hexdigest()


def _check_keys(payload, allowed, required):
    if not isinstance(payload, Mapping):
        raise TypeError("contract payload must be a mapping")
    extra, missing = set(payload) - set(allowed), set(required) - set(payload)
    if extra or missing:
        raise ValueError("contract fields invalid: unknown=%s, missing=%s" % (sorted(extra), sorted(missing)))
    _version(payload.get("version"))


def _check_claimed_digest(payload, actual):
    claimed = payload.get("digest")
    if claimed is not None and claimed != actual:
        raise ValueError("contract digest does not match its contents")


class _ImmutableContract:
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # Default ndarray pickle restoration can produce writable arrays. Always
        # restore via the validating constructor, including immutable byte backing.
        return type(self).from_dict, (self.to_dict(),)


def _orient(array, orientation):
    if orientation == 1:
        return array
    if orientation == 2:
        return array[:, ::-1]
    if orientation == 3:
        return array[::-1, ::-1]
    if orientation == 4:
        return array[::-1]
    if orientation == 5:
        return array.swapaxes(0, 1)
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return array.swapaxes(0, 1)[::-1, ::-1]
    return np.rot90(array, 1)


@dataclass(frozen=True, eq=False)
class ImageData(_ImmutableContract):
    """Owned canonical RGB float64 pixels plus explicit source interpretation.

    Use ``from_array`` for integer/range/channel/orientation conversion. Direct
    construction means pixels are already normalized RGB in their declared
    encoding. It never infers an ICC profile or performs color adaptation.
    """

    pixels: np.ndarray
    color_space: str = "srgb"
    source_dtype: str = "float64"
    source_range: Tuple[float, float] = (0.0, 1.0)
    source_channel_order: str = "RGB"
    source_orientation: int = 1
    orientation_applied: bool = True
    source_shape: Optional[Tuple[int, int, int]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    version: int = CONTRACT_VERSION
    digest: str = field(init=False)
    pixel_digest: str = field(init=False)

    def __post_init__(self):
        _version(self.version)
        if self.color_space not in _RGB_SPACES:
            raise ValueError("image color_space must be srgb or linear-srgb")
        pixels = _readonly_array(self.pixels, label="image pixels")
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError("image pixels must have nonempty HxWx3 shape")
        if (pixels < 0).any() or (pixels > 1).any():
            raise ValueError("image pixels must be normalized in [0,1]")
        dtype = np.dtype(self.source_dtype)
        if not _image_dtype(dtype):
            raise ValueError("source_dtype must be uint8, uint16, or floating RGB")
        ranges = tuple(float(value) for value in self.source_range)
        if len(ranges) != 2 or not all(math.isfinite(value) for value in ranges) or ranges[1] <= ranges[0]:
            raise ValueError("source_range must be an increasing finite pair")
        if self.source_channel_order not in ("RGB", "BGR"):
            raise ValueError("source_channel_order must be RGB or BGR")
        orientation = _integer(self.source_orientation, "source_orientation")
        if orientation > 8 or not isinstance(self.orientation_applied, bool):
            raise ValueError("orientation must be EXIF 1..8 and applied flag must be boolean")
        source_shape = tuple(self.source_shape) if self.source_shape is not None else tuple(pixels.shape)
        if len(source_shape) != 3 or source_shape[2] != 3:
            raise ValueError("source_shape must be HxWx3")
        source_shape = tuple(_integer(v, "source_shape dimension") for v in source_shape)
        expected = source_shape[:2]
        if self.orientation_applied and orientation >= 5:
            expected = expected[::-1]
        if tuple(pixels.shape[:2]) != tuple(expected):
            raise ValueError("pixel shape contradicts source orientation/shape")
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "source_dtype", dtype.name)
        object.__setattr__(self, "source_range", ranges)
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "source_orientation", orientation)
        object.__setattr__(self, "provenance", _metadata(self.provenance))
        pixel_digest = _hash_parts({"color_space": self.color_space, "channel_order": "RGB"}, (pixels,))
        object.__setattr__(self, "pixel_digest", pixel_digest)
        object.__setattr__(self, "digest", _hash_parts(self._metadata_fields(), (pixels,)))

    @classmethod
    def from_array(cls, array, *, channel_order="RGB", color_space="srgb", value_range=None,
                   orientation=1, apply_orientation=True, provenance=None):
        source = np.asarray(array)
        if not _image_dtype(source.dtype):
            raise TypeError("array dtype must be uint8, uint16, or floating")
        if source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
            raise ValueError("array must have nonempty HxWx3 shape")
        if channel_order not in ("RGB", "BGR"):
            raise ValueError("channel_order must be RGB or BGR")
        orientation = _integer(orientation, "orientation")
        if orientation > 8 or not isinstance(apply_orientation, bool):
            raise ValueError("orientation must be EXIF 1..8 and apply_orientation boolean")
        if value_range is None:
            value_range = (0.0, float(np.iinfo(source.dtype).max)) if source.dtype.kind == "u" else (0.0, 1.0)
        if not isinstance(value_range, (tuple, list)) or len(value_range) != 2:
            raise ValueError("value_range must be an explicit (minimum, maximum) pair")
        lo, hi = (float(v) for v in value_range)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("value_range must be increasing and finite")
        if not np.isfinite(source).all() or (source < lo).any() or (source > hi).any():
            raise ValueError("source pixels lie outside the declared finite value_range")
        normalized = (source.astype(np.float64) - lo) / (hi - lo)
        if channel_order == "BGR":
            normalized = normalized[:, :, ::-1]
        if apply_orientation:
            normalized = _orient(normalized, orientation)
        return cls(normalized, color_space=color_space, source_dtype=source.dtype.name,
                   source_range=(lo, hi), source_channel_order=channel_order,
                   source_orientation=orientation, orientation_applied=apply_orientation,
                   source_shape=tuple(source.shape), provenance=provenance if provenance is not None else {})

    @property
    def channel_order(self):
        return "RGB"

    @property
    def shape(self):
        return self.pixels.shape

    def _metadata_fields(self):
        return {"version": self.version, "color_space": self.color_space, "channel_order": "RGB",
                "shape": list(self.pixels.shape), "source_dtype": self.source_dtype,
                "source_range": list(self.source_range), "source_channel_order": self.source_channel_order,
                "source_orientation": self.source_orientation, "orientation_applied": self.orientation_applied,
                "source_shape": list(self.source_shape), "provenance": _plain(self.provenance)}

    def to_metadata(self):
        return dict(self._metadata_fields(), digest=self.digest, pixel_digest=self.pixel_digest)

    def to_dict(self):
        return dict(self.to_metadata(), pixels=self.pixels.tolist())

    @classmethod
    def from_dict(cls, payload):
        allowed = {"version", "color_space", "channel_order", "shape", "source_dtype", "source_range",
                   "source_channel_order", "source_orientation", "orientation_applied", "source_shape",
                   "provenance", "digest", "pixel_digest", "pixels"}
        _check_keys(payload, allowed, {"version", "pixels", "color_space"})
        if payload.get("channel_order", "RGB") != "RGB":
            raise ValueError("serialized canonical image channel_order must be RGB")
        constructor = {key: value for key, value in payload.items() if key not in ("digest", "pixel_digest", "shape", "channel_order")}
        result = cls(**constructor)
        if "shape" in payload and tuple(payload["shape"]) != result.shape:
            raise ValueError("serialized image shape does not match pixels")
        if "pixel_digest" in payload and payload["pixel_digest"] != result.pixel_digest:
            raise ValueError("image pixel digest does not match contents")
        _check_claimed_digest(payload, result.digest)
        return result

    def to_srgb(self):
        """Return an independent, writable encoded-sRGB legacy adapter."""
        if self.color_space == "srgb":
            return self.pixels.copy()
        from .colorimetry import linear_to_srgb
        return np.asarray(linear_to_srgb(self.pixels, policy="raise"), dtype=np.float64).copy()

    def with_orientation_applied(self):
        """Normalize a deliberately pending orientation once, preserving source facts."""
        if self.orientation_applied:
            return self
        return replace(self, pixels=_orient(self.pixels, self.source_orientation), orientation_applied=True)

    def to_bgr8(self):
        """Explicit nearest-integer 8-bit detector view, never canonical storage."""
        return np.rint(self.to_srgb()[:, :, ::-1] * 255.0).astype(np.uint8)
