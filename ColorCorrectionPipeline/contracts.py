"""Versioned, owned image/reference/observation contracts.

These contracts describe interpretation and identity. They do not fit a model,
infer a color space from pixels, or certify that nominal chart values are valid
for a physical chart. Legacy ndarray APIs can use the explicit adapters without
changing their historical return shapes.
"""

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

CONTRACT_VERSION = 1
_RGB_SPACES = ("srgb", "linear-srgb")
_REFERENCE_SPACES = ("lab", "xyz", "srgb", "linear-srgb")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIXED_TEMPLATES = {"classic_4x6", "spyder_4x6", "digital_sg_10x14"}


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


def _immutable_image_snapshot(value):
    """Freeze canonical f64 storage once, never trust only writeable=False.

    Exact ndarray views ultimately backed by immutable bytes can be reused.
    Read-only views over mutable buffers, mmap, ctypes, and array owners cannot.
    Validation is deliberately performed by the caller on this owned snapshot.
    """
    array = np.asarray(value, dtype=np.dtype("<f8"))
    root = array
    while type(root) is np.ndarray and root.base is not None:
        root = root.base
    if array.flags.c_contiguous and type(root) is bytes and len(root) == array.nbytes:
        # Rebuild the header as well: shape/dtype assignments on the caller's
        # read-only ndarray must not mutate this contract's ndarray metadata.
        return np.frombuffer(root, dtype=np.dtype("<f8")).reshape(array.shape)
    # tobytes(order='C') also avoids a separate contiguous temporary for strided
    # inputs while preserving the exact canonical byte order and digest format.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(array.shape)


def _image_bounds(array):
    """Two scalar reductions replace three full-size boolean workspaces."""
    with np.errstate(invalid="ignore"):
        low, high = float(array.min()), float(array.max())
    return low, high


def _image_digests(metadata, pixels):
    pixel_metadata = {"color_space": metadata["color_space"], "channel_order": "RGB"}
    if pixels.nbytes < 8 * 1024 * 1024:
        return _hash_parts(pixel_metadata, (pixels,)), _hash_parts(metadata, (pixels,))
    # Both historical digests include every immutable pixel byte with different
    # prefixes; neither can be derived from the other. OpenSSL releases the GIL
    # for these large updates. One bounded worker overlaps the independent SHA
    # scans without changing identity, retaining caller buffers, or caching a
    # mutable ndarray's identity. The context joins on success and on failure.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ccp-image-sha256") as worker:
        try:
            pixel = worker.submit(_hash_parts, pixel_metadata, (pixels,))
        except RuntimeError:
            # Resource-constrained processes need no new thread to preserve the
            # original identities. Do not catch failures raised by hash work.
            return _hash_parts(pixel_metadata, (pixels,)), _hash_parts(metadata, (pixels,))
        image = _hash_parts(metadata, (pixels,))
        return pixel.result(), image


def _normalized_image_snapshot(source, lo, hi):
    """Normalize strided RGB into immutable storage with bounded workspace."""
    if source.size * 8 < 8 * 1024 * 1024:
        values = source.astype(np.float64)
        low, high = _image_bounds(values)
        if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
            raise ValueError("source pixels lie outside the declared finite value_range")
        values -= lo
        values /= hi-lo
        return values
    # Fill the private destination directly, release every writable export, then
    # expose immutable bytes. No full-frame cast/subtract/divide temporaries.
    with io.BytesIO() as stream:
        # Reserve the exact canonical extent once. Incrementally growing this
        # large buffer repeatedly copies prior chunks on some Python builds.
        stream.seek(source.size * 8 - 1)
        stream.write(b"\0")
        with stream.getbuffer() as buffer:
            values = np.ndarray(source.shape, dtype=np.dtype("<f8"), buffer=buffer)
            try:
                from ._image_numeric import try_normalize_unsigned
                if not try_normalize_unsigned(source, values, lo, hi):
                    # Also overwrites every value after a partial backend failure.
                    np.copyto(values, source, casting="unsafe")
                    low, high = _image_bounds(values)
                    if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
                        raise ValueError("source pixels lie outside the declared finite value_range")
                    values -= lo
                    values /= hi-lo
            finally:
                del values
        payload = stream.getvalue()
    return np.frombuffer(payload, dtype=np.dtype("<f8")).reshape(source.shape)


def _image_dtype(dtype):
    dtype = np.dtype(dtype)
    return (dtype.kind == "u" and dtype.itemsize in (1, 2)) or (dtype.kind == "f" and dtype.itemsize in (2, 4, 8))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % label)
    return value


def _integer(value, label, low=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError("%s must be an integer >= %d" % (label, low))
    return int(value)


def _unit(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("%s must be finite in %s" % (label, "(0,1]" if positive else "[0,1]"))
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > 1 or (positive and number == 0):
        raise ValueError("%s must be finite in %s" % (label, "(0,1]" if positive else "[0,1]"))
    return number


def _digest(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("%s must be a lowercase SHA-256 digest" % label)
    return value


def _layout(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("layout must be (rows, columns)")
    rows, columns = (_integer(item, "layout dimension") for item in value)
    if rows * columns > 4096:
        raise ValueError("layout has too many cells")
    return rows, columns


def _identifiers(value, label, count=None):
    if isinstance(value, str):
        raise ValueError("%s must be a sequence of identifiers" % label)
    values = tuple(_text(item, label) for item in value)
    if count is not None and len(values) != count:
        raise ValueError("%s must contain %d values" % (label, count))
    if len(set(values)) != len(values):
        raise ValueError("%s must be unique" % label)
    return values


def _canonical_cells(layout):
    return tuple("r%dc%d" % (r + 1, c + 1) for r in range(layout[0]) for c in range(layout[1]))


def _white(value):
    if isinstance(value, str):
        normalized = value.upper()
        if normalized not in ("D50", "D65"):
            raise ValueError("source_white must be D50, D65, or an explicit xy pair")
        return normalized
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all() or (array <= 0).any() or array.sum() >= 1:
        raise ValueError("source_white must be a finite physical xy pair")
    return tuple(float(v) for v in array)


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

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == "pixels" and type(value) is np.ndarray:
            # Read-only NumPy data still permits shape/dtype/stride assignments.
            # Give each access a separate header, retaining immutable storage,
            # so public views cannot invalidate this contract's digest or facts.
            return value.view()
        return value

    def __post_init__(self):
        _version(self.version)
        if self.color_space not in _RGB_SPACES:
            raise ValueError("image color_space must be srgb or linear-srgb")
        pixels = _immutable_image_snapshot(self.pixels)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError("image pixels must have nonempty HxWx3 shape")
        low, high = _image_bounds(pixels)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("image pixels must contain finite values")
        if low < 0 or high > 1:
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
        pixel_digest, image_digest = _image_digests(self._metadata_fields(), pixels)
        object.__setattr__(self, "pixel_digest", pixel_digest)
        object.__setattr__(self, "digest", image_digest)

    @classmethod
    def from_array(cls, array, *, channel_order="RGB", color_space="srgb", value_range=None,
                   orientation=1, apply_orientation=True, provenance=None):
        source = np.asarray(array).view()
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
        # Validate the captured values, not the mutable caller before copying.
        # Normalization retains the historical IEEE operation order, including
        # signed-zero and unusual declared-range behavior.
        layout = source[:, :, ::-1] if channel_order == "BGR" else source
        if apply_orientation:
            layout = _orient(layout, orientation)
        if source.dtype == np.dtype("<f8") and lo == 0.0 and not np.signbit(lo) and hi == 1.0:
            normalized = _immutable_image_snapshot(layout)
            low, high = _image_bounds(normalized)
            if not math.isfinite(low) or not math.isfinite(high) or low < lo or high > hi:
                raise ValueError("source pixels lie outside the declared finite value_range")
        else:
            normalized = _normalized_image_snapshot(layout, lo, hi)
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


@dataclass(frozen=True, eq=False)
class ReferenceProfile(_ImmutableContract):
    """Unclipped canonical Lab D50 truth with explicit template/cell identity."""

    profile_id: str
    revision: int
    names: Tuple[str, ...]
    canonical_lab_d50: np.ndarray
    neutral_ids: Tuple[str, ...]
    layout: Tuple[int, int] = (4, 6)
    name: str = "Custom reference"
    cell_ids: Optional[Tuple[str, ...]] = None
    template_id: str = "classic_4x6"
    source_color_space: str = "lab"
    source_white: Any = "D50"
    source_scale: float = 1.0
    adaptation: str = "Bradford"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    version: int = CONTRACT_VERSION
    digest: str = field(init=False)

    def __post_init__(self):
        _version(self.version)
        _text(self.profile_id, "profile_id")
        _text(self.name, "name")
        _text(self.template_id, "template_id")
        revision = _integer(self.revision, "revision")
        layout = _layout(self.layout)
        if self.template_id in ("classic_4x6", "spyder_4x6") and layout != (4, 6):
            raise ValueError("template_id contradicts layout")
        if self.template_id == "digital_sg_10x14" and layout != (10, 14):
            raise ValueError("template_id contradicts layout")
        count = layout[0] * layout[1]
        names = tuple(_text(v, "reference name") for v in self.names)
        if len(names) != count:
            raise ValueError("names must match the reference layout")
        cell_ids = self.cell_ids if self.cell_ids is not None else _canonical_cells(layout)
        cell_ids = _identifiers(cell_ids, "cell_ids", count)
        if self.template_id in _FIXED_TEMPLATES and cell_ids != _canonical_cells(layout):
            raise ValueError("known-template cell_ids must be canonical rNcM row-major order")
        neutrals = _identifiers(self.neutral_ids, "neutral_ids")
        if not set(neutrals).issubset(cell_ids):
            raise ValueError("neutral_ids must refer to declared cell_ids")
        values = _readonly_array(self.canonical_lab_d50, (count, 3), "canonical_lab_d50")
        if self.source_color_space not in _REFERENCE_SPACES:
            raise ValueError("unsupported source_color_space")
        white = _white(self.source_white)
        scale = float(self.source_scale)
        allowed = (1.0,) if self.source_color_space == "lab" else (1.0, 100.0) if self.source_color_space == "xyz" else (1.0, 255.0, 65535.0)
        if isinstance(self.source_scale, bool) or not math.isfinite(scale) or scale not in allowed:
            raise ValueError("source_scale contradicts source_color_space")
        if self.source_color_space in _RGB_SPACES and white != "D65" and white != (0.31271, 0.32902):
            raise ValueError("sRGB reference encodings require their D65 source white")
        if self.adaptation not in ("Bradford", "CAT02"):
            raise ValueError("adaptation must be explicitly Bradford or CAT02")
        for key, value in (("revision", revision), ("layout", layout), ("names", names), ("cell_ids", cell_ids),
                           ("neutral_ids", neutrals), ("canonical_lab_d50", values), ("source_white", white),
                           ("source_scale", scale), ("provenance", _metadata(self.provenance))):
            object.__setattr__(self, key, value)
        object.__setattr__(self, "digest", _hash_parts(self._metadata_fields(), (values,)))

    @property
    def neutral_indices(self):
        return tuple(self.cell_ids.index(cell) for cell in self.neutral_ids)

    def _metadata_fields(self):
        return {"version": self.version, "profile_id": self.profile_id, "revision": self.revision,
                "name": self.name, "names": list(self.names), "cell_ids": list(self.cell_ids),
                "neutral_ids": list(self.neutral_ids), "layout": list(self.layout), "template_id": self.template_id,
                "canonical_color_space": "lab", "canonical_white": "D50", "canonical_scale": 1,
                "source_color_space": self.source_color_space, "source_white": _plain(self.source_white),
                "source_scale": self.source_scale, "adaptation": self.adaptation, "provenance": _plain(self.provenance)}

    def to_dict(self):
        return dict(self._metadata_fields(), canonical_lab_d50=self.canonical_lab_d50.tolist(), digest=self.digest)

    @classmethod
    def from_dict(cls, payload):
        allowed = {"version", "profile_id", "revision", "name", "names", "cell_ids", "neutral_ids", "layout",
                   "template_id", "canonical_color_space", "canonical_white", "canonical_scale", "source_color_space",
                   "source_white", "source_scale", "adaptation", "provenance", "canonical_lab_d50", "digest"}
        _check_keys(payload, allowed, {"version", "profile_id", "revision", "names", "canonical_lab_d50", "neutral_ids"})
        if payload.get("canonical_color_space", "lab") != "lab" or payload.get("canonical_white", "D50") != "D50" or payload.get("canonical_scale", 1) != 1:
            raise ValueError("canonical reference representation must be Lab D50 scale 1")
        constructor = {key: value for key, value in payload.items() if key not in ("digest", "canonical_color_space", "canonical_white", "canonical_scale")}
        result = cls(**constructor)
        _check_claimed_digest(payload, result.digest)
        return result

    @classmethod
    def from_values(cls, values, *, color_space, white, scale, profile_id, revision, names, neutral_ids,
                    adaptation="Bradford", **kwargs):
        from .colorimetry import reference_values_to_lab_d50
        canonical = reference_values_to_lab_d50(values, color_space=color_space, white=white,
                                               scale=scale, adaptation=adaptation)
        return cls(profile_id, revision, names, canonical, neutral_ids,
                   source_color_space=color_space, source_white=white, source_scale=scale,
                   adaptation=adaptation, **kwargs)

    @classmethod
    def classic24(cls):
        import colour
        chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        names = tuple(str(name) for name in chart.data)
        cells = tuple("r%dc%d" % (r + 1, c + 1) for r in range(4) for c in range(6))
        # Catalog coordinates are fixed xyY facts in their declared unit scale.
        # Do not let colour's process-global domain_range_scale reinterpret them.
        xy_y = np.asarray(list(chart.data.values()), dtype=np.float64)
        x, y, luminance = xy_y.T
        xyz = np.column_stack((x * luminance / y, luminance, (1 - x - y) * luminance / y))
        return cls.from_values(xyz, color_space="xyz",
                               white=np.asarray(chart.illuminant), scale=1,
                               profile_id="classic24-after-2014", revision=1, names=names,
                               neutral_ids=cells[-6:], cell_ids=cells,
                               name="ColorChecker24 - After November 2014",
                               provenance={"source": "colour.CCS_COLOURCHECKERS", "edition": "after-november-2014"})

    def display_rgb(self, *, clip=True, adaptation=None):
        from .colorimetry import display_reference_rgb
        return display_reference_rgb(self.canonical_lab_d50,
                                     adaptation=adaptation or self.adaptation, clip=clip)


def _quads(value, count, image_shape):
    quads = _readonly_array(value, (count, 4, 2), "quadrilaterals")
    h, w = image_shape
    if (quads[:, :, 0] < 0).any() or (quads[:, :, 0] > w - 1).any() or (quads[:, :, 1] < 0).any() or (quads[:, :, 1] > h - 1).any():
        raise ValueError("quadrilaterals must stay inside the declared image")
    edges = np.roll(quads, -1, axis=1) - quads
    next_edges = np.roll(edges, -1, axis=1)
    cross = edges[:, :, 0] * next_edges[:, :, 1] - edges[:, :, 1] * next_edges[:, :, 0]
    if not np.all(np.all(cross > 1e-9, axis=1) | np.all(cross < -1e-9, axis=1)):
        raise ValueError("quadrilaterals must be nondegenerate convex ordered corners")
    centers = quads.mean(axis=1)
    if len(np.unique(np.round(centers, 8), axis=0)) != count:
        raise ValueError("different cells cannot share one sampling region center")
    return quads


@dataclass(frozen=True, eq=False)
class ChartObservation(_ImmutableContract):
    """One face's fixed ordered cells and measured per-cell quality.

    ``clipped_fractions`` refer to sampled pixels (any channel at threshold),
    not clipped patch means. Missing faces are represented by ObservationSet's
    unavailable reason, never invented zero-valued chart measurements.
    """

    chart_id: str
    face_id: str
    reference_digest: str
    cell_ids: Tuple[str, ...]
    quadrilaterals: np.ndarray
    measured_rgb: np.ndarray
    clipped_fractions: np.ndarray
    confidence: float
    image_shape: Tuple[int, int]
    layout: Tuple[int, int] = (4, 6)
    template_id: str = "classic_4x6"
    color_space: str = "srgb"
    stage: str = "source"
    cell_statuses: Tuple[str, ...] = ()
    sample_count: int = 1
    clipping_threshold: float = 0.99
    clipping_basis: str = "sampled-pixels"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    version: int = CONTRACT_VERSION
    digest: str = field(init=False)

    def __post_init__(self):
        _version(self.version)
        for key in ("chart_id", "face_id", "stage", "template_id"):
            _text(getattr(self, key), key)
        _digest(self.reference_digest, "reference_digest")
        layout = _layout(self.layout)
        count = layout[0] * layout[1]
        cells = _identifiers(self.cell_ids, "cell_ids", count)
        if self.template_id in ("classic_4x6", "spyder_4x6") and layout != (4, 6):
            raise ValueError("template_id contradicts observation layout")
        if self.template_id == "digital_sg_10x14" and layout != (10, 14):
            raise ValueError("template_id contradicts observation layout")
        if self.template_id in _FIXED_TEMPLATES and cells != _canonical_cells(layout):
            raise ValueError("known-template cell_ids must be canonical rNcM row-major order")
        if not isinstance(self.image_shape, (tuple, list)) or len(self.image_shape) not in (2, 3):
            raise ValueError("image_shape must be (height,width) or (height,width,3)")
        shape = tuple(_integer(v, "image_shape dimension") for v in self.image_shape)
        if len(shape) == 3 and shape[2] != 3:
            raise ValueError("observation image_shape must have 3 channels")
        shape = shape[:2]
        quads = _quads(self.quadrilaterals, count, shape)
        measured = _readonly_array(self.measured_rgb, (count, 3), "measured_rgb")
        fractions = _readonly_array(self.clipped_fractions, (count,), "clipped_fractions")
        if (measured < 0).any() or (measured > 1).any():
            raise ValueError("measured_rgb must be normalized in [0,1]")
        if (fractions < 0).any() or (fractions > 1).any():
            raise ValueError("clipped_fractions must be in [0,1]")
        if self.color_space not in _RGB_SPACES:
            raise ValueError("observation color_space must be srgb or linear-srgb")
        if self.clipping_basis != "sampled-pixels":
            raise ValueError("clipping_basis must explicitly be sampled-pixels")
        statuses = tuple(self.cell_statuses) if self.cell_statuses else tuple("clipped" if fraction > 0 else "measured" for fraction in fractions)
        if len(statuses) != count or any(v not in ("measured", "clipped", "excluded", "estimated") for v in statuses):
            raise ValueError("cell_statuses must describe every declared cell")
        if any(status == "measured" and fraction > 0 for status, fraction in zip(statuses, fractions)):
            raise ValueError("clipped samples cannot be labeled unqualified measured")
        values = {"layout": layout, "cell_ids": cells, "image_shape": shape, "quadrilaterals": quads,
                  "measured_rgb": measured, "clipped_fractions": fractions, "cell_statuses": statuses,
                  "confidence": _unit(self.confidence, "confidence"),
                  "sample_count": _integer(self.sample_count, "sample_count"),
                  "clipping_threshold": _unit(self.clipping_threshold, "clipping_threshold", positive=True),
                  "provenance": _metadata(self.provenance)}
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "digest", _hash_parts(self._metadata_fields(), (quads, measured, fractions)))

    @property
    def area(self):
        quads = self.quadrilaterals
        return float(np.abs(np.sum(quads[:, :, 0] * np.roll(quads[:, :, 1], -1, axis=1) - quads[:, :, 1] * np.roll(quads[:, :, 0], -1, axis=1), axis=1)).sum() * 0.5)

    def validate_reference(self, profile):
        if not isinstance(profile, ReferenceProfile):
            raise TypeError("reference assignment requires ReferenceProfile")
        if self.reference_digest != profile.digest or self.template_id != profile.template_id or self.layout != profile.layout or self.cell_ids != profile.cell_ids:
            raise ValueError("observation reference assignment, template, or ordered cell IDs do not match")
        return self

    def _metadata_fields(self):
        return {"version": self.version, "chart_id": self.chart_id, "face_id": self.face_id,
                "reference_digest": self.reference_digest, "cell_ids": list(self.cell_ids), "layout": list(self.layout),
                "template_id": self.template_id, "color_space": self.color_space, "stage": self.stage,
                "confidence": self.confidence, "image_shape": list(self.image_shape),
                "cell_statuses": list(self.cell_statuses), "sample_count": self.sample_count,
                "clipping_threshold": self.clipping_threshold, "clipping_basis": self.clipping_basis,
                "provenance": _plain(self.provenance)}

    def to_dict(self):
        return dict(self._metadata_fields(), quadrilaterals=self.quadrilaterals.tolist(),
                    measured_rgb=self.measured_rgb.tolist(), clipped_fractions=self.clipped_fractions.tolist(), digest=self.digest)

    @classmethod
    def from_dict(cls, payload):
        allowed = {"version", "chart_id", "face_id", "reference_digest", "cell_ids", "layout", "template_id",
                   "color_space", "stage", "confidence", "image_shape", "cell_statuses", "sample_count",
                   "clipping_threshold", "clipping_basis", "provenance", "quadrilaterals", "measured_rgb",
                   "clipped_fractions", "digest"}
        _check_keys(payload, allowed, {"version", "chart_id", "face_id", "reference_digest", "cell_ids", "quadrilaterals", "measured_rgb", "clipped_fractions", "confidence", "image_shape"})
        result = cls(**{key: value for key, value in payload.items() if key != "digest"})
        _check_claimed_digest(payload, result.digest)
        return result

    @classmethod
    def from_samples(cls, *, samples, clipping_threshold=0.99, **kwargs):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 3 or samples.shape[2] != 3 or samples.shape[1] < 1:
            raise ValueError("samples must have shape (cells,samples_per_cell,3)")
        if not np.isfinite(samples).all() or (samples < 0).any() or (samples > 1).any():
            raise ValueError("samples must be finite normalized RGB")
        threshold = _unit(clipping_threshold, "clipping_threshold", positive=True)
        fractions = np.mean(np.any(samples >= threshold, axis=2), axis=1)
        return cls(measured_rgb=samples.mean(axis=1), clipped_fractions=fractions,
                   sample_count=samples.shape[1], clipping_threshold=threshold, **kwargs)


@dataclass(frozen=True, eq=False)
class ObservationSet(_ImmutableContract):
    """One source/stage's independently identified chart faces."""

    observations: Tuple[ChartObservation, ...]
    source_digest: str
    unavailable_reason: Optional[str] = None
    version: int = CONTRACT_VERSION
    digest: str = field(init=False)

    def __post_init__(self):
        _version(self.version)
        _digest(self.source_digest, "source_digest")
        observations = tuple(self.observations)
        if any(not isinstance(value, ChartObservation) for value in observations):
            raise TypeError("observations must contain ChartObservation objects")
        keys = [(value.chart_id, value.face_id) for value in observations]
        if len(set(keys)) != len(keys):
            raise ValueError("chart/face identities must be unique within one observation set")
        if len({value.stage for value in observations}) > 1 or len({value.image_shape for value in observations}) > 1:
            raise ValueError("one observation set must describe one stage and image shape")
        if not observations:
            _text(self.unavailable_reason, "unavailable_reason for empty observations")
        elif self.unavailable_reason is not None:
            raise ValueError("a populated observation set cannot be marked unavailable")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "digest", _hash_parts({"version": self.version, "source_digest": self.source_digest,
                                                     "unavailable_reason": self.unavailable_reason,
                                                     "observations": [value.digest for value in observations]}))

    def select_face(self, policy="highest-confidence", *, chart_id=None, face_id=None):
        if policy not in ("highest-confidence", "largest"):
            raise ValueError("face policy must be highest-confidence or largest")
        selected = [value for value in self.observations if (chart_id is None or value.chart_id == chart_id) and (face_id is None or value.face_id == face_id)]
        if not selected:
            raise ValueError("no matching chart face is available")
        if (chart_id is not None or face_id is not None) and len(selected) != 1:
            raise ValueError("chart/face selection is ambiguous; supply both identities")
        key = (lambda value: (-value.confidence, -value.area, value.chart_id, value.face_id)) if policy == "highest-confidence" else (lambda value: (-value.area, -value.confidence, value.chart_id, value.face_id))
        return min(selected, key=key)

    def validate_reference(self, profile):
        for observation in self.observations:
            observation.validate_reference(profile)
        return self

    def to_dict(self):
        return {"version": self.version, "source_digest": self.source_digest, "unavailable_reason": self.unavailable_reason,
                "observations": [value.to_dict() for value in self.observations], "digest": self.digest}

    @classmethod
    def from_dict(cls, payload):
        _check_keys(payload, {"version", "source_digest", "unavailable_reason", "observations", "digest"},
                    {"version", "source_digest", "observations"})
        result = cls(tuple(ChartObservation.from_dict(value) for value in payload["observations"]),
                     payload["source_digest"], payload.get("unavailable_reason"), payload["version"])
        _check_claimed_digest(payload, result.digest)
        return result


__all__ = ["CONTRACT_VERSION", "ImageData", "ReferenceProfile", "ChartObservation", "ObservationSet"]
