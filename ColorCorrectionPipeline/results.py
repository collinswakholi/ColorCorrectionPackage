"""Compact, owned batch records; report conversion never embeds pixel arrays."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("report values must be finite JSON data; pixel arrays are not report fields")


def _owned_mapping(value):
    if not isinstance(value, Mapping):
        raise TypeError("batch metadata must be a mapping")
    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item
    return freeze(_plain(value))


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    input_identity: object
    status: str
    calibration_id: str
    images: object = field(default_factory=dict)
    exports: object = field(default_factory=dict)
    error: object = None
    warnings: tuple = ()
    timings: object = field(default_factory=dict)
    provenance: object = field(default_factory=dict)

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("batch index must be a nonnegative integer")
        if self.status not in ("succeeded", "failed", "cancelled"):
            raise ValueError("unknown batch item status")
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("calibration_id must be a nonempty string")
        if not isinstance(self.input_identity, (str, Mapping)):
            raise TypeError("input_identity must be text or a compact JSON mapping")
        identity = self.input_identity if isinstance(self.input_identity, str) else _owned_mapping(self.input_identity)
        if not isinstance(self.images, Mapping) or any(not isinstance(key, str) or not isinstance(value, np.ndarray) for key, value in self.images.items()):
            raise TypeError("images must map stage names to NumPy arrays")
        if not isinstance(self.timings, Mapping):
            raise TypeError("timings must be a mapping")
        for key, value in self.timings.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("batch timings must be finite nonnegative seconds")
        if self.status == "failed" and not isinstance(self.error, Mapping):
            raise ValueError("failed results require structured error details")
        object.__setattr__(self, "input_identity", identity)
        # Pixel arrays belong to this output; do not duplicate large images merely
        # to make a record. The stage mapping itself is read-only.
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))
        object.__setattr__(self, "exports", _owned_mapping(self.exports))
        object.__setattr__(self, "error", None if self.error is None else _owned_mapping(self.error))
        object.__setattr__(self, "warnings", tuple(_owned_mapping(warning) for warning in self.warnings))
        object.__setattr__(self, "timings", _owned_mapping(self.timings))
        object.__setattr__(self, "provenance", _owned_mapping(self.provenance))

    def to_dict(self):
        return {"index": self.index, "input_identity": _plain(self.input_identity),
                "status": self.status, "calibration_id": self.calibration_id,
                "images": {name: {"shape": list(image.shape), "dtype": str(image.dtype),
                                  "bytes": int(image.nbytes)} for name, image in self.images.items()},
                "exports": _plain(self.exports), "error": _plain(self.error),
                "warnings": _plain(self.warnings), "timings": _plain(self.timings),
                "provenance": _plain(self.provenance)}


class BatchResult:
    """Collected records. Streaming callers avoid this intentional retention."""
    def __init__(self, items, summary):
        self._items = tuple(items)
        if any(not isinstance(item, BatchItemResult) for item in self._items):
            raise TypeError("items must contain BatchItemResult records")
        self._summary = _plain(summary)

    @property
    def items(self):
        return self._items

    @property
    def summary(self):
        return deepcopy(self._summary)

    def to_dict(self):
        return {"schema_version": 1, "summary": self.summary,
                "items": [item.to_dict() for item in self.items]}
