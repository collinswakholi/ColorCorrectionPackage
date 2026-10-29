"""Atomic compact JSON reports without implicit pixel or model serialization."""
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

from .results import _plain


def save_report(path, report):
    """Write a finite JSON snapshot; failed encoding/write preserves destination."""
    path = Path(path)
    value = report.to_dict() if callable(getattr(report, "to_dict", None)) else report
    if not isinstance(value, Mapping):
        raise TypeError("report must be a JSON mapping or provide to_dict()")
    encoded = json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False).encode("utf8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="."+path.name+".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        # Decode our prepared artifact before replacing a previous report.
        json.loads(temporary.read_text(encoding="utf8"))
        os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
