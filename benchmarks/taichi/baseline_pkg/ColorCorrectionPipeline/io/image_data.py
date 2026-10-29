"""Explicit image decoding that preserves precision and records assumptions."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..contracts import ImageData


def read_image_data(path, *, color_space="srgb", apply_orientation=True, icc_policy="reject"):
    """Read uint8/uint16 RGB, applying EXIF orientation exactly once.

    ICC transformation is deliberately not inferred. Embedded profiles are
    rejected unless the caller explicitly asserts that pixels may be interpreted
    as the declared color space with ``icc_policy='assume-declared'``. That choice
    and the original profile hash are retained in provenance. Legacy read_image
    behavior is unchanged.
    """
    if icc_policy not in ("reject", "assume-declared"):
        raise ValueError("icc_policy must be reject or assume-declared")
    if not isinstance(apply_orientation, bool):
        raise TypeError("apply_orientation must be boolean")
    path = Path(path)
    data = path.read_bytes()
    with Image.open(path) as metadata:
        if getattr(metadata, "n_frames", 1) != 1:
            raise ValueError("explicit image input requires a single frame")
        orientation = metadata.getexif().get(274, 1)
        profile = metadata.info.get("icc_profile")
    if profile and icc_policy == "reject":
        raise ValueError("embedded ICC requires explicit conversion or icc_policy='assume-declared'; no implicit color transform was performed")
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("cannot decode image: %s" % path)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("explicit image input requires three channels; choose grayscale/alpha handling before import")
    provenance = {"source_kind": "file", "filename": path.name,
                  "file_sha256": hashlib.sha256(data).hexdigest(), "decoder": "OpenCV IMREAD_UNCHANGED",
                  "icc_policy": icc_policy, "icc_present": bool(profile), "icc_transform": "none"}
    if profile:
        provenance["icc_sha256"] = hashlib.sha256(profile).hexdigest()
    return ImageData.from_array(pixels, channel_order="BGR", color_space=color_space,
                                orientation=orientation, apply_orientation=apply_orientation, provenance=provenance)
