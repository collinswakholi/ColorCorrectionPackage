"""Explicit decoding that preserves precision and records preprocessing choices."""

import hashlib
import io
import json
import math
import stat
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

from ..contracts import ImageData
from ..colorimetry import srgb_to_linear, linear_to_srgb

_EXPORT_KEY = "ColorCorrectionPipeline"


class _DuplicateMetadataKey(ValueError):
    pass


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateMetadataKey("duplicate image encoding metadata key")
        result[key] = value
    return result


def _project_encoding(text, *, owned=False):
    if not text:
        if owned:
            raise ValueError("empty image encoding metadata")
        return None
    if not isinstance(text, str) or len(text) > 65536:
        if owned or not isinstance(text, str) or _EXPORT_KEY in text:
            raise ValueError("invalid or oversized image encoding metadata")
        return None  # Large ordinary TIFF/OME descriptions are not our schema.
    try:
        metadata = json.loads(text, object_pairs_hook=_unique_keys)
    except _DuplicateMetadataKey:
        if owned or _EXPORT_KEY in text:
            raise ValueError("duplicate image encoding metadata key")
        return None
    except (ValueError, TypeError):
        if owned:
            raise ValueError("invalid image encoding metadata")
        return None  # Ordinary TIFF descriptions are not our schema.
    if not isinstance(metadata, dict) or _EXPORT_KEY not in metadata:
        if owned:
            raise ValueError("invalid image encoding metadata")
        return None
    if set(metadata) != {_EXPORT_KEY}:
        raise ValueError("unsupported image encoding metadata fields")
    declared = metadata[_EXPORT_KEY]
    if not isinstance(declared, dict) or set(declared) != {"schema", "encoding"} or type(declared.get("schema")) is not int or declared.get("schema") != 1 or declared.get("encoding") not in ("srgb", "linear-srgb"):
        raise ValueError("unsupported image encoding metadata")
    return declared["encoding"]


def _frame(value, count):
    if value is None:
        if count != 1:
            raise ValueError("explicit image input requires a single frame; choose frame_index explicitly")
        return 0
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("frame_index must select an existing nonnegative frame")
    return value


def _profile_decision(profile, policy):
    if not profile or policy is None:
        return
    if policy == "reject":
        raise ValueError("embedded ICC requires explicit conversion or icc_policy='assume-declared'; no implicit color transform was performed")
    if not isinstance(profile, bytes) or len(profile) < 128 or not 128 <= int.from_bytes(profile[:4], "big") <= len(profile) or profile[36:40] != b"acsp":
        raise ValueError("embedded ICC profile header is malformed")
    if policy == "convert-srgb":
        # Reject bad profiles before a native image decoder sees their chunks.
        ImageCms.ImageCmsProfile(io.BytesIO(profile))


def _check_allocation(width, height, native_bytes, max_pixels, max_decoded_bytes):
    pixels = int(width) * int(height)
    if pixels < 1:
        raise ValueError("image has invalid empty dimensions")
    if max_pixels is not None and pixels > max_pixels:
        raise ValueError("image dimensions exceed max_pixels before pixel decoding")
    # Count at least decoded source plus the canonical float64 RGB allocation.
    # Decoder workspaces and later stage intermediates require additional memory.
    if max_decoded_bytes is not None and int(native_bytes) + pixels * 3 * 8 > max_decoded_bytes:
        raise ValueError("decoded source plus canonical RGB exceed max_decoded_bytes before pixel decoding")


def _decode(data, path, frame_index=None, icc_policy=None, max_pixels=None, max_decoded_bytes=None):
    """Decode raw pixels without orientation/color transforms or bit reduction."""
    if data[:4] in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"):
        import tifffile
        with tifffile.TiffFile(io.BytesIO(data)) as image:
            count = len(image.pages)
            selected = _frame(frame_index, count)
            page = image.pages[selected]
            photometric = page.photometric.name
            if photometric not in ("RGB", "MINISBLACK", "MINISWHITE"):
                raise ValueError("unsupported TIFF photometric interpretation: %s" % photometric)
            if len(page.shape) not in (2, 3) or page.samplesperpixel not in (1, 3, 4):
                raise ValueError("unsupported TIFF sample layout before decoding")
            expected = ((page.imagelength, page.imagewidth) if page.samplesperpixel == 1 else
                        (page.samplesperpixel, page.imagelength, page.imagewidth) if page.planarconfig == 2 else
                        (page.imagelength, page.imagewidth, page.samplesperpixel))
            if tuple(page.shape) != tuple(expected):
                raise ValueError("unsupported volumetric TIFF sample layout before decoding")
            _check_allocation(page.imagewidth, page.imagelength, math.prod(page.shape) * page.dtype.itemsize,
                              max_pixels, max_decoded_bytes)
            profile = page.tags[34675].value if 34675 in page.tags else None
            _profile_decision(profile, icc_policy)
            pixels = page.asarray()
            if page.planarconfig == 2 and pixels.ndim == 3:
                pixels = np.moveaxis(pixels, 0, -1)
            return pixels, {"decoder": "tifffile", "format": "TIFF",
                            "orientation": page.tags[274].value if 274 in page.tags else 1,
                            "profile": profile,
                            "declared_encoding": _project_encoding(page.description),
                            "frame_index": selected, "frame_count": count, "mode": photometric,
                            "photometric": photometric, "channel_order": "RGB",
                            "associated_alpha": any(int(value) == 1 for value in page.extrasamples)}
    with Image.open(io.BytesIO(data)) as image:
        count = getattr(image, "n_frames", 1)
        selected = _frame(frame_index, count)
        image.seek(selected)
        orientation = image.getexif().get(274, 1)
        profile = image.info.get("icc_profile")
        _profile_decision(profile, icc_policy)
        declared = _project_encoding(image.info.get(_EXPORT_KEY), owned=_EXPORT_KEY in image.info)
        mode, fmt = image.mode, image.format
        sample_bytes = (2 if fmt == "PNG" and len(data) > 24 and data[24] == 16 else
                        1 if fmt in ("PNG", "JPEG", "WEBP", "GIF", "BMP") else 8)
        channels = (4 if mode in ("RGBA", "RGBa", "LA", "La", "CMYK") or "transparency" in image.info else
                    1 if mode in ("1", "L", "I", "F", "I;16", "I;16B", "I;16L") else 3)
        native_bytes = image.width * image.height * channels * sample_bytes
        _check_allocation(image.width, image.height, native_bytes, max_pixels, max_decoded_bytes)
        if mode == "CMYK":
            pixels, order = np.asarray(image).copy(), "CMYK"
        elif selected:
            preview = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if preview is not None and preview.dtype == np.uint16:
                raise ValueError("high-precision non-TIFF frame selection is unsupported; no uint8 downgrade performed")
            converted = image.convert("RGBA" if "transparency" in image.info or "A" in image.mode else "RGB")
            pixels, order = np.asarray(converted).copy(), "RGB"
        else:
            pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if pixels is None:
                if fmt not in ("GIF",):
                    raise ValueError("cannot decode image: %s" % path)
                converted = image.convert("RGBA" if "transparency" in image.info else "RGB")
                pixels, order = np.asarray(converted).copy(), "RGB"
            else:
                order = "BGR"
    return pixels, {"decoder": "Pillow selected frame" if selected or fmt == "GIF" else "OpenCV IMREAD_UNCHANGED",
                    "format": fmt, "orientation": orientation, "profile": profile,
                    "declared_encoding": declared, "frame_index": selected, "frame_count": count,
                    "mode": mode, "channel_order": order, "associated_alpha": mode in ("RGBa", "La")}


def read_image_data(path, *, color_space=None, apply_orientation=True, icc_policy="reject",
                    alpha_policy="reject", alpha_background=(0.0, 0.0, 0.0),
                    grayscale_policy="reject", frame_index=None,
                    max_pixels=100_000_000, max_decoded_bytes=1_073_741_824,
                    max_file_bytes=1_073_741_824):
    """Read RGB without silently reducing sample precision.

    Ordinary files default to sRGB. Our exports declare their encoding; explicit
    conflicting caller interpretation fails. ICC conversion requires a request
    and supported uint8 samples. Alpha compositing occurs in linear light.
    """
    if color_space not in (None, "srgb", "linear-srgb"):
        raise ValueError("color_space must be srgb or linear-srgb")
    if icc_policy not in ("reject", "assume-declared", "convert-srgb"):
        raise ValueError("icc_policy must be reject, assume-declared or convert-srgb")
    if type(apply_orientation) is not bool:
        raise TypeError("apply_orientation must be boolean")
    for key, value in (("max_pixels", max_pixels), ("max_decoded_bytes", max_decoded_bytes), ("max_file_bytes", max_file_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(key + " must be a positive integer")
    if alpha_policy not in ("reject", "drop", "composite") or grayscale_policy not in ("reject", "replicate"):
        raise ValueError("unsupported alpha_policy or grayscale_policy")
    background = np.asarray(alpha_background, dtype=np.float64)
    if background.shape != (3,) or not np.isfinite(background).all() or np.any((background < 0) | (background > 1)):
        raise ValueError("alpha_background must be three finite normalized RGB values")
    path = Path(path)
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("explicit image input requires a regular file")
    if file_stat.st_size > max_file_bytes:
        raise ValueError("encoded file exceeds max_file_bytes before reading")
    with path.open("rb") as stream:
        data = stream.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise ValueError("encoded file grew beyond max_file_bytes while reading")
    pixels, facts = _decode(data, path, frame_index, icc_policy, max_pixels, max_decoded_bytes)
    original_shape, original_dtype = list(pixels.shape), pixels.dtype.name
    profile, declared = facts.pop("profile"), facts.pop("declared_encoding")
    selected_space = color_space or declared or "srgb"
    if declared is not None and color_space is not None and color_space != declared:
        raise ValueError("declared color_space conflicts with exported image encoding metadata")
    if profile and icc_policy == "reject":
        raise ValueError("embedded ICC requires explicit conversion or icc_policy='assume-declared'; no implicit color transform was performed")
    if profile and icc_policy == "convert-srgb" and color_space not in (None, "srgb"):
        raise ValueError("ICC convert-srgb requires an sRGB output interpretation")
    if pixels.dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"), np.dtype("float64")):
        raise TypeError("decoded dtype must be uint8, uint16, float32 or float64")
    if pixels.ndim not in (2, 3) or (pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4)):
        raise ValueError("unsupported decoded channel layout")
    if facts["channel_order"] == "BGR" and pixels.ndim == 3 and pixels.shape[2] >= 3:
        pixels = pixels[..., [2, 1, 0, 3] if pixels.shape[2] == 4 else [2, 1, 0]]
    icc_transform = "none"
    if facts["channel_order"] == "CMYK" and not (profile and icc_policy == "convert-srgb"):
        raise ValueError("CMYK input requires an embedded ICC profile and explicit convert-srgb")
    if profile and icc_policy == "convert-srgb":
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("ICC conversion supports uint8 RGB/CMYK only; no high-precision downgrade performed")
        source_mode = "CMYK" if facts["channel_order"] == "CMYK" else "RGB"
        color_pixels = pixels if source_mode == "CMYK" else pixels[..., :3]
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        converted = ImageCms.profileToProfile(Image.fromarray(color_pixels, mode=source_mode), source_profile,
                                              ImageCms.createProfile("sRGB"), outputMode="RGB", renderingIntent=1)
        converted = np.asarray(converted)
        if pixels.shape[2] == 4 and source_mode != "CMYK":
            converted = np.concatenate((converted, pixels[..., 3:4]), axis=2)
        pixels, selected_space = converted, "srgb"
        icc_transform = "Pillow ImageCms profileToProfile to sRGB, relative-colorimetric intent1"
    if pixels.ndim == 2 or pixels.shape[2] == 1:
        if grayscale_policy != "replicate":
            raise ValueError("explicit image input requires three channels; choose grayscale_policy='replicate'")
        if pixels.ndim == 3:
            pixels = pixels[..., 0]
        if facts.get("photometric") == "MINISWHITE":
            maximum = np.iinfo(pixels.dtype).max if pixels.dtype.kind == "u" else 1.0
            pixels = maximum - pixels
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.shape[2] == 4:
        if alpha_policy == "reject":
            raise ValueError("explicit image input requires three channels; choose an alpha_policy")
        if facts["associated_alpha"]:
            raise ValueError("associated/premultiplied alpha requires explicit external normalization")
        if alpha_policy == "drop":
            pixels = pixels[..., :3]
        else:
            maximum = np.iinfo(pixels.dtype).max if pixels.dtype.kind == "u" else 1.0
            values = pixels.astype(np.float64) / maximum
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("alpha samples must lie in their normalized finite range")
            color, opacity = values[..., :3], values[..., 3:4]
            pixels = (linear_to_srgb(srgb_to_linear(color) * opacity + srgb_to_linear(background) * (1-opacity), policy="raise")
                      if selected_space == "srgb" else color * opacity + background * (1-opacity))
    provenance = {"source_kind": "file", "filename": path.name,
                  "file_sha256": hashlib.sha256(data).hexdigest(), **facts,
                  "decoded_shape": original_shape, "decoded_dtype": original_dtype,
                  "icc_policy": icc_policy, "icc_present": bool(profile), "icc_transform": icc_transform,
                  "declared_file_encoding": declared, "alpha_policy": alpha_policy,
                  "alpha_background": background.tolist(), "alpha_compositing_domain": "linear-light",
                  "grayscale_policy": grayscale_policy}
    provenance["decode_limits"] = {"max_pixels": max_pixels, "max_decoded_bytes": max_decoded_bytes,
                                   "max_file_bytes": max_file_bytes,
                                   "byte_basis": "decoded source plus canonical float64 RGB; excludes decoder workspaces/stage intermediates"}
    if profile:
        provenance["icc_sha256"] = hashlib.sha256(profile).hexdigest()
    return ImageData.from_array(pixels, channel_order="RGB", color_space=selected_space,
                                orientation=int(facts["orientation"]), apply_orientation=apply_orientation,
                                provenance=provenance)
