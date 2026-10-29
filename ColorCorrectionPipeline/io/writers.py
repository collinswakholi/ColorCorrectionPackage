"""
Image writing utilities
=======================

This module provides utilities for writing images with proper color space
handling and format conversion.

Functions:
    - write_image: Write image to file
"""

import os
import hashlib
import json
import struct
import tempfile
import zlib
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

__all__ = ["write_image", "write_image_data", "ExportRecord"]


def write_image(
    path: Union[str, Path],
    image: np.ndarray,
    color_space: str = "rgb",
    quality: int = 95,
    create_dirs: bool = True,
) -> bool:
    """
    Write image to file with proper color space handling.
    
    Args:
        path: Output file path
        image: Image array (H, W, C) or (H, W)
        color_space: Input color space ('rgb', 'bgr', 'gray')
        quality: JPEG quality (1-100) or PNG compression (0-9)
        create_dirs: Whether to create parent directories
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> img_rgb = np.random.rand(480, 640, 3).astype(np.float64)
        >>> success = write_image("output.jpg", img_rgb, color_space="rgb", quality=95)
        >>> print(f"Write {'succeeded' if success else 'failed'}")
    """
    path = Path(path)
    
    # Create parent directories if needed
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype in (np.float32, np.float64):
        # Assume 0-1 range for float
        img_out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        # Scale 0-65535 to 0-255
        img_out = (image // 257).astype(np.uint8)
    else:
        img_out = image.astype(np.uint8)
    
    # Handle color space conversion (OpenCV writes BGR)
    if len(img_out.shape) == 2:
        # Grayscale - no conversion needed
        pass
    else:
        if color_space.lower() == "rgb":
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        elif color_space.lower() == "bgr":
            pass  # Already BGR
        elif color_space.lower() == "gray":
            if len(img_out.shape) == 3:
                img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)
        else:
            print(f"Warning: Unknown color_space '{color_space}', assuming BGR")
    
    # Set encoding parameters
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        # PNG compression level (0-9)
        compression = min(9, max(0, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        params = []
    
    # Write image
    try:
        success = cv2.imwrite(str(path), img_out, params)
        if success:
            print(f"Info: Image written to {path}")
        else:
            print(f"Error: Failed to write image to {path}")
        return success
    except Exception as e:
        print(f"Error: Exception writing image to {path}: {e}")
        return False


@dataclass(frozen=True)
class ExportRecord:
    """Facts about an atomically published image, without retained image arrays."""
    path: str
    format: str
    dtype: str
    shape: tuple
    encoding: str
    source_encoding: str
    source_dtype: str
    source_digest: str
    source_pixel_digest: str
    sha256: str
    bytes: int
    precision_loss: bool
    codec_lossy: bool
    maximum_quantization_error: float
    maximum_codec_error: float
    source_range: tuple
    source_channel_order: str
    source_orientation: int
    source_file_sha256: Optional[str]
    orientation_applied: bool = True
    schema: int = 1

    def to_dict(self):
        result = asdict(self)
        result["shape"] = list(self.shape)
        result["source_range"] = list(self.source_range)
        return result


def _png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


def _png_metadata(payload, encoding):
    """Declare transfer and primaries without changing encoded sample bytes.

    Native chunks follow W3C PNG3 section11.3.2; project text allows our decoder
    to preserve an explicit linear-sRGB interpretation rather than guess it.
    """
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("encoder did not produce PNG")
    ihdr_end = 8 + 12 + struct.unpack(">I", payload[8:12])[0]
    description = json.dumps({"ColorCorrectionPipeline": {"schema": 1, "encoding": encoding}}, sort_keys=True).encode("ascii")
    extra = (_png_chunk(b"sRGB", b"\x00") if encoding == "srgb" else b"")
    extra += _png_chunk(b"gAMA", struct.pack(">I", 45455 if encoding == "srgb" else 100000))
    extra += _png_chunk(b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000))
    extra += _png_chunk(b"tEXt", b"ColorCorrectionPipeline\0" + description)
    return payload[:ihdr_end] + extra + payload[ihdr_end:]


def _encode_strict(path, pixels, encoding, quality):
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        stream = io.BytesIO()
        description = json.dumps({"ColorCorrectionPipeline": {"schema": 1, "encoding": encoding}}, sort_keys=True)
        # No OpenCV LogLuv conversion or lossy compressor is used.
        tifffile.imwrite(stream, pixels, photometric="rgb", compression=None,
                         metadata=None, description=description, extratags=[(274, "H", 1, 1, False)])
        return stream.getvalue(), "TIFF", False
    params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, (100-quality)//10)] if suffix == ".png" else [
        cv2.IMWRITE_JPEG_QUALITY if suffix in (".jpg", ".jpeg") else cv2.IMWRITE_WEBP_QUALITY, quality]
    # This is an exact channel permutation, not a color transform. OpenCV's
    # native swap avoids NumPy's generic strided full-frame channel copy.
    ok, encoded = cv2.imencode(suffix, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise OSError("image encoder reported failure")
    payload = encoded.tobytes()
    if suffix == ".png":
        payload = _png_metadata(payload, encoding)
    return payload, {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[suffix], suffix != ".png"


def _encode_png_fast(pixels, encoding):
    """Explicit filter-None, level-zero RGB PNG; sample precision is unchanged.

    PNG3 sections 9 and 10 permit this lossless representation. Default encoding
    keeps its original bytes; only this selected profile changes the byte layout.
    Publication still uses the same complete byte/decode/pixel/metadata checks.
    """
    height, width = pixels.shape[:2]
    if not (0 < width < 2**31 and 0 < height < 2**31):
        raise ValueError("PNG dimensions must be positive and below 2**31")
    if pixels.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("fast PNG requires uint8 or uint16 RGB")
    network = pixels if pixels.dtype.itemsize == 1 else pixels.astype(">u2")
    rows = np.empty((height, width * 3 * pixels.dtype.itemsize + 1), dtype=np.uint8)
    rows[:, 0] = 0  # Each scanline explicitly declares filter None.
    rows[:, 1:] = network.view(np.uint8).reshape(height, -1)
    compressed = zlib.compress(memoryview(rows), 0)
    header = struct.pack(">IIBBBBB", width, height, pixels.dtype.itemsize * 8, 2, 0, 0, 0)
    # Bounded IDAT chunks also keep each CRC workspace bounded.
    chunks = [_png_chunk(b"IDAT", compressed[start:start + 1024*1024])
              for start in range(0, len(compressed), 1024*1024)]
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + b"".join(chunks) + _png_chunk(b"IEND", b"")
    return _png_metadata(payload, encoding)


_EXPORT_CHUNK_VALUES = 3 * 65536


def _quantize_strict(values, dtype, maximum):
    """Apply the existing rounding equation with bounded temporary workspace.

    Canonical values are owned immutable float64. Reusing them for float64 TIFF
    avoids a redundant full-frame copy; encoders receive read-only data. Error
    maxima use the same per-value operations as the original full-array formula.
    """
    values_flat = values.reshape(-1)
    if dtype == values.dtype:
        storage = values
    else:
        storage = np.empty(values.shape, dtype=dtype)
    storage_flat = storage.reshape(-1)
    if dtype.kind == "u":
        from .._image_numeric import try_quantize_unsigned
        accelerated_error = try_quantize_unsigned(values, storage, maximum)
        if accelerated_error is not None:
            return storage, accelerated_error
    quantization = 0.0
    for start in range(0, values_flat.size, _EXPORT_CHUNK_VALUES):
        stop = min(start + _EXPORT_CHUNK_VALUES, values_flat.size)
        source = values_flat[start:stop]
        if dtype.kind == "u":
            scaled = source * maximum
            np.rint(scaled, out=scaled)
            storage_flat[start:stop] = scaled.astype(dtype)
            # All uint8/uint16 integers are exact in float64. The rounded
            # workspace therefore equals storage.astype(float64), apart from
            # signed zero, which abs removes. Reuse it with the original
            # divide/subtract/abs order instead of recasting and allocating.
            scaled /= maximum
            scaled -= source
            np.abs(scaled, out=scaled)
            quantization = max(quantization, float(scaled.max()))
        elif storage is not values:
            storage_flat[start:stop] = source.astype(dtype)
            error = np.abs(storage_flat[start:stop].astype(np.float64) / maximum - source)
            quantization = max(quantization, float(error.max()))
    return storage, quantization


def _codec_error_strict(decoded, storage, maximum):
    """Keep lossy error reporting exact without full float64 frame temporaries."""
    # Decoded BGR->RGB may be a strided view; reshape by rows to avoid creating
    # a contiguous full-frame copy merely to bound comparison workspace.
    error = 0.0
    for region in _export_regions(storage.shape):
        difference = np.abs(decoded[region].astype(np.float64) / maximum
                            - storage[region].astype(np.float64) / maximum)
        error = max(error, float(difference.max()))
    return error


def _export_regions(shape):
    """Bound comparison workspace for both tall and extremely wide images."""
    rows = max(1, _EXPORT_CHUNK_VALUES // min(shape[1] * shape[2], _EXPORT_CHUNK_VALUES))
    columns = max(1, _EXPORT_CHUNK_VALUES // (rows * shape[2]))
    for top in range(0, shape[0], rows):
        for left in range(0, shape[1], columns):
            yield (slice(top, top + rows), slice(left, left + columns), slice(None))


def _verify_export_metadata(facts, fmt, encoding):
    if (facts.get("orientation") != 1 or facts.get("frame_count") != 1
            or facts.get("frame_index") != 0 or facts.get("profile")
            or facts.get("associated_alpha", False)):
        raise OSError("encoded image read-back contradicts canonical RGB interpretation")
    if fmt == "TIFF" and facts.get("photometric") != "RGB":
        raise OSError("encoded TIFF read-back contradicts RGB photometric interpretation")
    declared = facts.get("declared_encoding")
    if declared != encoding and (fmt in ("PNG", "TIFF") or declared is not None):
        raise OSError("encoded image read-back contradicts declared transfer encoding")


def write_image_data(path, image, *, dtype="preserve", input_encoding=None, output_encoding=None,
                     channel_order="RGB", value_range=None, allow_precision_loss=False,
                     quality=95, create_dirs=True, overwrite=False, png_profile="compatible"):
    """Publish a validated RGB export atomically and return its actual facts.

    Defaults preserve source storage precision. PNG supports uint8/uint16; TIFF
    additionally supports normalized float32/float64 RGB. Lossy preview formats
    require uint8, and reducing source precision requires explicit consent.
    Existing destinations are preserved unless ``overwrite=True`` is requested.
    ``png_profile='fast-lossless'`` selects filter None and compression level 0;
    it overrides PNG quality/compression, changes encoded bytes, and retains all
    sample, metadata and strict publication guarantees. The default is unchanged.
    """
    from ..contracts import ImageData
    from ..colorimetry import srgb_to_linear, linear_to_srgb
    from .image_data import _decode
    for key, value in (("allow_precision_loss", allow_precision_loss), ("create_dirs", create_dirs), ("overwrite", overwrite)):
        if type(value) is not bool:
            raise TypeError(key + " must be boolean")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality must be an integer from 1 to 100")
    if png_profile not in ("compatible", "fast-lossless"):
        raise ValueError("png_profile must be compatible or fast-lossless")
    if input_encoding not in (None, "srgb", "linear-srgb") or output_encoding not in (None, "srgb", "linear-srgb"):
        raise ValueError("input/output encoding must be srgb or linear-srgb")
    if isinstance(image, ImageData):
        if value_range is not None or channel_order != "RGB":
            raise ValueError("typed ImageData already declares its range and RGB channel order")
        if input_encoding is not None and input_encoding != image.color_space:
            raise ValueError("input_encoding conflicts with ImageData")
        source = image
    else:
        source = ImageData.from_array(image, channel_order=channel_order, color_space=input_encoding or "srgb", value_range=value_range)
    prepared = source.with_orientation_applied()
    encoding = output_encoding or source.color_space
    # ImageData owns immutable pixels. All conversions below produce new arrays;
    # quantization/encoding never mutate this borrowed canonical source.
    # A view also owns a separate ndarray header: caller changes to the public
    # source ndarray's shape/dtype cannot alter this in-flight working view.
    values = prepared.pixels.view()
    if source.color_space != encoding:
        values = srgb_to_linear(values) if encoding == "linear-srgb" else linear_to_srgb(values, policy="raise")
    if dtype == "preserve":
        dtype = source.source_dtype
        if dtype == "float16":
            dtype = "float32"  # lossless promotion to a widely supported TIFF sample type
    if dtype not in ("uint8", "uint16", "float32", "float64"):
        raise ValueError("dtype must be preserve, uint8, uint16, float32 or float64")
    ranks = {"uint8": 8, "uint16": 16, "float16": 16, "float32": 32, "float64": 64}
    precision_loss = (ranks[dtype] < ranks[source.source_dtype]
                      or (np.dtype(source.source_dtype).kind == "f" and np.dtype(dtype).kind == "u")
                      or (source.color_space != encoding and np.dtype(dtype).kind == "u"))
    if precision_loss and not allow_precision_loss:
        raise ValueError("requantizing converted RGB or reducing source precision requires allow_precision_loss=True")
    path = Path(path)
    suffix = path.suffix.lower()
    if png_profile != "compatible" and suffix != ".png":
        raise ValueError("fast-lossless PNG profile requires a .png destination")
    supported = {".png": {"uint8", "uint16"}, ".tif": {"uint8", "uint16", "float32", "float64"},
                 ".tiff": {"uint8", "uint16", "float32", "float64"}, ".jpg": {"uint8"}, ".jpeg": {"uint8"}, ".webp": {"uint8"}}
    if suffix not in supported or dtype not in supported[suffix]:
        raise ValueError("requested format cannot preserve selected dtype; explicitly choose a supported format/dtype")
    if suffix in (".jpg", ".jpeg", ".webp") and encoding != "srgb":
        raise ValueError("JPEG/WebP preview export requires output_encoding='srgb'")
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    if path.exists() and not path.is_file():
        raise ValueError("output destination must be a file")
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        raise FileNotFoundError(str(path.parent))
    target_dtype = np.dtype(dtype)
    maximum = np.iinfo(target_dtype).max if target_dtype.kind == "u" else 1.0
    storage, quantization = _quantize_strict(values, target_dtype, maximum)
    if png_profile == "fast-lossless":
        payload, fmt, codec_lossy = _encode_png_fast(storage, encoding), "PNG", False
    else:
        payload, fmt, codec_lossy = _encode_strict(path, storage, encoding, quality)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        saved = temporary.read_bytes()
        if saved != payload:
            raise OSError("temporary image read-back differs from encoded bytes")
        pixels = storage.shape[0] * storage.shape[1]
        try:
            decoded, facts = _decode(saved, path, 0, max_pixels=pixels,
                max_decoded_bytes=storage.nbytes + storage.size * 8)
        except ValueError as error:
            if "exceed" in str(error) and ("max_pixels" in str(error) or "max_decoded_bytes" in str(error)):
                raise OSError("encoded image read-back exceeds requested image allocation") from error
            raise
        if facts["channel_order"] == "BGR":
            decoded = decoded[..., ::-1]
        if decoded.shape != storage.shape or decoded.dtype != storage.dtype or facts["format"] != fmt:
            raise OSError("encoded image read-back contradicts requested shape/dtype/format")
        _verify_export_metadata(facts, fmt, encoding)
        if not codec_lossy and any(not np.array_equal(decoded[region], storage[region]) for region in _export_regions(storage.shape)):
            raise OSError("lossless export did not reproduce encoded samples exactly")
        # Exact sample equality above proves a zero codec error for lossless
        # formats; normalizing/copying both full frames cannot add evidence.
        codec_error = _codec_error_strict(decoded, storage, maximum) if codec_lossy else 0.0
        if overwrite:
            os.replace(temporary, path)
        elif os.name == "nt":
            # Windows rename is atomic and refuses an existing destination.
            os.rename(temporary, path)
        else:
            # Atomic no-clobber publication on the same filesystem. A race with
            # another writer fails without truncating or replacing its output.
            os.link(temporary, path)
        return ExportRecord(str(path.resolve()), fmt, dtype, tuple(storage.shape), encoding, source.color_space,
                            source.source_dtype, source.digest, source.pixel_digest, hashlib.sha256(saved).hexdigest(),
                            len(saved), precision_loss, codec_lossy, quantization, codec_error,
                            tuple(source.source_range), source.source_channel_order, source.source_orientation,
                            source.provenance.get("file_sha256"))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
