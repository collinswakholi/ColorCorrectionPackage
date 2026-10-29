"""Precision, interpretation and transactional image I/O contracts."""
import hashlib
import io
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms, PngImagePlugin

from ColorCorrectionPipeline import ImageData
from ColorCorrectionPipeline.colorimetry import linear_to_srgb, srgb_to_linear
from ColorCorrectionPipeline.io import read_image, read_image_data, write_image, write_image_data
from ColorCorrectionPipeline.io import writers


def colors(dtype="uint16"):
    values = np.array([[[3, 17, 253], [999, 32000, 65001]], [[4321, 5678, 9012], [65535, 0, 32769]]], dtype=np.uint16)
    if dtype == "uint8":
        return (values // 257).astype(dtype)
    if dtype.startswith("float"):
        return (values / 65535).astype(dtype)
    return values.astype(dtype)


@pytest.mark.parametrize("suffix,dtype", [(".png", "uint8"), (".png", "uint16"),
    (".tiff", "uint8"), (".tiff", "uint16"), (".tiff", "float32"), (".tiff", "float64")])
def test_lossless_storage_round_trip_preserves_dtype_and_asymmetric_channels(tmp_path, suffix, dtype):
    source = colors(dtype)
    path = tmp_path / ("image" + suffix)
    record = write_image_data(path, source)
    restored = read_image_data(path)
    assert record.dtype == dtype and restored.source_dtype == dtype
    expected = source.astype(np.float64) / (np.iinfo(dtype).max if source.dtype.kind == "u" else 1)
    np.testing.assert_array_equal(restored.pixels, expected)
    assert record.maximum_codec_error == 0 and not record.codec_lossy
    assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.bytes == path.stat().st_size
    assert record.shape == source.shape and record.encoding == "srgb"
    json.dumps(record.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        record.dtype = "uint8"


def test_tiff_float_export_is_ieee_rgb_not_opencv_logluv(tmp_path):
    source = np.random.default_rng(78).uniform(0, 1, (17, 23, 3))
    path = tmp_path / "scientific.tiff"
    write_image_data(path, source)
    with tifffile.TiffFile(path) as image:
        page = image.pages[0]
        assert page.photometric.name == "RGB" and page.dtype == np.float64
        assert page.compression.name == "NONE"
        np.testing.assert_array_equal(page.asarray(), source)


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_linear_encoding_metadata_survives_and_conflicting_assumption_rejected(tmp_path, suffix):
    path = tmp_path / ("linear" + suffix)
    image = ImageData.from_array(colors("uint16"), color_space="linear-srgb")
    record = write_image_data(path, image)
    assert record.encoding == "linear-srgb"
    restored = read_image_data(path)
    assert restored.color_space == "linear-srgb"
    np.testing.assert_array_equal(restored.pixels, image.pixels)
    with pytest.raises(ValueError, match="conflicts"):
        read_image_data(path, color_space="srgb")
    if suffix == ".png":
        with Image.open(path) as png:
            assert png.info["gamma"] == 1
            assert "srgb" not in png.info


def test_explicit_encoding_conversion_uses_transfer_only_and_reports_quantization(tmp_path):
    image = ImageData.from_array(colors("float64"), color_space="linear-srgb")
    path = tmp_path / "encoded.png"
    record = write_image_data(path, image, output_encoding="srgb", dtype="uint16", allow_precision_loss=True)
    expected = np.rint(linear_to_srgb(image.pixels) * 65535) / 65535
    np.testing.assert_array_equal(read_image_data(path).pixels, expected)
    assert record.source_encoding == "linear-srgb" and record.encoding == "srgb"
    assert record.precision_loss and record.maximum_quantization_error <= .5/65535 + 1e-15


@pytest.mark.parametrize("dtype,source_encoding,target_encoding", [
    ("uint8", "srgb", "linear-srgb"), ("uint16", "linear-srgb", "srgb")])
def test_integer_encoding_conversion_requires_requantization_consent(tmp_path, dtype, source_encoding, target_encoding):
    source = ImageData.from_array(colors(dtype), color_space=source_encoding)
    path = tmp_path / "converted.png"
    with pytest.raises(ValueError, match="precision"):
        write_image_data(path, source, output_encoding=target_encoding)
    assert not path.exists()
    record = write_image_data(path, source, output_encoding=target_encoding, allow_precision_loss=True)
    assert record.precision_loss and record.encoding == target_encoding
    assert record.maximum_quantization_error <= .5/np.iinfo(dtype).max + 1e-15
    same = write_image_data(tmp_path / "same.png", source, output_encoding=source_encoding)
    assert not same.precision_loss
    np.testing.assert_array_equal(read_image_data(same.path).pixels, source.pixels)


def test_float16_preserve_promotes_losslessly_to_float32_tiff(tmp_path):
    source = colors("float16")
    record = write_image_data(tmp_path / "promoted.tif", source)
    assert record.dtype == "float32" and not record.precision_loss
    np.testing.assert_array_equal(read_image_data(record.path).pixels, source.astype(np.float64))


@pytest.mark.parametrize("source,dtype,suffix", [("uint16", "uint8", ".jpg"), ("float64", "uint16", ".png"),
    ("float64", "float32", ".tif"), ("float32", "uint8", ".png"), ("float16", "uint16", ".png")])
def test_precision_reduction_requires_consent(tmp_path, source, dtype, suffix):
    path = tmp_path / ("out" + suffix)
    with pytest.raises(ValueError, match="precision"):
        write_image_data(path, colors(source), dtype=dtype)
    assert not path.exists()
    record = write_image_data(path, colors(source), dtype=dtype, allow_precision_loss=True)
    assert record.precision_loss


@pytest.mark.parametrize("suffix,dtype", [(".png", "float32"), (".jpg", "uint16"), (".webp", "float64"), (".bmp", "uint8")])
def test_format_never_silently_reduces_dtype(tmp_path, suffix, dtype):
    with pytest.raises(ValueError, match="format"):
        write_image_data(tmp_path / ("unsupported" + suffix), colors(dtype), allow_precision_loss=True)


def test_jpeg_is_declared_lossy_preview_and_linear_requires_conversion(tmp_path):
    source = np.repeat(np.repeat(colors("uint8"), 16, axis=0), 16, axis=1)
    record = write_image_data(tmp_path / "preview.jpg", source, quality=90)
    assert record.codec_lossy and record.maximum_codec_error > 0
    with pytest.raises(ValueError, match="sRGB|srgb"):
        write_image_data(tmp_path / "linear.jpg", ImageData.from_array(source, color_space="linear-srgb"))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied_once_and_fresh_export_has_identity(tmp_path, orientation):
    source = np.repeat(np.repeat(colors("uint8"), 8, axis=0), 8, axis=1)
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(source).save(path, exif=exif)
    loaded = read_image_data(path)
    expected = ImageData.from_array(source, orientation=orientation)
    np.testing.assert_array_equal(loaded.pixels, expected.pixels)
    pending = read_image_data(path, apply_orientation=False)
    target = tmp_path / "fresh.png"
    write_image_data(target, pending)
    fresh = read_image_data(target)
    assert fresh.source_orientation == 1
    np.testing.assert_array_equal(fresh.pixels, loaded.pixels)


def test_tiff_orientation_preserves_uint16_and_is_applied_once(tmp_path):
    path = tmp_path / "oriented.tif"
    source = colors()
    tifffile.imwrite(path, source, photometric="rgb", extratags=[(274, "H", 1, 6, False)])
    loaded = read_image_data(path)
    np.testing.assert_array_equal(loaded.pixels, np.rot90(source, 3)/65535)
    assert loaded.source_dtype == "uint16" and loaded.source_orientation == 6


def test_metadata_and_pixels_use_same_snapshot_bytes(tmp_path, monkeypatch):
    first, second = io.BytesIO(), io.BytesIO()
    Image.fromarray(colors("uint8")).save(first, format="PNG")
    Image.fromarray(np.zeros((7, 11, 3), np.uint8)).save(second, format="PNG")
    path = tmp_path / "changing.png"
    path.write_bytes(first.getvalue())
    original = Path.open
    def read_then_replace(value, mode="r", *args, **kwargs):
        if value == path and mode == "rb":
            path.write_bytes(second.getvalue())
            return io.BytesIO(first.getvalue())
        return original(value, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read_then_replace)
    loaded = read_image_data(path)
    assert loaded.provenance["file_sha256"] == hashlib.sha256(first.getvalue()).hexdigest()
    np.testing.assert_array_equal(loaded.pixels, colors("uint8")/255)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_alpha_reject_drop_and_linear_light_composite(tmp_path, dtype):
    maximum = np.iinfo(dtype).max
    rgb = colors(dtype)
    alpha = np.array([[0, maximum], [maximum//2, maximum//4]], dtype=dtype)[..., None]
    rgba = np.concatenate((rgb, alpha), axis=2)
    path = tmp_path / "rgba.png"
    assert cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]])
    with pytest.raises(ValueError, match="alpha"):
        read_image_data(path)
    dropped = read_image_data(path, alpha_policy="drop")
    np.testing.assert_array_equal(dropped.pixels, rgb/maximum)
    background = [.2, .4, .6]
    result = read_image_data(path, alpha_policy="composite", alpha_background=background)
    expected = linear_to_srgb(srgb_to_linear(rgb/maximum) * (alpha/maximum) + srgb_to_linear(background) * (1-alpha/maximum))
    np.testing.assert_array_equal(result.pixels, expected)
    assert result.provenance["decoded_dtype"] == dtype


def test_associated_alpha_is_not_guessed(tmp_path):
    path = tmp_path / "associated.tif"
    rgba = np.ones((3, 5, 4), np.float32) * .5
    tifffile.imwrite(path, rgba, photometric="rgb", extrasamples=["ASSOCALPHA"])
    with pytest.raises(ValueError, match="premultiplied"):
        read_image_data(path, alpha_policy="composite")


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_grayscale_requires_explicit_replication(tmp_path, dtype):
    source = colors(dtype)[..., 0]
    path = tmp_path / "gray.tif"
    tifffile.imwrite(path, source)
    with pytest.raises(ValueError, match="grayscale"):
        read_image_data(path)
    result = read_image_data(path, grayscale_policy="replicate")
    expected = source / (np.iinfo(dtype).max if source.dtype.kind == "u" else 1)
    np.testing.assert_array_equal(result.pixels, np.repeat(expected[..., None], 3, axis=2))


def test_white_is_zero_tiff_gray_is_not_inverted_incorrectly(tmp_path):
    source = colors()[..., 0]
    path = tmp_path / "whitezero.tif"
    tifffile.imwrite(path, source, photometric="miniswhite")
    result = read_image_data(path, grayscale_policy="replicate")
    np.testing.assert_array_equal(result.pixels[..., 0], (65535-source)/65535)


def test_multiframe_tiff_requires_selection_and_keeps_native_samples(tmp_path):
    path = tmp_path / "frames.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(colors(), photometric="rgb")
        writer.write(colors()//2, photometric="rgb")
    with pytest.raises(ValueError, match="frame_index"):
        read_image_data(path)
    result = read_image_data(path, frame_index=1)
    np.testing.assert_array_equal(result.pixels, (colors()//2)/65535)
    assert result.provenance["frame_count"] == 2 and result.provenance["frame_index"] == 1
    for invalid in (-1, 2, True, "0"):
        with pytest.raises(ValueError, match="frame_index"):
            read_image_data(path, frame_index=invalid)


def test_icc_decisions_and_explicit_conversion_are_recorded(tmp_path):
    path = tmp_path / "profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(colors("uint8")).save(path, icc_profile=profile)
    with pytest.raises(ValueError, match="embedded ICC"):
        read_image_data(path)
    assumed = read_image_data(path, icc_policy="assume-declared")
    np.testing.assert_array_equal(assumed.pixels, colors("uint8")/255)
    converted = read_image_data(path, icc_policy="convert-srgb")
    np.testing.assert_array_equal(converted.pixels, colors("uint8")/255)
    assert "profileToProfile" in converted.provenance["icc_transform"]
    assert converted.provenance["icc_sha256"] == hashlib.sha256(profile).hexdigest()
    with pytest.raises(ValueError, match="sRGB"):
        read_image_data(path, icc_policy="convert-srgb", color_space="linear-srgb")


def test_high_precision_icc_conversion_never_downgrades_to_uint8(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "profile16.tif"
    tifffile.imwrite(path, colors(), photometric="rgb", extratags=[(34675, "B", len(profile), profile, False)])
    with pytest.raises(ValueError, match="no high-precision downgrade"):
        read_image_data(path, icc_policy="convert-srgb")
    np.testing.assert_array_equal(read_image_data(path, icc_policy="assume-declared").pixels, colors()/65535)


def test_invalid_icc_profile_conversion_fails_instead_of_relabeling(tmp_path):
    path = tmp_path / "invalid-profile.png"
    Image.fromarray(colors("uint8")).save(path, icc_profile=b"not an ICC profile")
    with pytest.raises((ValueError, OSError, ImageCms.PyCMSError)):
        read_image_data(path, icc_policy="convert-srgb")


def test_icc_rejection_occurs_before_native_pixel_decode(tmp_path, monkeypatch):
    path = tmp_path / "invalid-profile.png"
    Image.fromarray(colors("uint8")).save(path, icc_profile=b"not an ICC profile")
    def forbidden(*args, **kwargs):
        raise AssertionError("native decode must not receive a rejected ICC image")
    monkeypatch.setattr(cv2, "imdecode", forbidden)
    with pytest.raises(ValueError, match="embedded ICC"):
        read_image_data(path)
    with pytest.raises(ValueError, match="header"):
        read_image_data(path, icc_policy="convert-srgb")


@pytest.mark.parametrize("kwargs", [{"alpha_policy": "guess"}, {"grayscale_policy": "guess"},
    {"icc_policy": "guess"}, {"color_space": "AdobeRGB"}, {"apply_orientation": 1},
    {"alpha_background": [0, 0, 2]}, {"alpha_background": [0, np.nan, 0]}])
def test_invalid_read_options_fail_before_decoding(tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        read_image_data(tmp_path / "missing.png", **kwargs)


def test_unknown_tiff_photometric_is_not_reinterpreted_as_gray(tmp_path):
    path = tmp_path / "palette.tif"
    tifffile.imwrite(path, np.zeros((8, 9), np.uint8), photometric="palette", colormap=np.zeros((3, 256), np.uint16))
    with pytest.raises(ValueError, match="photometric"):
        read_image_data(path, grayscale_policy="replicate")


def test_float_outside_contract_fails_without_silent_clipping(tmp_path):
    path = tmp_path / "hdr.tif"
    tifffile.imwrite(path, np.full((4, 5, 3), 1.2), photometric="rgb")
    with pytest.raises(ValueError, match="range"):
        read_image_data(path)


def test_no_clobber_default_and_explicit_atomic_replacement(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        write_image_data(path, colors())
    assert path.read_bytes() == b"previous"
    write_image_data(path, colors(), overwrite=True)
    np.testing.assert_array_equal(read_image_data(path).pixels, colors()/65535)
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("failure", ["encoder", "fsync", "replace", "readback"])
def test_write_failure_preserves_destination_and_removes_temporary(tmp_path, monkeypatch, failure):
    path = tmp_path / "image.png"
    path.write_bytes(b"previous")
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    if failure == "encoder":
        monkeypatch.setattr(writers, "_encode_strict", fail)
    elif failure == "fsync":
        monkeypatch.setattr(writers.os, "fsync", fail)
    elif failure == "replace":
        monkeypatch.setattr(writers.os, "replace", fail)
    else:
        original = writers._encode_strict
        def wrong_pixels(path, pixels, encoding, quality):
            wrong = pixels.copy()
            wrong[0, 0, 0] ^= 1
            return original(path, wrong, encoding, quality)
        monkeypatch.setattr(writers, "_encode_strict", wrong_pixels)
    with pytest.raises(OSError):
        write_image_data(path, colors(), overwrite=True)
    assert path.read_bytes() == b"previous" and list(tmp_path.glob(".*.tmp")) == []


def test_racing_destination_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "raced.png"
    primitive = "rename" if os.name == "nt" else "link"
    original = getattr(writers.os, primitive)
    def raced(source, target):
        Path(target).write_bytes(b"other writer")
        return original(source, target)
    monkeypatch.setattr(writers.os, primitive, raced)
    with pytest.raises(FileExistsError):
        write_image_data(path, colors())
    assert path.read_bytes() == b"other writer" and list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("kwargs", [{"quality": True}, {"quality": 101}, {"overwrite": 1},
    {"allow_precision_loss": 1}, {"create_dirs": 1}, {"dtype": "int32"}, {"input_encoding": "guess"}])
def test_invalid_export_options_fail_without_artifacts(tmp_path, kwargs):
    path = tmp_path / "absent" / "image.png"
    with pytest.raises((ValueError, TypeError)):
        write_image_data(path, colors(), **kwargs)
    assert not path.exists()


def test_typed_image_cannot_be_reinterpreted_by_export_options(tmp_path):
    image = ImageData.from_array(colors())
    for kwargs in ({"channel_order": "BGR"}, {"value_range": [0, 65535]}, {"input_encoding": "linear-srgb"}):
        with pytest.raises(ValueError):
            write_image_data(tmp_path / "out.png", image, **kwargs)


def test_strict_reader_and_legacy_reader_keep_their_explicit_contracts(tmp_path):
    path = tmp_path / "precision.png"
    write_image_data(path, colors())
    np.testing.assert_array_equal(read_image(path, strict=True), colors()/65535)
    np.testing.assert_array_equal(read_image(path, strict=True, color_space="bgr", dtype="uint16", range_="0-65535"), colors()[..., ::-1])
    np.testing.assert_array_equal(read_image(path), colors()/65535)
    with pytest.raises(ValueError, match="requires strict"):
        read_image(path, image_options={})


def test_legacy_writer_still_returns_bool_and_reduces_uint16_as_before(tmp_path):
    path = tmp_path / "legacy.png"
    assert write_image(path, colors()) is True
    np.testing.assert_array_equal(cv2.imread(str(path))[..., ::-1], (colors()//257).astype(np.uint8))


@pytest.mark.parametrize("suffix", [".png", ".tif"])
def test_resource_limits_reject_before_native_allocation(tmp_path, monkeypatch, suffix):
    path = tmp_path / ("limited" + suffix)
    write_image_data(path, colors())
    def forbidden(*args, **kwargs):
        raise AssertionError("native decoding must not run after a rejected header")
    monkeypatch.setattr(cv2, "imdecode", forbidden)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    with pytest.raises(ValueError, match="max_pixels"):
        read_image_data(path, max_pixels=3)
    with pytest.raises(ValueError, match="max_decoded_bytes"):
        read_image_data(path, max_decoded_bytes=4*3*8)  # source allocation also counts


@pytest.mark.parametrize("key,value", [("max_pixels", 0), ("max_pixels", True),
    ("max_decoded_bytes", -1), ("max_decoded_bytes", 1.5), ("max_file_bytes", True)])
def test_resource_limits_are_explicit_positive_integers(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        read_image_data(tmp_path / "absent.png", **{key: value})


def test_encoded_file_limit_precedes_read_and_bounds_growth_race(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    path.write_bytes(b"12345")
    original = Path.open
    def forbidden(*args, **kwargs):
        raise AssertionError("file larger than cap must not be opened")
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="before reading"):
        read_image_data(path, max_file_bytes=4)
    calls = []
    class Growing(io.BytesIO):
        def read(self, size=-1):
            calls.append(size)
            return super().read(size)
    def growing(value, mode="r", *args, **kwargs):
        return Growing(b"123456789") if value == path and mode == "rb" else original(value, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", growing)
    with pytest.raises(ValueError, match="grew beyond"):
        read_image_data(path, max_file_bytes=5)
    assert calls == [6]


@pytest.mark.parametrize("transparent", [False, True])
def test_paletted_decode_budget_counts_expanded_native_channels(tmp_path, monkeypatch, transparent):
    path = tmp_path / "palette.png"
    image = Image.fromarray(np.zeros((4, 5), dtype=np.uint8), mode="P")
    image.putpalette([0, 100, 200] * 256)
    image.save(path, transparency=0 if transparent else None)
    def forbidden(*args, **kwargs):
        raise AssertionError("expanded palette allocation must be rejected before decode")
    monkeypatch.setattr(cv2, "imdecode", forbidden)
    native_channels = 4 if transparent else 3
    with pytest.raises(ValueError, match="max_decoded_bytes"):
        read_image_data(path, max_decoded_bytes=4*5*(24 + native_channels)-1, alpha_policy="drop")


@pytest.mark.parametrize("text", [
    '{"ColorCorrectionPipeline":{"schema":1,"encoding":"srgb","encoding":"linear-srgb"}}',
    '{"ColorCorrectionPipeline":{"schema":true,"encoding":"srgb"}}',
    '{"ColorCorrectionPipeline":{"schema":1,"encoding":"srgb","unknown":0}}',
    'not JSON', '[]', '',
])
def test_owned_encoding_metadata_cannot_be_ambiguous(tmp_path, text):
    path = tmp_path / "ambiguous.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("ColorCorrectionPipeline", text)
    Image.fromarray(colors("uint8")).save(path, pnginfo=info)
    with pytest.raises(ValueError, match="encoding metadata"):
        read_image_data(path)


def test_unknown_codec_precision_uses_conservative_predecode_bound(tmp_path, monkeypatch):
    path = tmp_path / "sixteen.ppm"
    rgb = colors()
    path.write_bytes(b"P6\n2 2\n65535\n" + rgb.astype(">u2").tobytes())
    np.testing.assert_array_equal(read_image_data(path).pixels, rgb/65535)
    def forbidden(*args, **kwargs):
        raise AssertionError("unknown codec bound must precede native decode")
    monkeypatch.setattr(cv2, "imdecode", forbidden)
    with pytest.raises(ValueError, match="max_decoded_bytes"):
        read_image_data(path, max_decoded_bytes=4*(24 + 3*8)-1)


def test_directory_is_not_treated_as_an_encoded_file(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        read_image_data(tmp_path)


def test_large_foreign_tiff_description_is_not_mistaken_for_owned_metadata(tmp_path):
    path = tmp_path / "foreign-description.tif"
    tifffile.imwrite(path, colors(), photometric="rgb", metadata=None, description="ordinary-OME-metadata " + "x"*70000)
    np.testing.assert_array_equal(read_image_data(path).pixels, colors()/65535)
