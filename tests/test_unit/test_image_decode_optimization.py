"""Bounded snapshots and metadata-only PNG inspection preserve interpretation."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, PngImagePlugin

from ColorCorrectionPipeline import ColorCorrection, ImageData
from ColorCorrectionPipeline.colorimetry import ColorSemantics
from ColorCorrectionPipeline.io import read_image_data
from ColorCorrectionPipeline.io.image_data import _read_snapshot, _READ_CHUNK_BYTES
from ColorCorrectionPipeline.io.writers import _png_chunk


@pytest.fixture(scope="module")
def baseline():
    path = Path(__file__).parents[1]/"fixtures/image_data_p9_reference.py"
    spec = importlib.util.spec_from_file_location("ColorCorrectionPipeline.io._reader_p9", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixels(dtype):
    result = np.array([[[3, 37, 255], [12, 240, 19]], [[64, 124, 9], [25, 1, 92]]], np.uint8)
    return result if dtype == "uint8" else result.astype(np.uint16)*257


def late_chunks(data, kinds):
    start, kept, moved = 8, [], []
    while start < len(data):
        length = struct.unpack(">I", data[start:start+4])[0]
        chunk = data[start:start+length+12]
        kind = data[start+4:start+8]
        if kind in kinds:
            moved.append(chunk)
        elif kind == b"IEND":
            return data[:8]+b"".join(kept+moved+[chunk])
        else:
            kept.append(chunk)
        start += length+12
    raise AssertionError("missing IEND")


@pytest.mark.parametrize("count", [0, 5, _READ_CHUNK_BYTES-1, _READ_CHUNK_BYTES+17])
def test_snapshot_never_requests_file_cap_and_preserves_all_bytes(count):
    data = b"x"*count
    class Stream(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= _READ_CHUNK_BYTES
            return super().read(size)
    assert _read_snapshot(Stream(data), max(count, 1)) == data


def test_snapshot_detects_growth_and_short_reads():
    class Short(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 3))
    assert _read_snapshot(Short(b"abcdefg"), 7) == b"abcdefg"
    with pytest.raises(ValueError, match="grew"):
        _read_snapshot(Short(b"abcdefgh"), 7)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("orientation", [1, 6])
def test_static_png_decodes_only_once_with_full_p9_identity(tmp_path, monkeypatch, baseline, dtype, orientation):
    path = tmp_path/"pixels.png"
    assert cv2.imwrite(str(path), pixels(dtype)[..., ::-1])
    if orientation != 1:
        exif = Image.Exif(); exif[274] = orientation
        encoded = path.read_bytes()
        ihdr_end = 8+12+int.from_bytes(encoded[8:12], "big")
        path.write_bytes(encoded[:ihdr_end]+_png_chunk(b"eXIf", exif.tobytes()[6:])+encoded[ihdr_end:])
    expected = baseline.read_image_data(path)
    def no_pillow_pixels(*args, **kwargs):
        pytest.fail("static PNG must not allocate a Pillow pixel image")
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", no_pillow_pixels)
    decode, calls = cv2.imdecode, []
    def once(*args, **kwargs):
        calls.append(1)
        return decode(*args, **kwargs)
    monkeypatch.setattr(cv2, "imdecode", once)
    actual = read_image_data(path)
    assert len(calls) == 1
    assert actual.to_metadata() == expected.to_metadata()
    assert actual.pixels.tobytes() == expected.pixels.tobytes()


@pytest.mark.parametrize("limit", [{"max_pixels": 1}, {"max_decoded_bytes": 1}])
def test_png_allocation_limits_precede_all_pixel_decoders(tmp_path, monkeypatch, limit):
    path = tmp_path/"capped.png"
    Image.fromarray(pixels("uint8")).save(path)
    def forbidden(*args, **kwargs):
        pytest.fail("pixel allocation must not run before the cap")
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", forbidden)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "getexif", forbidden)
    monkeypatch.setattr(cv2, "imdecode", forbidden)
    with pytest.raises(ValueError, match="exceed"):
        read_image_data(path, **limit)


@pytest.mark.parametrize("kind", ["plain", "compressed", "international"])
def test_late_encoding_metadata_without_pixel_inflation(tmp_path, monkeypatch, baseline, kind):
    path = tmp_path/"late.png"
    info = PngImagePlugin.PngInfo()
    text = json.dumps({"ColorCorrectionPipeline": {"schema": 1, "encoding": "linear-srgb"}})
    if kind == "international":
        info.add_itxt("ColorCorrectionPipeline", text, zip=True)
    else:
        info.add_text("ColorCorrectionPipeline", text, zip=kind == "compressed")
    stream = io.BytesIO(); Image.fromarray(pixels("uint8")).save(stream, format="PNG", pnginfo=info)
    path.write_bytes(late_chunks(stream.getvalue(), {b"tEXt", b"iTXt", b"zTXt"}))
    expected = baseline.read_image_data(path)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *args: pytest.fail("no PNG pixel inflate"))
    actual = read_image_data(path)
    assert actual.color_space == "linear-srgb"
    assert actual.to_metadata() == expected.to_metadata()
    np.testing.assert_array_equal(actual.pixels, expected.pixels)
    with pytest.raises(ValueError, match="conflict"):
        read_image_data(path, color_space="srgb")


def test_late_icc_is_rejected_before_pixels_even_with_header_exif(tmp_path, monkeypatch):
    path = tmp_path/"late-profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif(); exif[274] = 1
    stream = io.BytesIO(); Image.fromarray(pixels("uint8")).save(stream, format="PNG", exif=exif, icc_profile=profile)
    path.write_bytes(late_chunks(stream.getvalue(), {b"iCCP"}))
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *args: pytest.fail("no PNG inflate"))
    monkeypatch.setattr(cv2, "imdecode", lambda *args: pytest.fail("rejected ICC must not decode"))
    with pytest.raises(ValueError, match="embedded ICC"):
        read_image_data(path)


def test_late_exif_orientation_is_applied_once_without_pixel_inflation(tmp_path, monkeypatch, baseline):
    path = tmp_path/"late-orientation.png"
    exif = Image.Exif(); exif[274] = 6
    stream = io.BytesIO(); Image.fromarray(pixels("uint8")).save(stream, format="PNG", exif=exif)
    path.write_bytes(late_chunks(stream.getvalue(), {b"eXIf"}))
    expected = baseline.read_image_data(path)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *args: pytest.fail("no PNG inflate"))
    result = read_image_data(path)
    assert result.to_metadata() == expected.to_metadata()
    np.testing.assert_array_equal(result.pixels, np.rot90(pixels("uint8"), 3)/255)


def test_prepared_final_borrow_does_not_change_intermediates_or_public_adapter():
    cc = ColorCorrection(); cc.models.model_wb = np.eye(3)*.8
    image = ImageData.from_array(pixels("uint8"))
    data, working = cc._prepare_input(image, ColorSemantics(), _borrow_readonly=True)
    assert np.shares_memory(working, image.pixels) and working is not image.pixels
    assert not working.flags.writeable
    public = image.to_srgb()
    assert public.flags.writeable and not np.shares_memory(public, image.pixels)
    with cc.prepare_execution({"mode": "direct", "backend": "numpy", "validation_samples": 64}) as prepared:
        result = prepared.predict_image(image)
        assert result.images["CC"].flags.writeable
        result.images["CC"].fill(0)
        assert np.any(image.pixels)
        stages = prepared.predict_image(image, stages="all")
        assert stages.images["FFC"].flags.writeable
        assert not np.shares_memory(stages.images["FFC"], image.pixels)
    source32 = pixels("uint8").astype(np.float32)/255
    _, working32 = cc._prepare_input(source32, ColorSemantics(), _borrow_readonly=True)
    assert working32.dtype == np.float32
    np.testing.assert_array_equal(working32, source32)
