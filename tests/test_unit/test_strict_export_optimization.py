"""Exact legacy equations/bytes and strict publication survive reduced workspace."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import tifffile

from ColorCorrectionPipeline import ImageData
from ColorCorrectionPipeline.io import writers
from ColorCorrectionPipeline.io import image_data


@pytest.fixture(scope="module")
def frozen_writer():
    # Same dependencies/contracts isolate the writer change. The complete frozen
    # P9 wheel also remains available for the parent's process-level comparison.
    site = os.environ.get("CCP_P9_SITE")
    if site:
        path = Path(site) / "ColorCorrectionPipeline/io/writers.py"
    else:
        candidates = [Path(__file__).parent/"fixtures/writers_p9_reference.py",
                      Path(__file__).parents[1]/"fixtures/writers_p9_reference.py"]
        path = next(path for path in candidates if path.is_file())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "6fc0cbb328baa607f78dd901b781c5a46eace9ddd6318c9d926da900d8d88f3a"
    name = "ColorCorrectionPipeline.io._p9_writer_reference"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def values(dtype="float64", shape=(11, 17, 3)):
    raw = np.random.default_rng(619).integers(0, 65536, shape, dtype=np.uint16)
    if dtype == "uint8":
        return (raw // 257).astype(dtype)
    if dtype == "uint16":
        return raw
    return (raw / 65535).astype(dtype)


@pytest.mark.parametrize("dtype", [np.dtype("uint8"), np.dtype("uint16")])
def test_integer_quantization_reuses_workspace_with_exact_historical_error(dtype):
    maximum = np.iinfo(dtype).max
    levels = np.arange(maximum + 1, dtype=np.float64)
    ties = (levels[:-1] + .5) / maximum
    edges = np.concatenate((levels / maximum, ties,
        np.nextafter(ties, 0.), np.nextafter(ties, 1.),
        np.array([-0., 0., np.nextafter(0., 1.), np.nextafter(1., 0.), 1.])))
    count = max(edges.size, writers._EXPORT_CHUNK_VALUES + 3)
    count += -count % 3
    source = np.resize(edges, count).reshape(1, -1, 3)
    source.setflags(write=False)
    for layout in (source, source[:, ::-1, ::-1]):
        before = layout.tobytes()
        # Frozen P9's full-array equations provide an independent oracle for
        # the new bounded workspace, including ties and chunk boundaries.
        expected = np.rint(layout * maximum).astype(dtype)
        error = float(np.max(np.abs(expected.astype(np.float64) / maximum - layout)))
        actual, actual_error = writers._quantize_strict(layout, dtype, maximum)
        assert actual.tobytes() == expected.tobytes()
        assert actual_error.hex() == error.hex()
        assert layout.tobytes() == before and not layout.flags.writeable


@pytest.mark.parametrize("suffix,source_dtype,target_dtype", [
    (".png", "uint8", "preserve"), (".png", "uint16", "preserve"),
    (".png", "float64", "uint8"), (".png", "float64", "uint16"),
    (".tiff", "float16", "preserve"), (".tiff", "float32", "preserve"),
    (".tiff", "float64", "preserve"), (".tiff", "uint16", "preserve"),
    (".jpg", "float64", "uint8"), (".webp", "float64", "uint8"),
])
@pytest.mark.parametrize("oriented", [False, True])
def test_full_export_record_and_encoded_bytes_match_p9(tmp_path, frozen_writer, suffix, source_dtype, target_dtype, oriented):
    source = ImageData.from_array(values(source_dtype), color_space="srgb",
        channel_order="BGR", orientation=6 if oriented else 1, apply_orientation=not oriented,
        provenance={"file_sha256": "1"*64, "nested": {"origin": "writer-parity"}})
    before = source.pixels.tobytes()
    opts = dict(dtype=target_dtype, allow_precision_loss=True)
    old_path, new_path = tmp_path / ("old"+suffix), tmp_path / ("new"+suffix)
    expected = frozen_writer.write_image_data(old_path, source, **opts).to_dict()
    actual = writers.write_image_data(new_path, source, **opts).to_dict()
    assert old_path.read_bytes() == new_path.read_bytes()
    expected.pop("path"); actual.pop("path")
    assert actual == expected
    assert source.pixels.tobytes() == before and not source.pixels.flags.writeable
    with pytest.raises(ValueError):
        source.pixels.setflags(write=True)


@pytest.mark.parametrize("source_encoding,target_encoding", [("srgb", "linear-srgb"), ("linear-srgb", "srgb")])
def test_transfer_conversion_export_record_and_bytes_match_p9(tmp_path, frozen_writer, source_encoding, target_encoding):
    source = ImageData.from_array(values(), color_space=source_encoding)
    opts = dict(output_encoding=target_encoding, dtype="uint16", allow_precision_loss=True)
    old = frozen_writer.write_image_data(tmp_path/"old.png", source, **opts).to_dict()
    new = writers.write_image_data(tmp_path/"new.png", source, **opts).to_dict()
    old.pop("path"); new.pop("path")
    assert old == new and (tmp_path/"old.png").read_bytes() == (tmp_path/"new.png").read_bytes()


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
def test_bounded_quantization_matches_whole_array_equation(dtype):
    samples = values(shape=(1, 65540, 3))
    # Include ties, neighbors, endpoints, signed zero and subnormal float64.
    boundary = np.array([0., -0., 1., .5, .5/255, np.nextafter(.5/255, 0),
                         np.nextafter(.5/255, 1), np.nextafter(0., 1), 1/65535])
    samples.reshape(-1)[:len(boundary)] = boundary
    target = np.dtype(dtype)
    maximum = np.iinfo(target).max if target.kind == "u" else 1.
    expected = np.rint(samples*maximum).astype(target) if target.kind == "u" else samples.astype(target)
    error = float(np.max(np.abs(expected.astype(np.float64)/maximum-samples)))
    actual, actual_error = writers._quantize_strict(samples, target, maximum)
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    assert actual_error == error
    if dtype == "float64":
        assert actual is samples


@pytest.mark.parametrize("shape", [(1, 65540, 3), (65540, 1, 3), (13, 17, 3)])
def test_chunked_lossy_error_keeps_exact_equation_and_bounds(shape):
    storage = values("uint8", shape)
    decoded = storage[..., ::-1].copy()[..., ::-1]  # negative-stride logical RGB
    decoded[0, 0, 0] ^= 31
    expected = float(np.max(np.abs(decoded.astype(np.float64)/255 - storage.astype(np.float64)/255)))
    assert writers._codec_error_strict(decoded, storage, 255) == expected
    covered = 0
    for region in writers._export_regions(shape):
        size = storage[region].size
        assert 0 < size <= writers._EXPORT_CHUNK_VALUES
        covered += size
    assert covered == storage.size


def test_lossless_still_decodes_and_compares_all_pixels_without_normalizing_twice(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("lossless equality already establishes zero codec error")
    monkeypatch.setattr(writers, "_codec_error_strict", forbidden)
    record = writers.write_image_data(tmp_path/"strict.png", values(), dtype="uint8", allow_precision_loss=True)
    assert record.maximum_codec_error == 0
    original = writers._encode_strict
    def wrong_pixel(path, storage, encoding, quality):
        wrong = storage.copy()
        wrong[-1, -1, -1] ^= 1
        return original(path, wrong, encoding, quality)
    monkeypatch.setattr(writers, "_encode_strict", wrong_pixel)
    with pytest.raises(OSError, match="samples exactly"):
        writers.write_image_data(tmp_path/"wrong.png", values(), dtype="uint8", allow_precision_loss=True)
    assert not (tmp_path/"wrong.png").exists()


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_valid_but_wrong_encoding_metadata_cannot_publish(tmp_path, monkeypatch, suffix):
    original = writers._encode_strict
    def wrong_encoding(path, storage, encoding, quality):
        return original(path, storage, "linear-srgb" if encoding == "srgb" else "srgb", quality)
    monkeypatch.setattr(writers, "_encode_strict", wrong_encoding)
    path = tmp_path/("prior"+suffix)
    path.write_bytes(b"prior")
    with pytest.raises(OSError, match="transfer encoding"):
        writers.write_image_data(path, values("uint8"), overwrite=True)
    assert path.read_bytes() == b"prior" and not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure", ["orientation", "frame", "photometric"])
def test_noncanonical_tiff_metadata_cannot_publish(tmp_path, monkeypatch, failure):
    def malformed(path, storage, encoding, quality):
        stream = io.BytesIO()
        description = json.dumps({"ColorCorrectionPipeline": {"schema": 1, "encoding": encoding}})
        with tifffile.TiffWriter(stream) as file:
            file.write(storage, photometric="minisblack" if failure == "photometric" else "rgb",
                planarconfig="contig", metadata=None, description=description,
                extratags=[(274, "H", 1, 6 if failure == "orientation" else 1, False)])
            if failure == "frame":
                file.write(storage, photometric="rgb", metadata=None, description=description)
        return stream.getvalue(), "TIFF", False
    monkeypatch.setattr(writers, "_encode_strict", malformed)
    path = tmp_path/"prior.tiff"
    path.write_bytes(b"prior")
    with pytest.raises(OSError, match="interpretation"):
        writers.write_image_data(path, values("uint8"), overwrite=True)
    assert path.read_bytes() == b"prior" and not list(tmp_path.glob(".*.tmp"))


def test_wrong_larger_encoder_shape_is_capped_before_opencv_decode(tmp_path, monkeypatch):
    original = writers._encode_strict
    def oversized(path, storage, encoding, quality):
        return original(path, np.repeat(storage, 2, axis=1), encoding, quality)
    monkeypatch.setattr(writers, "_encode_strict", oversized)
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: pytest.fail("wrong dimensions must fail before native decode"))
    with pytest.raises(OSError, match="allocation"):
        writers.write_image_data(tmp_path/"too-big.png", values("uint8"))
    assert not (tmp_path/"too-big.png").exists()


def test_byte_readback_and_decode_happen_after_fsync_before_publish(tmp_path, monkeypatch):
    events = []
    fsync, read, decode, replace = writers.os.fsync, Path.read_bytes, image_data._decode, writers.os.replace
    def sync(*args):
        result = fsync(*args); events.append("fsync"); return result
    def readback(path, *args):
        result = read(path, *args)
        if path.suffix == ".tmp":
            events.append("readback")
        return result
    def verify(*args, **kwargs):
        assert events == ["fsync", "readback"]
        result = decode(*args, **kwargs); events.append("decode"); return result
    def publish(*args):
        assert events == ["fsync", "readback", "decode"]
        result = replace(*args); events.append("publish"); return result
    monkeypatch.setattr(writers.os, "fsync", sync)
    monkeypatch.setattr(Path, "read_bytes", readback)
    monkeypatch.setattr(image_data, "_decode", verify)
    monkeypatch.setattr(writers.os, "replace", publish)
    path = tmp_path/"strict.png"
    record = writers.write_image_data(path, values("uint8"), overwrite=True)
    assert events == ["fsync", "readback", "decode", "publish"]
    assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_borrowed_float64_storage_has_an_independent_ndarray_header(tmp_path, monkeypatch):
    source = ImageData.from_array(values())
    header = source.pixels
    shape, dtype = header.shape, header.dtype
    original = writers._encode_strict
    def mutate_public_header(path, storage, encoding, quality):
        assert storage is not header and np.shares_memory(storage, header)
        header.shape = (shape[0]*shape[1], 3)
        assert storage.shape == shape and storage.dtype == dtype
        return original(path, storage, encoding, quality)
    monkeypatch.setattr(writers, "_encode_strict", mutate_public_header)
    try:
        record = writers.write_image_data(tmp_path/"header.tiff", source)
        assert record.shape == shape and record.source_digest == source.digest
    finally:
        header.shape = shape
