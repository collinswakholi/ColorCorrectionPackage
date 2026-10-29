"""Installed-wheel image/export parity against frozen P9 source, Python 3.8+."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import struct
import sys


@contextmanager
def replaced(owner, name, value):
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield original
    finally:
        setattr(owner, name, original)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raises(kind, text, operation):
    try:
        operation()
    except kind as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("expected %s: %s" % (kind.__name__, text))


def late_chunks(data, kinds):
    offset, kept, moved = 8, [], []
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset+4])[0]
        chunk = data[offset:offset+length+12]
        kind = data[offset+4:offset+8]
        if kind in kinds:
            moved.append(chunk)
        elif kind == b"IEND":
            return data[:8] + b"".join(kept + moved + [chunk])
        else:
            kept.append(chunk)
        offset += length+12
    raise AssertionError("PNG lacks IEND")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / (args.output.stem + "-artifacts")
    work.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    from PIL import Image, ImageCms, PngImagePlugin
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ImageData
    from ColorCorrectionPipeline.io import read_image_data, writers, image_data

    assert sys.prefix != sys.base_prefix
    assert Path(sys.prefix).resolve() in Path(package.__file__).resolve().parents
    old_contract = load("ColorCorrectionPipeline._audit_p9_contract", args.fixtures / "image_contract_p9.py")
    old_reader = load("ColorCorrectionPipeline.io._audit_p9_reader", args.fixtures / "image_data_p9_reference.py")
    old_writer = load("ColorCorrectionPipeline.io._audit_p9_writer", args.fixtures / "writers_p9_reference.py")
    fixture_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(args.fixtures.glob("*.py"))}

    rng = np.random.default_rng(612)
    cases = [(rng.integers(0, 256, (4, 6, 3), dtype=np.uint8), None),
             (rng.integers(0, 65536, (4, 6, 3), dtype=np.uint16), None),
             (rng.integers(0, 65536, (4, 6, 3), dtype=np.uint16).astype(">u2"), None)]
    cases += [(rng.random((4, 6, 3)).astype(dtype), None) for dtype in ("float16", "float32", "float64", ">f8")]
    cases += [(np.asfortranarray(rng.random((4, 6, 3))), None),
              (rng.random((8, 12, 3))[::2, ::2, ::-1], None),
              (rng.uniform(-2, 4, (4, 6, 3)), (-2, 4)),
              (np.array([[[-0., +0., 1.]]]), None),
              (np.array([[[-0., +0., 1.]]]), (-0., 1.)),
              (np.array([[[-0., +0., 1.]]], dtype=np.float32), (-0., 1.))]
    identities = 0
    for orientation in range(1, 9):
        for order in ("RGB", "BGR"):
            for pixels, value_range in cases:
                options = dict(channel_order=order, orientation=orientation, value_range=value_range,
                               provenance={"case": [1, {"flags": [True, None]}]})
                old, new = old_contract.ImageData.from_array(pixels, **options), ImageData.from_array(pixels, **options)
                assert new.pixels.tobytes() == old.pixels.tobytes()
                assert new.to_metadata() == old.to_metadata()
                assert new.digest == old.digest and new.pixel_digest == old.pixel_digest
                raises(ValueError, "WRITEABLE", lambda: new.pixels.setflags(write=True))
                identities += 1
    large_identities = 0
    for dtype in ("uint8", "uint16", ">u2", "float16", "float32", "float64", ">f8"):
        target = np.dtype(dtype)
        if target.kind == "u":
            values = rng.integers(0, np.iinfo(target).max+1, (513, 683, 3), dtype=target.newbyteorder("=")).astype(target)
            value_range = None
        else:
            values = rng.uniform(-3, 10, (513, 683, 3)).astype(target)
            value_range = (-3, 10)
        values = values[::-1, ::-1]
        options = dict(channel_order="BGR", orientation=7, value_range=value_range)
        old, new = old_contract.ImageData.from_array(values, **options), ImageData.from_array(values, **options)
        assert new.pixels.tobytes() == old.pixels.tobytes()
        assert new.to_metadata() == old.to_metadata()
        large_identities += 1
    owner = np.full((3, 4, 3), .5)
    view = owner.view(); view.setflags(write=False)
    captured = ImageData.from_array(view)
    owner.fill(1)
    np.testing.assert_array_equal(captured.pixels, np.full((3, 4, 3), .5))

    def samples(dtype="float64", shape=(11, 17, 3)):
        raw = np.random.default_rng(619).integers(0, 65536, shape, dtype=np.uint16)
        if dtype == "uint8":
            return (raw // 257).astype(dtype)
        if dtype == "uint16":
            return raw
        return (raw / 65535).astype(dtype)

    exports = []
    specifications = [(".png", "uint8", "preserve"), (".png", "uint16", "preserve"),
        (".png", "float64", "uint8"), (".png", "float64", "uint16"),
        (".tiff", "float16", "preserve"), (".tiff", "float32", "preserve"),
        (".tiff", "float64", "preserve"), (".tiff", "uint16", "preserve"),
        (".jpg", "float64", "uint8"), (".webp", "float64", "uint8")]
    for oriented in (False, True):
        for suffix, source_dtype, target_dtype in specifications:
            source = ImageData.from_array(samples(source_dtype), color_space="srgb", channel_order="BGR",
                orientation=6 if oriented else 1, apply_orientation=not oriented,
                provenance={"file_sha256": "1"*64, "nested": {"origin": "writer-parity"}})
            before = source.pixels.tobytes()
            index = len(exports)
            old_path, new_path = work / ("old-%d%s" % (index, suffix)), work / ("new-%d%s" % (index, suffix))
            options = dict(dtype=target_dtype, allow_precision_loss=True, overwrite=True)
            old = old_writer.write_image_data(old_path, source, **options).to_dict()
            new = writers.write_image_data(new_path, source, **options).to_dict()
            assert old_path.read_bytes() == new_path.read_bytes()
            old.pop("path"); new.pop("path")
            assert new == old, (suffix, source_dtype, oriented, new, old)
            assert source.pixels.tobytes() == before and not source.pixels.flags.writeable
            exports.append({"format": suffix, "source_dtype": source_dtype, "target_dtype": target_dtype,
                "pending_orientation": oriented, "byte_and_record_exact": True, "sha256": new["sha256"]})
    transfer_exports = 0
    for source_encoding, output_encoding in (("srgb", "linear-srgb"), ("linear-srgb", "srgb")):
        source = ImageData.from_array(samples(), color_space=source_encoding)
        options = dict(output_encoding=output_encoding, dtype="uint16", allow_precision_loss=True, overwrite=True)
        old_path, new_path = work / "transfer-old.png", work / "transfer-new.png"
        old = old_writer.write_image_data(old_path, source, **options).to_dict()
        new = writers.write_image_data(new_path, source, **options).to_dict()
        old.pop("path"); new.pop("path")
        assert old == new and old_path.read_bytes() == new_path.read_bytes()
        transfer_exports += 1

    pixels = np.array([[[3, 37, 255], [12, 240, 19]], [[64, 124, 9], [25, 1, 92]]], np.uint8)
    pngs = 0
    def forbid(*unused, **kwargs):
        raise AssertionError("Pillow pixels must not inflate on static PNG metadata path")
    for dtype in ("uint8", "uint16"):
        for orientation in (1, 6):
            path = work / ("static-%s-%d.png" % (dtype, orientation))
            array = pixels if dtype == "uint8" else pixels.astype(np.uint16)*257
            assert cv2.imwrite(str(path), array[..., ::-1])
            if orientation != 1:
                exif = Image.Exif(); exif[274] = orientation
                encoded = path.read_bytes()
                ihdr = 20 + int.from_bytes(encoded[8:12], "big")
                path.write_bytes(encoded[:ihdr]+writers._png_chunk(b"eXIf", exif.tobytes()[6:])+encoded[ihdr:])
            old = old_reader.read_image_data(path)
            calls = []
            decode = cv2.imdecode
            def once(*arguments, **kwargs):
                calls.append(1)
                return decode(*arguments, **kwargs)
            with replaced(PngImagePlugin.PngImageFile, "load", forbid), replaced(cv2, "imdecode", once):
                new = read_image_data(path)
            assert len(calls) == 1 and new.to_metadata() == old.to_metadata()
            assert new.pixels.tobytes() == old.pixels.tobytes()
            pngs += 1
    late_metadata = 0
    for kind in ("plain", "compressed", "international"):
        info = PngImagePlugin.PngInfo()
        text = json.dumps({"ColorCorrectionPipeline": {"schema": 1, "encoding": "linear-srgb"}})
        if kind == "international":
            info.add_itxt("ColorCorrectionPipeline", text, zip=True)
        else:
            info.add_text("ColorCorrectionPipeline", text, zip=kind == "compressed")
        stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="PNG", pnginfo=info)
        path = work / ("late-%s.png" % kind)
        path.write_bytes(late_chunks(stream.getvalue(), {b"tEXt", b"iTXt", b"zTXt"}))
        old = old_reader.read_image_data(path)
        with replaced(PngImagePlugin.PngImageFile, "load", forbid):
            new = read_image_data(path)
        assert new.color_space == "linear-srgb" and new.to_metadata() == old.to_metadata()
        assert new.pixels.tobytes() == old.pixels.tobytes()
        raises(ValueError, "conflict", lambda: read_image_data(path, color_space="srgb"))
        late_metadata += 1
    cap_path = work / "capped.png"
    Image.fromarray(pixels).save(cap_path)
    with replaced(PngImagePlugin.PngImageFile, "load", forbid), replaced(cv2, "imdecode", forbid):
        for options in ({"max_pixels": 1}, {"max_decoded_bytes": 1}):
            raises(ValueError, "exceed", lambda: read_image_data(cap_path, **options))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif(); exif[274] = 1
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="PNG", exif=exif, icc_profile=profile)
    profile_path = work / "late-profile.png"
    profile_path.write_bytes(late_chunks(stream.getvalue(), {b"iCCP"}))
    with replaced(PngImagePlugin.PngImageFile, "load", forbid), replaced(cv2, "imdecode", forbid):
        raises(ValueError, "embedded ICC", lambda: read_image_data(profile_path))
    transactional = 0
    encode = writers._encode_strict
    def wrong_encoding(path, storage, encoding, quality):
        return encode(path, storage, "linear-srgb" if encoding == "srgb" else "srgb", quality)
    with replaced(writers, "_encode_strict", wrong_encoding):
        for suffix in (".png", ".tiff"):
            path = work / ("prior" + suffix); path.write_bytes(b"prior")
            raises(OSError, "transfer encoding", lambda: writers.write_image_data(path, pixels, overwrite=True))
            assert path.read_bytes() == b"prior" and not list(work.glob(".*.tmp"))
            transactional += 1
    class Short(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= image_data._READ_CHUNK_BYTES
            return super().read(min(size, 3))
    assert image_data._read_snapshot(Short(b"abcdefg"), 7) == b"abcdefg"
    raises(ValueError, "grew", lambda: image_data._read_snapshot(Short(b"abcdefgh"), 7))

    versions = {}
    for name in ("ColorCorrectionPipeline", "numpy", "Pillow", "opencv-python", "tifffile", "numba"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "package_file": package.__file__, "versions": versions,
        "frozen_fixture_sha256": fixture_hashes, "small_contract_identity_cases": identities,
        "large_chunked_identity_cases": large_identities, "mutable_readonly_owner_isolated": True,
        "export_byte_and_record_parity": exports, "transfer_export_parity_cases": transfer_exports,
        "static_png_single_decode_cases": pngs, "late_encoding_parity_cases": late_metadata,
        "predecode_allocation_caps": True, "late_icc_predecode_rejection": True,
        "transactional_wrong_encoding_rejections": transactional, "short_reads_and_growth_checks": True}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
