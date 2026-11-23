"""Legacy ``write_image``/``read_image`` quantize like the strict export path."""

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline.io import read_image, write_image
from ColorCorrectionPipeline.io.writers import write_image_data


def _raw(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _gray_ramp_rgb(values):
    return np.repeat(np.asarray(values)[None, :, None], 3, axis=2)


@pytest.mark.parametrize("offset", [0.0, -0.3, 0.3, -0.49])
def test_float_values_round_to_nearest_code(tmp_path, offset):
    # Truncation used to map anything just below a code (e.g. 0.3/255 short of it)
    # to the code beneath.
    codes = np.arange(256)
    image = _gray_ramp_rgb(np.clip(codes + offset, 0, 255) / 255.0)
    assert write_image(tmp_path / "legacy.png", image)
    np.testing.assert_array_equal(_raw(tmp_path / "legacy.png")[0, :, 0], codes)


def test_float_rounding_matches_strict_writer(tmp_path):
    rng = np.random.default_rng(3)
    image = rng.random((17, 23, 3))
    assert write_image(tmp_path / "legacy.png", image)
    write_image_data(tmp_path / "strict.png", image, dtype="uint8", allow_precision_loss=True)
    np.testing.assert_array_equal(_raw(tmp_path / "legacy.png"), _raw(tmp_path / "strict.png"))


def test_float_out_of_range_is_clipped(tmp_path):
    image = _gray_ramp_rgb([-0.5, 1.5, 0.998, 0.002])
    assert write_image(tmp_path / "clip.png", image)
    np.testing.assert_array_equal(_raw(tmp_path / "clip.png")[0, :, 0], [0, 255, 254, 1])


def test_float16_is_rounded(tmp_path):
    image = _gray_ramp_rgb(np.array([0.5, 1.0], dtype=np.float16))
    assert write_image(tmp_path / "half.png", image)
    np.testing.assert_array_equal(_raw(tmp_path / "half.png")[0, :, 0], [128, 255])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(tmp_path, bad):
    image = np.full((2, 2, 3), 0.5)
    image[0, 0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        write_image(tmp_path / "nan.png", image)
    assert not (tmp_path / "nan.png").exists()


def test_integer_codes_clip_instead_of_wrapping(tmp_path):
    image = _gray_ramp_rgb(np.array([300, -5, 128, 255], dtype=np.int32))
    assert write_image(tmp_path / "int.png", image)
    np.testing.assert_array_equal(_raw(tmp_path / "int.png")[0, :, 0], [255, 0, 128, 255])


def test_uint16_is_rounded_to_nearest_code(tmp_path):
    values = np.array([0, 128, 129, 200, 385, 386, 65407, 65535], dtype=np.uint16)
    assert write_image(tmp_path / "u16.png", _gray_ramp_rgb(values))
    expected = np.rint(values / 257.0).astype(np.uint8)
    np.testing.assert_array_equal(_raw(tmp_path / "u16.png")[0, :, 0], expected)
    np.testing.assert_array_equal(expected, [0, 0, 1, 1, 1, 2, 255, 255])


def test_uint8_input_is_written_unchanged(tmp_path):
    codes = np.arange(256, dtype=np.uint8)
    assert write_image(tmp_path / "u8.png", _gray_ramp_rgb(codes))
    np.testing.assert_array_equal(_raw(tmp_path / "u8.png")[0, :, 0], codes)


@pytest.fixture
def uint16_png(tmp_path):
    values = np.array([0, 128, 129, 257, 32767, 65279, 65535], dtype=np.uint16)
    path = tmp_path / "source16.png"
    assert cv2.imwrite(str(path), _gray_ramp_rgb(values))
    return path, values.astype(np.float64)


def test_reader_scales_uint16_to_requested_0_255_range(uint16_png):
    path, values = uint16_png
    img = read_image(path, dtype="float64", range_="0-255")
    assert img.max() <= 255.0
    np.testing.assert_allclose(img[0, :, 0], values / 257.0, rtol=0, atol=1e-12)


def test_reader_keeps_existing_float_ranges(uint16_png, tmp_path):
    path, values = uint16_png
    np.testing.assert_allclose(read_image(path)[0, :, 0], values / 65535.0, atol=1e-15)
    np.testing.assert_array_equal(read_image(path, range_="0-65535")[0, :, 0], values)
    codes = np.arange(256, dtype=np.uint8)
    path8 = tmp_path / "source8.png"
    assert cv2.imwrite(str(path8), _gray_ramp_rgb(codes))
    np.testing.assert_array_equal(read_image(path8, range_="0-255")[0, :, 0], codes)
    np.testing.assert_allclose(read_image(path8)[0, :, 0], codes / 255.0, atol=1e-15)


def test_reader_uint16_to_uint8_rounds_like_strict_reader(uint16_png):
    path, values = uint16_png
    legacy = read_image(path, dtype="uint8", range_="0-255")
    np.testing.assert_array_equal(legacy[0, :, 0], np.rint(values / 257.0))
    strict = read_image(path, dtype="uint8", range_="0-255", strict=True)
    np.testing.assert_array_equal(legacy, strict)


def test_reader_float_to_integer_rounds(tmp_path):
    # Float TIFF sources go through the same nearest-code rounding.
    pytest.importorskip("tifffile")
    import tifffile

    values = np.array([0.0, 0.5, 0.998, 1.0], dtype=np.float32)
    path = tmp_path / "f32.tif"
    tifffile.imwrite(path, _gray_ramp_rgb(values), photometric="rgb")
    out8 = read_image(path, dtype="uint8")
    np.testing.assert_array_equal(out8[0, :, 0], [0, 128, 254, 255])
    out16 = read_image(path, dtype="uint16")
    np.testing.assert_array_equal(
        out16[0, :, 0], np.rint(values.astype(np.float64) * 65535).astype(np.uint16)
    )
