"""Exact arithmetic, immutable capture and strict fast-PNG publication."""
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, ImageData
from ColorCorrectionPipeline import _image_numeric as numeric
from ColorCorrectionPipeline import contracts
from ColorCorrectionPipeline.io import read_image_data, write_image_data, writers


def require_numba():
    if not numeric.accel.HAS_NUMBA:
        pytest.skip('Numba is optional')


def large(dtype='uint16'):
    dtype = np.dtype(dtype)
    result = np.resize(np.arange(65536, dtype=np.uint16), 513*683*3).reshape(513, 683, 3)
    return (result % 256).astype(dtype) if dtype.itemsize == 1 else result.astype(dtype)


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', '>u2'])
@pytest.mark.parametrize('orientation', [1, 7])
def test_large_integer_capture_preserves_every_code_and_exact_identity(dtype, orientation):
    source = large(dtype)[::-1, ::-1]
    canonical = source[..., ::-1]
    if orientation == 7:
        canonical = canonical.swapaxes(0, 1)[::-1, ::-1]
    expected = canonical.astype(np.float64)
    expected -= 0.
    expected /= np.iinfo(source.dtype).max
    actual = ImageData.from_array(source, channel_order='BGR', orientation=orientation)
    assert actual.pixels.tobytes() == expected.tobytes()
    assert not actual.pixels.flags.writeable
    metadata = dict(actual._metadata_fields())
    assert actual.pixel_digest == contracts._hash_parts({'color_space':'srgb','channel_order':'RGB'}, (expected,))
    assert actual.digest == contracts._hash_parts(metadata, (expected,))
    source[:] = 0
    assert actual.pixels.tobytes() == expected.tobytes()


@pytest.mark.parametrize('operation', ['normalization', 'quantization'])
@pytest.mark.parametrize('mode', ['partial-raise', 'wrong-qualification'])
def test_failed_backend_never_poisons_numpy_recovery(monkeypatch, caplog, operation, mode):
    require_numba()
    source = large()
    values = ImageData.from_array(source).pixels
    size = source.size
    name = '_normalize_unsigned' if operation == 'normalization' else '_quantize_unsigned'
    original = getattr(numeric, name)
    def bad(*args):
        if mode == 'wrong-qualification':
            args[1][:] = 0
            return 0.
        if args[0].size == size:
            args[1].reshape(-1)[:size//2] = 0
            raise RuntimeError('partially written image backend')
        return original(*args)
    monkeypatch.setattr(numeric, name, bad)
    with caplog.at_level(logging.DEBUG, logger=numeric.__name__):
        if operation == 'normalization':
            actual = ImageData.from_array(source)
            assert actual.pixels.tobytes() == values.tobytes()
        else:
            actual, error = writers._quantize_strict(values, np.dtype('uint16'), 65535)
            expected = np.rint(values*65535).astype(np.uint16)
            assert actual.tobytes() == expected.tobytes()
            assert error == float(np.max(np.abs(expected.astype(np.float64)/65535-values)))
    assert 'NumPy' in caplog.text


def test_unavailable_backend_keeps_exact_capture_and_quantization(monkeypatch):
    source = large()
    expected = source.astype(np.float64)/65535
    monkeypatch.setattr(numeric.accel, 'HAS_NUMBA', False)
    actual = ImageData.from_array(source)
    assert actual.pixels.tobytes() == expected.tobytes()
    output, error = writers._quantize_strict(actual.pixels, np.dtype('uint16'), 65535)
    assert output.tobytes() == source.tobytes() and error == 0.


@pytest.mark.parametrize('maximum', [255, 65535])
def test_fused_rounding_and_reported_maximum_error_match_all_boundaries(maximum):
    require_numba()
    levels = np.arange(maximum+1, dtype=np.float64)
    ties = (levels[:-1]+.5)/maximum
    source = np.concatenate((levels/maximum, ties, np.nextafter(ties, 0), np.nextafter(ties, 1),
        np.random.default_rng(317).random(100_000), [0., -0., 1., np.nextafter(0., 1.)]))
    source = np.resize(source, max(source.size, numeric.MIN_VALUES+1))
    source = np.resize(source, source.size+(-source.size % 3)).reshape(1, -1, 3)
    source.setflags(write=False)
    dtype = np.dtype('uint8' if maximum==255 else 'uint16')
    expected = np.rint(source*maximum).astype(dtype)
    actual, error = writers._quantize_strict(source, dtype, maximum)
    assert actual.tobytes() == expected.tobytes()
    assert error == float(np.max(np.abs(expected.astype(np.float64)/maximum-source)))


def test_buffer_error_cannot_mask_capture_failure(monkeypatch):
    monkeypatch.setattr(numeric.accel, 'HAS_NUMBA', False)
    def failed(destination, *args, **kwargs):
        destination.reshape(-1)[:10] = .5
        raise RuntimeError('capture failed')
    monkeypatch.setattr(np, 'copyto', failed)
    with pytest.raises(RuntimeError, match='capture failed'):
        ImageData.from_array(large())


def test_source_header_changes_do_not_change_captured_interpretation(monkeypatch):
    monkeypatch.setattr(numeric.accel, 'HAS_NUMBA', False)
    source = large()
    shape = source.shape
    copyto = np.copyto
    def mutate_header(destination, *args, **kwargs):
        result = copyto(destination, *args, **kwargs)
        source.shape = (1, -1, 3)
        return result
    monkeypatch.setattr(np, 'copyto', mutate_header)
    actual = ImageData.from_array(source)
    assert actual.shape == shape and actual.source_shape == shape


@pytest.mark.parametrize('change', ['shape', 'dtype', 'strides'])
def test_public_pixel_header_cannot_invalidate_contract_or_export_facts(tmp_path, change):
    source = ImageData.from_array(np.arange(24, dtype=np.uint8).reshape(2,4,3))
    before = source.to_metadata()
    pixels = source.pixels
    if change == 'shape':
        pixels.shape = (4,2,3)
    elif change == 'dtype':
        pixels.dtype = np.uint8
    else:
        pixels.strides = (0,pixels.strides[1],pixels.strides[2])
    assert source.to_metadata() == before
    assert source.shape == (2,4,3) and source.pixels.dtype == np.float64
    assert source.pixels.tobytes() == ImageData.from_array(np.arange(24,dtype=np.uint8).reshape(2,4,3)).pixels.tobytes()
    record = write_image_data(tmp_path/'stable.png',source)
    assert record.shape == source.shape and record.source_digest == before['digest']


@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('space', ['srgb', 'linear-srgb'])
@pytest.mark.parametrize('quality', [1, 95, 100])
def test_fast_png_retains_pixels_precision_metadata_and_all_source_facts(tmp_path, dtype, space, quality):
    pixels = np.random.default_rng(531).integers(0, np.iinfo(dtype).max+1, (19, 23, 3), dtype=dtype)
    image = ImageData.from_array(pixels, color_space=space, channel_order='BGR',
        orientation=6, apply_orientation=False, provenance={'file_sha256':'3'*64})
    common = dict(quality=quality)
    before = write_image_data(tmp_path/'compatible.png', image, **common).to_dict()
    record = write_image_data(tmp_path/'fast.png', image, png_profile='fast-lossless', **common)
    after = record.to_dict()
    for key in ('path','sha256','bytes'):
        before.pop(key);after.pop(key)
    assert before == after
    decoded = read_image_data(tmp_path/'fast.png')
    assert decoded.pixels.tobytes() == image.with_orientation_applied().pixels.tobytes()
    assert decoded.source_dtype == dtype and decoded.color_space == space
    assert record.sha256 == hashlib.sha256((tmp_path/'fast.png').read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['pixel', 'encoding', 'exception'])
def test_fast_png_failure_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch, fault):
    encoder = writers._encode_png_fast
    def wrong(pixels, encoding):
        if fault == 'exception':
            raise RuntimeError('encoder failed')
        if fault == 'pixel':
            pixels = pixels.copy();pixels[-1,-1,-1] ^= 1
        return encoder(pixels, 'linear-srgb' if fault=='encoding' else encoding)
    monkeypatch.setattr(writers, '_encode_png_fast', wrong)
    path = tmp_path/'prior.png'
    path.write_bytes(b'prior')
    with pytest.raises((OSError, RuntimeError)):
        write_image_data(path, np.zeros((11, 13, 3), np.uint8), png_profile='fast-lossless', overwrite=True)
    assert path.read_bytes() == b'prior'
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('options', [{'png_profile':'bad'}, {'png_profile':None}, {'png_profile':True}])
def test_invalid_profile_cannot_publish(tmp_path, options):
    with pytest.raises(ValueError):
        write_image_data(tmp_path/'bad.png', np.zeros((1,1,3),np.uint8), **options)
    assert not list(tmp_path.iterdir())


def test_fast_png_requires_png_and_retains_no_clobber(tmp_path):
    pixels = np.zeros((2, 3, 3), np.uint8)
    with pytest.raises(ValueError, match='png destination'):
        write_image_data(tmp_path/'bad.tiff', pixels, png_profile='fast-lossless')
    path = tmp_path/'once.png'
    write_image_data(path, pixels, png_profile='fast-lossless')
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_image_data(path, pixels+1, png_profile='fast-lossless')
    assert path.read_bytes() == saved


def test_batch_passes_fast_profile_through_strict_writer(tmp_path):
    owner = ColorCorrection()
    with owner.iter_predict([ImageData.from_array(np.zeros((11,13,3),np.uint8))], output_dir=tmp_path, output_ext='.png',
        allow_identity=True, max_workers=1,
        export_options={'dtype':'uint8','allow_precision_loss':True,'png_profile':'fast-lossless'}) as stream:
        items = list(stream)
    assert len(items) == 1 and items[0].status == 'succeeded', items[0].error


def test_worker_threads_use_exact_serial_kernels_without_nested_pools(monkeypatch):
    require_numba()
    source = large()
    expected = ImageData.from_array(source)
    original_norm, original_quant = numeric._normalize_unsigned_serial, numeric._quantize_unsigned_serial
    calls = []
    def norm(*args):
        calls.append('normalization')
        return original_norm(*args)
    def quant(*args):
        calls.append('quantization')
        return original_quant(*args)
    def no_parallel(*args):
        pytest.fail('worker thread started a parallel I/O kernel')
    monkeypatch.setattr(numeric, '_normalize_unsigned_serial', norm)
    monkeypatch.setattr(numeric, '_quantize_unsigned_serial', quant)
    monkeypatch.setattr(numeric, '_normalize_unsigned', no_parallel)
    monkeypatch.setattr(numeric, '_quantize_unsigned', no_parallel)
    def run(_):
        actual = ImageData.from_array(source)
        storage, error = writers._quantize_strict(actual.pixels, np.dtype('uint16'), 65535)
        assert actual.to_metadata() == expected.to_metadata()
        assert storage.tobytes() == source.tobytes() and error == 0.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(4)))
    assert 'normalization' in calls and 'quantization' in calls


@pytest.mark.parametrize('fault', ['fsync', 'readback', 'publish'])
def test_fast_png_keeps_all_strict_io_failure_barriers(tmp_path, monkeypatch, fault):
    path = tmp_path/'prior.png'
    path.write_bytes(b'prior')
    def failed(*args, **kwargs):
        raise OSError('injected persistence failure')
    if fault == 'fsync':
        monkeypatch.setattr(writers.os, 'fsync', failed)
    elif fault == 'readback':
        read = Path.read_bytes
        def bad_read(target):
            return b'wrong' if target.suffix == '.tmp' else read(target)
        monkeypatch.setattr(Path, 'read_bytes', bad_read)
    else:
        monkeypatch.setattr(writers.os, 'replace', failed)
    with pytest.raises(OSError):
        write_image_data(path, np.zeros((13,17,3),np.uint8), png_profile='fast-lossless', overwrite=True)
    assert path.read_bytes() == b'prior' and not list(tmp_path.glob('.*.tmp'))
