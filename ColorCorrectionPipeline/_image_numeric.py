"""Optional exact CPU image arithmetic; public contracts still validate pixels.

These kernels allocate no model state, change no color equations, and never
publish files. Failed qualification or execution selects the NumPy calculation.
"""
from functools import lru_cache
import logging
import threading

import numpy as np

from .core import accel

# Below 8 MiB of canonical float64, measured complete-file latency showed no
# reliable JIT benefit. Keep small exports on NumPy and avoid their first JIT.
MIN_VALUES = 1024 * 1024
_log = logging.getLogger(__name__)
_normalize_unsigned = None
_quantize_unsigned = None
_normalize_unsigned_serial = None
_quantize_unsigned_serial = None

if accel.HAS_NUMBA:
    _prange = accel.nb_prange

    @accel.nb_njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _normalize_unsigned(source, output, maximum):
        width = source.shape[1]
        for index in _prange(source.shape[0] * width):
            row, column = index // width, index % width
            for channel in range(3):
                output[index * 3 + channel] = (np.float64(source[row, column, channel]) - 0.0) / maximum

    @accel.nb_njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _quantize_unsigned(source, output, maximum):
        error = 0.0
        for index in _prange(source.size):
            rounded = np.rint(source[index] * maximum)
            output[index] = rounded
            error = max(error, abs(rounded / maximum - source[index]))
        return error


    @accel.nb_njit(cache=True, nogil=True, fastmath=False)
    def _normalize_unsigned_serial(source, output, maximum):
        width = source.shape[1]
        for index in range(source.shape[0] * width):
            row, column = index // width, index % width
            for channel in range(3):
                output[index * 3 + channel] = (np.float64(source[row, column, channel]) - 0.0) / maximum

    @accel.nb_njit(cache=True, nogil=True, fastmath=False)
    def _quantize_unsigned_serial(source, output, maximum):
        error = 0.0
        for index in range(source.size):
            rounded = np.rint(source[index] * maximum)
            output[index] = rounded
            error = max(error, abs(rounded / maximum - source[index]))
        return error


@lru_cache(maxsize=8)
def _normalizer_qualified(kernel, dtype_name):
    """Check every integer code, including a strided channel permutation."""
    try:
        dtype = np.dtype(dtype_name)
        maximum = float(np.iinfo(dtype).max)
        codes = np.arange(int(maximum) + 1, dtype=np.uint64).astype(dtype)
        codes = np.resize(codes, codes.size + (-codes.size % 3)).reshape(1, -1, 3)
        for source in (codes, codes[..., ::-1]):
            expected = source.astype(np.float64)
            expected -= 0.0
            expected /= maximum
            output = np.empty(expected.shape, dtype=np.float64)
            kernel(source, output.reshape(-1), maximum)
            if output.tobytes() != expected.tobytes():
                raise ArithmeticError("unsigned normalization differs from NumPy")
        return True
    except Exception as error:
        _log.debug("Image normalization qualification selected NumPy: %s", error)
        return False


@lru_cache(maxsize=8)
def _quantizer_qualified(kernel, dtype_name):
    """Check all code values and rounding boundaries, neighbors and probes."""
    try:
        dtype = np.dtype(dtype_name)
        maximum = float(np.iinfo(dtype).max)
        levels = np.arange(int(maximum) + 1, dtype=np.float64)
        ties = (levels[:-1] + .5) / maximum
        source = np.concatenate((levels / maximum, ties, np.nextafter(ties, 0.),
            np.nextafter(ties, 1.), np.random.default_rng(519).random(4096),
            np.array([0., -0., np.nextafter(0., 1.), np.nextafter(1., 0.), 1.])))
        source.setflags(write=False)
        expected = np.rint(source * maximum).astype(dtype)
        error = float(np.max(np.abs(expected.astype(np.float64) / maximum - source)))
        output = np.empty(source.shape, dtype=dtype)
        actual_error = kernel(source, output, maximum)
        if output.tobytes() != expected.tobytes() or actual_error != error:
            raise ArithmeticError("unsigned quantization or error facts differ from NumPy")
        return True
    except Exception as error:
        _log.debug("Image quantization qualification selected NumPy: %s", error)
        return False


def try_normalize_unsigned(source, output, lo, hi):
    """Only full-range native unsigned RGB has an intrinsic finite/range proof.

    Every possible source code is valid, including values concurrently changed
    by a caller. Writes capture normalized values in the private destination;
    the public ImageData constructor still validates that owned snapshot.
    """
    if (not accel.HAS_NUMBA or source.size < MIN_VALUES or source.dtype.kind != "u"
            or source.dtype.itemsize not in (1, 2) or not source.dtype.isnative
            or lo != 0.0 or np.signbit(lo) or hi != float(np.iinfo(source.dtype).max)):
        return False
    # Worker calls release the GIL without creating nested native worker pools.
    kernel = (_normalize_unsigned if threading.current_thread() is threading.main_thread()
              else _normalize_unsigned_serial)
    if kernel is None or not _normalizer_qualified(kernel, source.dtype.name):
        return False
    try:
        kernel(source, output.reshape(-1), float(hi))
        return True
    except Exception as error:
        _log.debug("Image normalization selected NumPy after backend failure: %s", error)
        return False


def try_quantize_unsigned(values, storage, maximum):
    if (not accel.HAS_NUMBA or values.size < MIN_VALUES or values.dtype != np.dtype("float64")
            or not values.flags.c_contiguous or storage.dtype.kind != "u"
            or storage.dtype.itemsize not in (1, 2) or not storage.dtype.isnative):
        return None
    kernel = (_quantize_unsigned if threading.current_thread() is threading.main_thread()
              else _quantize_unsigned_serial)
    if kernel is None or not _quantizer_qualified(kernel, storage.dtype.name):
        return None
    try:
        error = float(kernel(values.reshape(-1), storage.reshape(-1), float(maximum)))
        if not np.isfinite(error) or error < 0 or error > .5 / maximum + np.finfo(np.float64).eps:
            raise ArithmeticError("invalid quantization error from image backend")
        return error
    except Exception as error:
        # The caller overwrites the entire destination from immutable values.
        _log.debug("Image quantization selected NumPy after backend failure: %s", error)
        return None
