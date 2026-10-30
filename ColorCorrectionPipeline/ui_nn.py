"""Ordered float64 training arithmetic for the UI residual network.

The native fitter starts dense sums with the bias and reduces gradients in row
order. BLAS reduction trees change low bits and can send long ReLU training down
a different path. Inference on existing saved networks keeps its original path.
"""

import logging
import math

import numpy as np

from .core import accel

_enabled = accel.HAS_NUMBA
_log = logging.getLogger(__name__)


def _dense_numpy(a, weight, bias):
    out = np.broadcast_to(bias, (len(a), len(bias))).copy()
    for index in range(a.shape[1]):
        out += a[:, index, None] * weight[index]
    return out


def _sum_numpy(a):
    out = np.zeros(a.shape[1], dtype=np.float64)
    for row in a:
        out += row
    return out


if accel.HAS_NUMBA:

    @accel.nb_njit(cache=True, fastmath=False)
    def _dense_numba(a, weight, bias):
        out = np.empty((len(a), weight.shape[1]), dtype=np.float64)
        for row in range(len(a)):
            for col in range(weight.shape[1]):
                value = bias[col]
                for index in range(weight.shape[0]):
                    value += a[row, index] * weight[index, col]
                out[row, col] = value
        return out

    @accel.nb_njit(cache=True, fastmath=False)
    def _sum_numba(a):
        out = np.zeros(a.shape[1], dtype=np.float64)
        for row in range(len(a)):
            for col in range(a.shape[1]):
                out[col] += a[row, col]
        return out


def _fallback(error):
    global _enabled
    _enabled = False
    _log.warning("Ordered NN arithmetic uses NumPy after Numba failure: %s", error)


def dense(a, weight, bias):
    if _enabled:
        try:
            return _dense_numba(a, weight, bias)
        except Exception as error:
            _fallback(error)
    return _dense_numpy(a, weight, bias)


def dot(a, b):
    return dense(a, b, np.zeros(b.shape[1], dtype=np.float64))


def sum_rows(a):
    if _enabled:
        try:
            return _sum_numba(a)
        except Exception as error:
            _fallback(error)
    return _sum_numpy(a)


def backend():
    return "cpu-numba-ordered-f64" if _enabled else "cpu-numpy-ordered-f64"


def design(encoded, degree):
    """Native scalar sRGB decode and successive polynomial powers, training only."""
    linear = np.array(
        [
            v / 12.92 if v <= 0.04045 else math.pow((v + 0.055) / 1.055, 2.4)
            for v in encoded.ravel()
        ],
        dtype=np.float64,
    ).reshape(encoded.shape)
    powers = [np.ones_like(linear)]
    for _ in range(degree):
        powers.append(powers[-1] * linear)
    columns = []
    for total in range(degree + 1):
        for r in range(total, -1, -1):
            for g in range(total - r, -1, -1):
                b = total - r - g
                columns.append(powers[r][:, 0] * powers[g][:, 1] * powers[b][:, 2])
                if len(columns) == 96:
                    return np.column_stack(columns)
    return np.column_stack(columns)
