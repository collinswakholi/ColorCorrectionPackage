"""Bounded workers for nogil image kernels, without an OpenMP spin phase."""

import threading
from concurrent.futures import ThreadPoolExecutor

from .core import accel


def worker_count(size, minimum):
    if size < minimum or threading.current_thread() is not threading.main_thread():
        return 1
    try:
        # Reading the configured cap must not initialize an OpenMP pool. Honor
        # an existing caller mask when another component already initialized it.
        try:
            accel.numba.threading_layer()
        except ValueError:
            configured = accel.numba.config.NUMBA_NUM_THREADS
        else:
            configured = accel.numba.get_num_threads()
        return max(1, min(8, int(configured), size // minimum))
    except (AttributeError, RuntimeError):
        return 1


def ranges(length, workers):
    step = (length + workers - 1) // workers
    return [(start, min(start + step, length)) for start in range(0, length, step)]


def run_slices(function, slices, workers):
    """Join every job on failure; no caller receives partially written output."""
    if workers == 1:
        return [function(start, stop) for start, stop in slices]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ccp-image") as pool:
        jobs = [pool.submit(function, start, stop) for start, stop in slices]
        return [job.result() for job in jobs]
