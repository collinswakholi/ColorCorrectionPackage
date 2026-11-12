"""Bounded workers for nogil image kernels, without an OpenMP spin phase."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_budget = threading.local()


def set_image_worker_limit(limit):
    """Initialize a private batch thread with the CPU budget its caller granted."""
    _budget.limit = limit


def image_worker_limit():
    if threading.current_thread() is not threading.main_thread():
        return getattr(_budget, "limit", 1)
    # Do not import or initialize an optional JIT to schedule an unfitted batch.
    numba = sys.modules.get("numba")
    try:
        # Reading the configured cap must not initialize an OpenMP pool. Honor
        # an existing caller mask when another component already initialized it.
        try:
            numba.threading_layer()
        except ValueError:
            configured = numba.config.NUMBA_NUM_THREADS
        else:
            configured = numba.get_num_threads()
        return max(1, min(8, int(configured)))
    except (AttributeError, RuntimeError):
        return 1


def worker_count(size, minimum):
    return max(1, min(image_worker_limit(), size // minimum)) if size >= minimum else 1


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
