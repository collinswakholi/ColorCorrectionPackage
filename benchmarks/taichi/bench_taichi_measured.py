"""Reproducible experimental kernels; this module is never imported by the package.

Run each backend/precision in a fresh process. CPU uses external NumPy buffers.
Vulkan host-to-host uses persistent model arrays, per-call image upload/download,
and converts back to the package's float64 result contract. Resident figures are
synchronized launch+compute lower bounds and must not be sold as pipeline latency.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['cpu', 'vulkan'], required=True)
parser.add_argument('--dtype', choices=['float64', 'float32'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=11)
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--operations', default='lut,lab,ffc,wb,poly3')
parser.add_argument('--sizes', default='512x512,1920x1080,3840x2160')
args = parser.parse_args()

imports_started = time.perf_counter_ns()
import numpy as np
import numba
from PIL import Image
import taichi as ti

ROOT = Path(__file__).resolve().parent
REPO = Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
sys.path.insert(0, str(ROOT / 'baseline_pkg'))
spec = importlib.util.spec_from_file_location('original_accel', ROOT / 'original_accel.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
from ColorCorrectionPipeline.core.correction import _predict_direct, Regressor_Model
from ColorCorrectionPipeline.core.transforms import get_poly_features
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits, threadpool_info
imports_ms = (time.perf_counter_ns() - imports_started) / 1e6

THREADS = min(32, os.cpu_count() or 1)
numba.set_num_threads(THREADS)
thread_limits = threadpool_limits(limits=THREADS)
DT = np.float64 if args.dtype == 'float64' else np.float32
FP = ti.f64 if args.dtype == 'float64' else ti.f32
start_init = time.perf_counter_ns()
ti.init(arch=ti.cpu if args.backend == 'cpu' else ti.vulkan,
        default_fp=FP, cpu_max_num_threads=THREADS, fast_math=False,
        enable_fallback=False, offline_cache=False,
        offline_cache_file_path=str(ROOT / 'ti-cache'))
ti.sync()
init_ms = (time.perf_counter_ns() - start_init) / 1e6
assert ti.lang.impl.current_cfg().arch == (ti.cpu if args.backend == 'cpu' else ti.vulkan)


@ti.func
def srgb_to_lab(r, g, b):
    lr = r / 12.92
    lg = g / 12.92
    lb = b / 12.92
    if r > 0.04045:
        lr = ((r + 0.055) / 1.055) ** 2.4
    if g > 0.04045:
        lg = ((g + 0.055) / 1.055) ** 2.4
    if b > 0.04045:
        lb = ((b + 0.055) / 1.055) ** 2.4
    x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047
    y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
    z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0
    fx = (kap * x + 16.0) / 116.0
    fy = (kap * y + 16.0) / 116.0
    fz = (kap * z + 16.0) / 116.0
    if x > eps:
        fx = x ** (1.0 / 3.0)
    if y > eps:
        fy = y ** (1.0 / 3.0)
    if z > eps:
        fz = z ** (1.0 / 3.0)
    return ti.Vector([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)])


@ti.func
def lab_to_srgb(L, a, b):
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0
    xn = (116.0 * fx - 16.0) / kap
    yn = (116.0 * fy - 16.0) / kap
    zn = (116.0 * fz - 16.0) / kap
    if fx * fx * fx > eps:
        xn = fx * fx * fx
    if fy * fy * fy > eps:
        yn = fy * fy * fy
    if fz * fz * fz > eps:
        zn = fz * fz * fz
    x = xn * 0.95047
    y = yn
    z = zn * 1.08883
    lr = ti.min(1.0, ti.max(0.0, 3.2404542 * x - 1.5371385 * y - 0.4985314 * z))
    lg = ti.min(1.0, ti.max(0.0, -0.9692660 * x + 1.8760108 * y + 0.0415560 * z))
    lb = ti.min(1.0, ti.max(0.0, 0.0556434 * x - 0.2040259 * y + 1.0572252 * z))
    r = 12.92 * lr
    g = 12.92 * lg
    b_ = 12.92 * lb
    if lr > 0.0031308:
        r = 1.055 * lr ** (1.0 / 2.4) - 0.055
    if lg > 0.0031308:
        g = 1.055 * lg ** (1.0 / 2.4) - 0.055
    if lb > 0.0031308:
        b_ = 1.055 * lb ** (1.0 / 2.4) - 0.055
    return ti.Vector([ti.min(1.0, ti.max(0.0, r)), ti.min(1.0, ti.max(0.0, g)), ti.min(1.0, ti.max(0.0, b_))])


@ti.kernel
def k_lab(img: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(img.shape[0]):
        lab = srgb_to_lab(img[i, 0], img[i, 1], img[i, 2])
        for c in ti.static(range(3)):
            out[i, c] = lab[c]


@ti.kernel
def k_ffc(img: ti.types.ndarray(), gain: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(img.shape[0]):
        lab = srgb_to_lab(img[i, 0], img[i, 1], img[i, 2])
        L = ti.min(100.0, ti.max(0.0, lab[0] * gain[i]))
        rgb = lab_to_srgb(L, lab[1], lab[2])
        for c in ti.static(range(3)):
            out[i, c] = rgb[c]


@ti.kernel
def k_wb(img: ti.types.ndarray(), matrix: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(img.shape[0]):
        for c in ti.static(range(3)):
            out[i, c] = img[i, 0] * matrix[0, c] + img[i, 1] * matrix[1, c] + img[i, 2] * matrix[2, c]


@ti.kernel
def k_lut(img: ti.types.ndarray(), lut: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(img.shape[0]):
        g = lut.shape[0]
        rf = ti.min(1.0, ti.max(0.0, img[i, 0])) * (g - 1)
        gf = ti.min(1.0, ti.max(0.0, img[i, 1])) * (g - 1)
        bf = ti.min(1.0, ti.max(0.0, img[i, 2])) * (g - 1)
        r0 = ti.cast(ti.min(rf, g - 2), ti.i32)
        g0 = ti.cast(ti.min(gf, g - 2), ti.i32)
        b0 = ti.cast(ti.min(bf, g - 2), ti.i32)
        rd = rf - r0
        gd = gf - g0
        bd = bf - b0
        for c in ti.static(range(3)):
            c00 = lut[r0, g0, b0, c] * (1.0 - rd) + lut[r0 + 1, g0, b0, c] * rd
            c01 = lut[r0, g0, b0 + 1, c] * (1.0 - rd) + lut[r0 + 1, g0, b0 + 1, c] * rd
            c10 = lut[r0, g0 + 1, b0, c] * (1.0 - rd) + lut[r0 + 1, g0 + 1, b0, c] * rd
            c11 = lut[r0, g0 + 1, b0 + 1, c] * (1.0 - rd) + lut[r0 + 1, g0 + 1, b0 + 1, c] * rd
            out[i, c] = (c00 * (1.0 - gd) + c10 * gd) * (1.0 - bd) + (c01 * (1.0 - gd) + c11 * gd) * bd


POWERS = tuple(tuple(int(v) for v in row) for row in PolynomialFeatures(3).fit(np.zeros((1, 3))).powers_)


@ti.kernel
def k_poly3(img: ti.types.ndarray(), coeff: ti.types.ndarray(), intercept: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(img.shape[0]):
        for c in ti.static(range(3)):
            total = intercept[c]
            for j in ti.static(range(20)):
                term = ti.cast(1.0, FP)
                for d in ti.static(range(3)):
                    for e in ti.static(range(POWERS[j][d])):
                        term *= img[i, d]
                total += term * coeff[j, c]
            out[i, c] = total


def stats(values):
    return {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'min_ms': float(min(values)), 'max_ms': float(max(values)), 'samples_ms': values}


def timed(fn):
    t = time.perf_counter_ns()
    out = fn()
    return out, (time.perf_counter_ns() - t) / 1e6


def error(output, reference):
    diff = abs(np.asarray(output, dtype=np.float64) - reference)
    return {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'rmse': float(np.sqrt(np.mean(diff * diff))), 'all_finite': bool(np.isfinite(output).all())}


rng = np.random.default_rng(20261028)
training = rng.uniform(0, 1, (256, 3))
targets = np.column_stack((0.9 * training[:, 0] + 0.05 * training[:, 1] ** 2,
                           0.96 * training[:, 1] + 0.03 * training[:, 0] * training[:, 2],
                           0.94 * training[:, 2] + 0.04 * training[:, 0] ** 3))
M = Regressor_Model()
M.degree = 3
M.use_lut = False
features, _ = get_poly_features(training, degree=3)
M.model = LinearRegression().fit(features, targets)
coeff = M.model.coef_.T.copy()
intercept = M.model.intercept_.copy()
lut = original.build_3d_lut(lambda v: _predict_direct(v, M), 33)
wb = np.diag(np.array([1.08, 0.97, 1.03]))
photo_path = REPO.parent / 'CCP_New_UI/test_images/ColorCard_Big.JPG'
photo = Image.open(photo_path).convert('RGB')
source_before = hashlib.sha256((ROOT / 'original_accel.py').read_bytes()).hexdigest()
results = {'date': '2026-10-28', 'backend': args.backend, 'requested_dtype': args.dtype,
           'actual_arch': str(ti.lang.impl.current_cfg().arch), 'cpu_fallback_allowed': False,
           'ti_visible_device': os.environ.get('TI_VISIBLE_DEVICE'),
           'vulkan_inventory': json.loads((ROOT / 'vulkan-devices.json').read_text()) if (ROOT / 'vulkan-devices.json').exists() else [],
           'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'numba': numba.__version__,
                        'taichi': '.'.join(map(str, ti.__version__))},
           'cpu_threads': THREADS, 'blas_libraries': threadpool_info(), 'init_ms': init_ms, 'imports_ms': imports_ms,
           'original_accel_sha256': source_before,
           'original_predict_direct_sha256': hashlib.sha256(inspect.getsource(_predict_direct).encode()).hexdigest(),
           'photo': {'name': photo_path.name, 'sha256': hashlib.sha256(photo_path.read_bytes()).hexdigest()},
           'timing_contract': 'Warmed complete function calls. Vulkan host-to-host includes per-call input float64 conversion/upload, kernel, sync, result download/float64 conversion; model parameters and matching FFC map persist. Resident is synchronized launch+compute only.',
           'original_contract': 'Original accel functions on float64. WB is the unchanged img@diag expression only, excluding detection/fit. poly3 is original _predict_direct with degree3/noLUT and fitted sklearn model, including existing feature generation and garbage collections. LUT is deployed-style 33cube color-correction inference with fixed fitted polynomial model.',
           'repeats': args.repeats, 'warmups': 3, 'cases': [], 'failures': []}
sizes = [(64, 64)] if args.smoke else [tuple(map(int, size.split('x'))) for size in args.sizes.split(',')]
firsts = set()
for w, h in sizes:
    for dataset in ['photo', 'seeded_random']:
        if dataset == 'photo':
            img = np.asarray(photo.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
        else:
            img = np.random.default_rng(20261028 + w + h).random((h, w, 3), dtype=np.float64)
        flat64 = np.ascontiguousarray(img.reshape(-1, 3))
        yy, xx = np.mgrid[:h, :w]
        gain = np.ascontiguousarray(1 + 0.18 * ((xx / max(1, w - 1) - 0.5) ** 2 + (yy / max(1, h - 1) - 0.5) ** 2))
        del yy, xx
        ops = {
            'lut': (lambda: original.trilinear_lut_apply(img, lut, 33), k_lut, [lut]),
            'lab': (lambda: original.srgb_to_lab_numba(img), k_lab, []),
            'ffc': (lambda: original.apply_ffc_float(img, gain), k_ffc, [gain.reshape(-1)]),
            'wb': (lambda: img @ wb, k_wb, [wb]),
            'poly3': (lambda: _predict_direct(img, M), k_poly3, [coeff, intercept]),
        }
        for operation in args.operations.split(','):
            baseline, kernel, params = ops[operation]
            case = {'size': [w, h], 'pixels': w * h, 'dataset': dataset, 'operation': operation,
                    'input_sha256': hashlib.sha256(flat64.data).hexdigest()}
            print('CASE', args.backend, args.dtype, w, h, dataset, operation, flush=True)
            try:
                reference, original_first_ms = timed(baseline)
                params = [np.ascontiguousarray(p, dtype=DT) for p in params]
                if args.backend == 'vulkan':
                    talloc = time.perf_counter_ns()
                    device_img = ti.ndarray(FP, shape=flat64.shape)
                    device_out = ti.ndarray(FP, shape=flat64.shape)
                    device_params = []
                    for param in params:
                        dev = ti.ndarray(FP, shape=param.shape)
                        dev.from_numpy(param)
                        device_params.append(dev)
                    ti.sync()
                    case['persistent_allocation_and_parameter_upload_ms'] = (time.perf_counter_ns() - talloc) / 1e6
                    def host_call():
                        device_img.from_numpy(np.ascontiguousarray(flat64, dtype=DT))
                        kernel(device_img, *device_params, device_out)
                        ti.sync()
                        return device_out.to_numpy().astype(np.float64, copy=False).reshape(img.shape)
                    def resident_call():
                        kernel(device_img, *device_params, device_out)
                        ti.sync()
                else:
                    def host_call():
                        arr = np.ascontiguousarray(flat64, dtype=DT)
                        out = np.empty_like(arr)
                        kernel(arr, *params, out)
                        ti.sync()
                        return out.astype(np.float64, copy=False).reshape(img.shape)
                    resident_call = None
                output, first_ms = timed(host_call)
                if operation not in firsts:
                    case['first_original_call_ms'] = original_first_ms
                    case['first_taichi_call_including_jit_ms'] = first_ms
                    firsts.add(operation)
                case['error_vs_original_float64'] = error(output, reference)
                for _ in range(3):
                    baseline(); host_call()
                    if resident_call:
                        resident_call()
                timings = {'original': [], 'taichi_host_to_host': []}
                if resident_call:
                    timings['taichi_resident'] = []
                # Alternate randomized order, with every timed Vulkan call synchronized.
                order_rng = np.random.default_rng(476)
                calls = {'original': baseline, 'taichi_host_to_host': host_call}
                if resident_call:
                    calls['taichi_resident'] = resident_call
                for rep in range(args.repeats):
                    order = list(calls)
                    order_rng.shuffle(order)
                    for label in order:
                        _, elapsed = timed(calls[label])
                        timings[label].append(elapsed)
                case['timings'] = {label: stats(vals) for label, vals in timings.items()}
                case['host_speedup_vs_original'] = case['timings']['original']['median_ms'] / case['timings']['taichi_host_to_host']['median_ms']
                results['cases'].append(case)
            except Exception as exc:
                failure = dict(case, error=type(exc).__name__ + ': ' + str(exc))
                results['failures'].append(failure)
                print('FAILED', failure['error'], flush=True)
            args.output.write_text(json.dumps(results, indent=2), encoding='utf8')
print('SAVED', args.output, 'cases', len(results['cases']), 'failures', len(results['failures']), flush=True)
