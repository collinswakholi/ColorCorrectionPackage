"""Real released UI worker through its Python SDK; writes only audit evidence.

Use --functional for untimed coverage, otherwise wait for an exclusive benchmark
window before running. All reported operation times include Python SDK overhead.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True

for variable in ['NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS']:
    os.environ[variable]='8'

import numpy as np
from PIL import Image, ImageOps
import tifffile

UI = Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\CCP_New_UI')
NATIVE = UI / '_implementation/ui-docs-20261026/sandbox/harness/candidate-win-x64-v3'
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(UI / 'api'))
from colorcorrect import Client, CalibrationOptions, ApplyOptions


class UncachedClient(Client):
    def _request(self, command, **parameters):
        if command == 'detect':
            parameters['bypass_cache'] = True
        return super()._request(command, **parameters)


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=plain), encoding='utf8')


def measured(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1e6


def summary(values):
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=float(np.percentile(values, 95)), raw_ms=values)


def fixtures():
    folder = BASE / 'inputs'
    folder.mkdir(exist_ok=True)
    original = ImageOps.exif_transpose(Image.open(UI / 'test_images/Sample_3572.JPG')).convert('RGB')
    result = {}
    for name, shape in [('512', (512, 512)), ('1080p', (1920, 1080)), ('4k', (3840, 2160))]:
        path = folder / (name + '.png')
        if not path.exists():
            original.resize(shape, Image.Resampling.LANCZOS).save(path)
        result[name] = path
    path16 = folder / 'sixteen.tif'
    if not path16.exists():
        x = np.arange(256, dtype=np.uint16) * 257
        tifffile.imwrite(path16, np.broadcast_to(x[None, :, None], (32, 256, 3)), photometric='rgb')
    chart_path = folder / 'chart-1000.png'
    if not chart_path.exists():
        chart = ImageOps.exif_transpose(Image.open(UI/'test_images/ColorCard_Big.JPG')).convert('RGB')
        chart.thumbnail((1000,1000),Image.Resampling.LANCZOS)
        chart.save(chart_path)
    return result, path16


def model_options(kind='cc_only', fast=False):
    return CalibrationOptions(model='linear', degree=1, fit_samples=1,
        do_ffc=False, do_gc=kind == 'full', do_wb=kind == 'full', do_sat=False,
        skip_saturated=False, cross_validate=False, write_output=False, fast_mode=fast)


def functional(output, inputs, sixteen):
    data = dict(cases=[])
    with UncachedClient(root=NATIVE, backend='auto', timeout=120) as client:
        data['ready_capabilities'] = sorted(client.capabilities)
        data['devices'] = client.engine.devices()
        for name in ['ColorCard_Big.JPG', 'ColorCard_Combo.JPG', 'Sample_3572.JPG', 'Sample_3573.JPG', 'Blank.JPG']:
            row = dict(image=name)
            try:
                det = client.detect(UI / 'test_images' / name)
                row.update(charts=len(det.charts), detection=det.raw_reply)
                if name != 'Blank.JPG':
                    model = client.calibrate(UI / 'test_images' / name, detection=det,
                        output_dir=output, options=model_options())
                    row.update(model=model.path, fit=model.raw_reply)
                    result = model.apply(inputs['512'], output_dir=output)
                    a = np.asarray(Image.open(result.outputs[0]))
                    loaded = client.load_calibration(model.path)
                    reapplied = loaded.apply(inputs['512'], output_dir=output)
                    b = np.asarray(Image.open(reapplied.outputs[0]))
                    row['reload_bit_identical'] = bool(np.array_equal(a, b))
                    for size in [33, 65]:
                        lut = loaded.export_lut(output / f'{Path(name).stem}-{size}.cube', size=size)
                        row[f'export_{size}'] = dict(path=lut.path, bytes=lut.path.stat().st_size)
                    if name == 'ColorCard_Big.JPG':
                        data['primary_model'] = model.path
                        data['primary_handshake'] = det.handshake
                        data['primary_report'] = model.raw_reply
                        mixed = loaded.apply([inputs['512'], sixteen], output_dir=output, on_error='continue')
                        row['mixed_8_16_batch'] = mixed.raw_reply
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc),
                           raw_reply=getattr(exc, 'raw_reply', None))
            data['cases'].append(row)
            write(output / 'functional.json', data)
            print(name, row.get('charts'), row.get('error_type'), flush=True)
    return data


def benchmark(output, inputs, repetitions):
    data = dict(runs=[], startup=[], cold_model_reload=[], summary={})
    chart = BASE / 'inputs/chart-1000.png'
    # Each worker lives across all warm operations; actual backend recorded.
    for backend in ['cpu', 'auto']:
        models={}
        client, init_ms = measured(lambda: UncachedClient(root=NATIVE, backend=backend, timeout=120))
        try:
            capabilities, startup_ms = measured(lambda: sorted(client.capabilities))
            data['startup'].append(dict(backend=backend, sdk_construction_ms=init_ms,
                worker_ready_ms=startup_ms, capabilities=capabilities, devices=client.engine.devices()))
            det, first_det_ms = measured(lambda: client.detect(chart))
            data['startup'][-1]['first_detection_ms'] = first_det_ms
            data['startup'][-1]['first_detection_reply'] = det.raw_reply
            for kind in ['cc_only', 'full']:
                model, first_fit_ms = measured(lambda: client.calibrate(chart, detection=det,
                    output_dir=output, options=model_options(kind, fast=True)))
                models[kind]=model.path
                data['startup'][-1][f'first_fit_{kind}_ms'] = first_fit_ms
                data['startup'][-1][f'first_fit_{kind}_reply'] = model.raw_reply
                for name, path in inputs.items():
                    first, first_ms = measured(lambda: model.apply(path, output_dir=output,
                        options=ApplyOptions(output_format='png', fast_mode=True)))
                    row = dict(backend=backend, kind=kind, size=name, first_ms=first_ms,
                        first_reply=first.raw_reply, model=model.path, first_output=first.outputs[0], warm=[])
                    for rep in range(repetitions):
                        result, elapsed = measured(lambda: model.apply(path, output_dir=output,
                            options=ApplyOptions(output_format='png', fast_mode=True)))
                        row['warm'].append(dict(repetition=rep, ms=elapsed, reply=result.raw_reply,
                            output=result.outputs[0]))
                    row['timing'] = summary([item['ms'] for item in row['warm']])
                    data['runs'].append(row)
                    write(output / 'benchmark.json', data)
                    print(backend, kind, name, row['timing']['median_ms'], flush=True)
                # Timed model handle load is Python metadata validation; first apply
                # separately above includes actual worker model parse + LUT build.
                loads = [measured(lambda: client.load_calibration(model.path))[1] for _ in range(repetitions)]
                data['summary'][backend + '/' + kind + '/load_handle'] = summary(loads)
                fits = [measured(lambda: client.calibrate(chart, detection=det, output_dir=output,
                    options=model_options(kind, fast=True)))[1] for _ in range(repetitions)]
                data['summary'][backend + '/' + kind + '/fit_save_report'] = summary(fits)
                for size in [33, 65]:
                    times = [measured(lambda: model.export_lut(output / f'{backend}-{kind}-{size}-{rep}.cube', size=size))[1]
                             for rep in range(repetitions)]
                    data['summary'][backend + '/' + kind + f'/export_{size}'] = summary(times)
            detects = [measured(lambda: client.detect(chart))[1] for _ in range(repetitions)]
            data['summary'][backend + '/uncached_detect'] = summary(detects)
            write(output / 'benchmark.json', data)
        finally:
            client.close()
        # A new worker has no pointwise LUT cache from calibration. Typed SDK
        # load only validates a Python handle; first apply performs native model
        # parsing, LUT construction, optional GPU initialization, and rendering.
        for kind,path in models.items():
            with UncachedClient(root=NATIVE,backend=backend,timeout=120) as cold:
                _,ready_ms=measured(lambda:sorted(cold.capabilities))
                handle,handle_ms=measured(lambda:cold.load_calibration(path))
                result,apply_ms=measured(lambda:handle.apply(inputs['512'],output_dir=output,
                    options=ApplyOptions(output_format='png',fast_mode=True)))
                data['cold_model_reload'].append(dict(backend=backend,kind=kind,worker_ready_ms=ready_ms,
                    load_handle_ms=handle_ms,first_native_load_build_apply_png_ms=apply_ms,reply=result.raw_reply))
            write(output/'benchmark.json',data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--functional', action='store_true')
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, sixteen = fixtures()
    metadata = dict(python=sys.version, platform=platform.platform(), sdk_version='2.2.1', threads=8,
        sdk_source=str(UI/'api'), sdk_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (UI/'api').glob('colorcorrect*.py')}, native_root=str(NATIVE),
        worker_sha256=hashlib.sha256((NATIVE/'ccp-engine.exe').read_bytes()).hexdigest(),
        model_sha256={str(path.relative_to(NATIVE)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (NATIVE/'models').rglob('*.onnx')},
        released_worker_source='ee2522eaddb615dcbddc2ea4505e9e435bfc08ae',
        ui_head='16c6f42176cbd392c167ecabea4bb2519984fda8',
        scope='File to 8-bit PNG; decode+ICC/orientation+apply+encode+SDK; 65^3 combined LUT fast mode; FFC/saturation off; no cross validation; fits use common chart max1000px',
        fixtures={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in inputs.items()})
    write(args.output/'metadata.json', metadata)
    if args.functional:
        functional(args.output, inputs, sixteen)
    else:
        benchmark(args.output, inputs, args.repetitions)


if __name__ == '__main__':
    main()
