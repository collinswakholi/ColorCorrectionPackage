"""Same-process alternation to distinguish runtime variance from mode semantics."""
import argparse
import contextlib
import ctypes
import io
import os
import sys
from pathlib import Path

from bench_package_api import BASE,measured,stats,write


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),
              ('available',ctypes.c_ulonglong),('pagefile_total',ctypes.c_ulonglong),('pagefile_available',ctypes.c_ulonglong),
              ('virtual_total',ctypes.c_ulonglong),('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]


def memory():
    m=MemoryStatus();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    result=dict(host_total_bytes=m.total,host_available_bytes=m.available,host_load_percent=m.load)
    try:
        import psutil
        result['rss_bytes']=psutil.Process().memory_info().rss
        result['python_processes']=[dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss,
            cpu_user_seconds=p.info['cpu_times'].user,cpu_system_seconds=p.info['cpu_times'].system)
            for p in psutil.process_iter(['name','memory_info','cpu_times']) if (p.info['name'] or '').lower().startswith('python')]
    except Exception as exc:result['process_inventory_unavailable']=str(exc)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--models-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.site))
    import numpy as np
    from PIL import Image
    import numba
    from ColorCorrectionPipeline import ColorCorrection,ImageData
    from threadpoolctl import threadpool_limits,threadpool_info
    controller=threadpool_limits(limits=8)
    data=dict(pid=os.getpid(),numba_threads=numba.get_num_threads(),numba_threading_layer=numba.threading_layer(),
        environment={k:os.environ[k] for k in ['NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS']},
        threadpools=threadpool_info(),before=memory(),cases=[])
    for kind,size in [('cc_only','4k'),('full','1080p')]:
        cc=ColorCorrection();cc.load_bundle(args.models_from/f'{kind}.ccpb')
        image=np.asarray(Image.open(BASE/'inputs'/f'{size}.png'),dtype=np.float64)/255
        typed,typed_setup=measured(lambda:ImageData.from_array(image))
        prepared={}
        for mode in ['direct','auto']:prepared[mode]=cc.prepare_execution({'mode':mode,'backend':'auto'})
        row=dict(kind=kind,size=size,typed_preparation_ms=typed_setup,preparations={m:p.preparation for m,p in prepared.items()},
            before=memory(),raw_array=[],typed=[])
        for input_kind,source in [('raw_array',image),('typed',typed)]:
            results={}
            for repetition in range(3):
                for mode in (['direct','auto'] if repetition%2==0 else ['auto','direct']):
                    result,elapsed=measured(lambda:prepared[mode].predict_image(source,stages='final'))
                    row[input_kind].append(dict(repetition=repetition,mode=mode,ms=elapsed,
                        numba_threads=numba.get_num_threads(),diagnostics=result.diagnostics))
                    results[mode]=result.images['CC']
            row[input_kind+'_same_mode_maxerror']=float(np.max(np.abs(results['direct']-results['auto'])))
        row['after']=memory();data['cases'].append(row);write(args.output/'runtime-spread.json',data)
        print(kind,size,{k:{m:stats([r['ms'] for r in row[k] if r['mode']==m])['median_ms'] for m in ['direct','auto']} for k in ['raw_array','typed']},flush=True)
        for p in prepared.values():p.close()
    data['after']=memory();write(args.output/'runtime-spread.json',data)


if __name__=='__main__':main()
