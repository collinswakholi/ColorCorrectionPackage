"""Alternate frozen and final public file workflows in one resource context."""
import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
AUDIT=ROOT.parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repetitions',type=int,default=5)
parser.add_argument('--sizes',nargs='+',default=['512','1080p','4k'])
args=parser.parse_args()
args.output.mkdir(exist_ok=False,parents=True)
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='8'
os.environ['MPLBACKEND']='Agg'
for key,directory in [('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba')]:
    (args.output/directory).mkdir(exist_ok=True); os.environ[key]=str(args.output/directory)
sys.dont_write_bytecode=True
repo=Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
sys.path.insert(0,str(repo))
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits
from ColorCorrectionPipeline import ColorCorrection,LutCache
from ColorCorrectionPipeline.io import write_image_data
controller=threadpool_limits(limits=8)
name='ColorCorrectionPipeline._frozen_direct_reference'
spec=importlib.util.spec_from_file_location(name,ROOT/'baseline-site/ColorCorrectionPipeline/acceleration.py')
old=importlib.util.module_from_spec(spec); sys.modules[name]=old; spec.loader.exec_module(old)
model=AUDIT/'ccp-patches9/ui/benchmark-p78/full.ccpb'
owner=ColorCorrection().load_bundle(model)
report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Alternating same-process public file->prediction->strict PNG8; exact frozen acceleration.py reference, all other baseline numerical code unchanged. Eight numerical threads; no LUT; no refit.',
    'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'threads':threadpool_info(),'runs':[]}
report['source_sha256']={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (repo/'ColorCorrectionPipeline').rglob('*.py')}
try:
    import psutil
    report['affinity']=psutil.Process().cpu_affinity()
    report['host_memory']=dict(psutil.virtual_memory()._asdict())
except ImportError: psutil=None
def timing(values): return {'n':len(values),'median_ms':statistics.median(values),'p95_ms':float(np.percentile(values,95)),'raw_ms':values}
def save(): (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
started=time.perf_counter()
baseline=old.PreparedExecution(owner,{'mode':'direct'},cache=old.LutCache())
report['baseline_prepare_ms']=(time.perf_counter()-started)*1000
started=time.perf_counter()
candidate=owner.prepare_execution({'mode':'direct'},cache=LutCache())
report['candidate_prepare_ms']=(time.perf_counter()-started)*1000
assert baseline.calibration_id==candidate.calibration_id
report['calibration_id']=candidate.calibration_id
report['qualification']=candidate.preparation['qualification']
for size in args.sizes:
    source=AUDIT/'ccp-patches9/ui/inputs'/(size+'.png')
    row={'size':size,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'samples':{'baseline':[],'candidate':[]},'first_file_ms':{}}
    for label,prepared in (('baseline',baseline),('candidate',candidate)):
        started=time.perf_counter(); warm=prepared.predict_image(source)
        row['first_file_ms'][label]=(time.perf_counter()-started)*1000
        del warm
    for repeat in range(args.repetitions):
        outputs={}; records={}
        order=('baseline','candidate') if repeat%2==0 else ('candidate','baseline')
        for label in order:
            gc.collect()
            prepared=baseline if label=='baseline' else candidate
            cpu=time.process_time_ns(); start=time.perf_counter_ns()
            predicted=prepared.predict_image(source)
            predict_done=time.perf_counter_ns()
            path=args.output/('%s-%s-%d.png'%(label,size,repeat))
            exported=write_image_data(path,predicted.images['CC'],dtype='uint8',allow_precision_loss=True)
            done=time.perf_counter_ns()
            sample={'order':list(order),'predict_ms':(predict_done-start)/1e6,'export_ms':(done-predict_done)/1e6,'total_ms':(done-start)/1e6,'process_cpu_ms':(time.process_time_ns()-cpu)/1e6,
                'decode_ms':predicted.diagnostics['seconds']['decode']*1000,'pointwise_ms':predicted.diagnostics['seconds']['pointwise_and_output']*1000}
            row['samples'][label].append(sample)
            outputs[label]=predicted.images['CC']; records[label]=exported
            if label=='candidate':
                assert predicted.diagnostics['backend']['GC_execution']['actual_backend']=='cpu-numba'
            path.unlink()
            del predicted
        error=float(np.max(np.abs(outputs['baseline']-outputs['candidate'])))
        assert error<=5e-12,error
        assert records['baseline'].sha256==records['candidate'].sha256
        row['max_rgb_error']=max(row.get('max_rgb_error',0.),error)
        row['encoded_png_sha256']=records['candidate'].sha256
        row['png_byte_identical']=True
        del outputs,records
        print(size,repeat,{label:row['samples'][label][-1]['total_ms'] for label in order},flush=True)
    row['timings']={label:{key:timing([sample[key] for sample in samples]) for key in ('predict_ms','export_ms','total_ms','decode_ms','pointwise_ms','process_cpu_ms')} for label,samples in row['samples'].items()}
    report['runs'].append(row); save()
    print('SUMMARY',size,{label:row['timings'][label]['total_ms']['median_ms'] for label in row['timings']},flush=True)
baseline.close(); candidate.close()
report['source_unchanged']=report['source_sha256']=={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (repo/'ColorCorrectionPipeline').rglob('*.py')}
assert report['source_unchanged']
report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
