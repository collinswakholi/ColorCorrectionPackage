"""Rotate frozen/default/fast PNG public workflows in one resource context."""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser=argparse.ArgumentParser()
for name in ('site','baseline-site','models','inputs','output'):parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--repetitions',type=int,default=12)
parser.add_argument('--sizes',nargs='+',default=['512','1080p','4k'])
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='8'
for key,directory in [('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba')]:
    (args.output/directory).mkdir(exist_ok=True);os.environ[key]=str(args.output/directory)
os.environ['MPLBACKEND']='Agg';sys.dont_write_bytecode=True
sys.path.insert(0,str(args.site))
import numpy as np
from ColorCorrectionPipeline import ColorCorrection,ImageData,LutCache
from ColorCorrectionPipeline.io import write_image_data
from ColorCorrectionPipeline import pipeline
from threadpoolctl import threadpool_info,threadpool_limits
controller=threadpool_limits(limits=8)

def module(name,relative):
    spec=importlib.util.spec_from_file_location(name,args.baseline_site/relative)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result
old_contract=module('ColorCorrectionPipeline._io_frozen_contract','ColorCorrectionPipeline/contracts.py')
old_reader=module('ColorCorrectionPipeline.io._io_frozen_reader','ColorCorrectionPipeline/io/image_data.py')
old_writer=module('ColorCorrectionPipeline.io._io_frozen_writer','ColorCorrectionPipeline/io/writers.py')
@contextmanager
def context(label):
    # The dataclass initializer, post-init, hashing, decoder, pipeline and color
    # kernels are byte/AST unchanged; bind the frozen from_array method to the
    # shared type so typed-input checks stay valid across the two implementations.
    method=vars(ImageData)['from_array'];reader=pipeline.read_image_data
    try:
        if label=='baseline':
            ImageData.from_array=classmethod(old_contract.ImageData.from_array.__func__)
            pipeline.read_image_data=old_reader.read_image_data
        yield old_writer.write_image_data if label=='baseline' else write_image_data
    finally:
        ImageData.from_array=method;pipeline.read_image_data=reader

def sources():return {path.relative_to(args.site).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.site/'ColorCorrectionPipeline').rglob('*.py')}
report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources(),
    'scope':'Serial same-process public file->direct prediction->strict PNG8. Rotating frozen/default/explicit fast-lossless order; no refit or LUT. Frozen reader/writer/from_array method share unchanged initializer, post-init, hash and color code.',
    'threads':threadpool_info(),'runs':[]}
try:
    import psutil
    report['host_memory']=dict(psutil.virtual_memory()._asdict())
    report['affinity']=psutil.Process().cpu_affinity()
except ImportError:pass
owner=ColorCorrection().load_bundle(args.models/'full.ccpb')
report['model_sha256']=hashlib.sha256((args.models/'full.ccpb').read_bytes()).hexdigest()
start=time.perf_counter();prepared=owner.prepare_execution({'mode':'direct'},cache=LutCache())
report['prepare_ms']=(time.perf_counter()-start)*1000
report['qualification']=prepared.preparation['qualification']
report['calibration_id']=prepared.calibration_id
def stats(values):return {'n':len(values),'median_ms':statistics.median(values),'p95_ms':float(np.percentile(values,95)),'raw_ms':values}
def save():(args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for size in args.sizes:
    path=args.inputs/(size+'.png')
    row={'size':size,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'first_file_ms':{},'samples':{label:[] for label in ('baseline','compatible','fast-lossless')}}
    for label in row['samples']:
        with context(label) as writer:
            target=args.output/('warm-'+label+'-'+size+'.png')
            start=time.perf_counter();result=prepared.predict_image(path)
            prediction_done=time.perf_counter()
            options={'png_profile':'fast-lossless'} if label=='fast-lossless' else {}
            writer(target,result.images['CC'],dtype='uint8',allow_precision_loss=True,**options)
            row['first_file_ms'][label]={'predict_ms':(prediction_done-start)*1000,'total_ms':(time.perf_counter()-start)*1000}
            target.unlink();del result
    for repeat in range(args.repetitions):
        labels=['baseline','compatible','fast-lossless']
        # Cover every position and pair direction, rather than only rotating
        # one cyclic order. Twelve repeats use each permutation twice.
        order=list(itertools.permutations(labels))[repeat % 6]
        facts={};pixel_hashes={}
        for label in order:
            gc.collect()
            with context(label) as writer:
                target=args.output/(label+'-'+size+'-'+str(repeat)+'.png')
                options={'png_profile':'fast-lossless'} if label=='fast-lossless' else {}
                cpu=time.process_time_ns();start=time.perf_counter_ns()
                result=prepared.predict_image(path);predicted=time.perf_counter_ns()
                record=writer(target,result.images['CC'],dtype='uint8',allow_precision_loss=True,**options)
                done=time.perf_counter_ns()
                row['samples'][label].append({'order':order,'predict_ms':(predicted-start)/1e6,'export_ms':(done-predicted)/1e6,'total_ms':(done-start)/1e6,
                    'decode_ms':result.diagnostics['seconds']['decode']*1000,'pointwise_ms':result.diagnostics['seconds']['pointwise_and_output']*1000,
                    'process_cpu_ms':(time.process_time_ns()-cpu)/1e6})
                pixel_hashes[label]=hashlib.sha256(memoryview(np.ascontiguousarray(result.images['CC']))).hexdigest()
                facts[label]=record.to_dict();facts[label].pop('path')
                target.unlink();del result
        assert len(set(pixel_hashes.values()))==1
        assert facts['baseline']==facts['compatible']
        assert {key:value for key,value in facts['compatible'].items() if key not in ('sha256','bytes')}=={key:value for key,value in facts['fast-lossless'].items() if key not in ('sha256','bytes')}
        row.update(float64_output_byte_identical=True,compatible_record_and_bytes_identical=True,fast_source_facts_identical=True,
            output_pixels_sha256=pixel_hashes['compatible'],compatible_png_sha256=facts['compatible']['sha256'],fast_png_sha256=facts['fast-lossless']['sha256'],
            compatible_bytes=facts['compatible']['bytes'],fast_bytes=facts['fast-lossless']['bytes'])
        print(size,repeat,{label:row['samples'][label][-1]['total_ms'] for label in order},flush=True)
    row['timings']={label:{key:stats([value[key] for value in samples]) for key in ('predict_ms','export_ms','total_ms','decode_ms','pointwise_ms','process_cpu_ms')} for label,samples in row['samples'].items()}
    report['runs'].append(row);save()
    print('SUMMARY',size,{label:row['timings'][label]['total_ms']['median_ms'] for label in row['samples']},flush=True)
prepared.close()
assert report['source_sha256']==sources()
report.update(source_unchanged=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
