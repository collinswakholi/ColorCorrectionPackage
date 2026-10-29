"""Exhaust all byte RGB colors and probe continuous/curve boundaries."""
import os
from pathlib import Path
import sys
import importlib.util
import hashlib
import json
import time
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--baseline-site',type=Path,required=True)
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--label',default='stress-results')
parser.add_argument('--real-only',action='store_true')
args=parser.parse_args()
ROOT=args.output
ROOT.mkdir(parents=True,exist_ok=True)
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='8'
os.environ['MPLBACKEND']='Agg'
for key,directory in [('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba-stress')]:
    (ROOT/directory).mkdir(exist_ok=True); os.environ[key]=str(ROOT/directory)
sys.dont_write_bytecode=True
repo=args.site.resolve()
sys.path.insert(0,str(repo))
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from ColorCorrectionPipeline import ColorCorrection,LutCache
from ColorCorrectionPipeline.core.execution import evaluate_tone,_tone_parameters
from ColorCorrectionPipeline.modern import apply_tone
from ColorCorrectionPipeline.stage_policies import MonotonicToneModel
from ColorCorrectionPipeline.colorimetry import D65,rgb_to_xyz,xyz_to_lab,lab_to_xyz,xyz_to_rgb,delta_e00
from ColorCorrectionPipeline.io import write_image_data
controller=threadpool_limits(limits=8)
module_name='ColorCorrectionPipeline._frozen_image_export_execution'
spec=importlib.util.spec_from_file_location(module_name,args.baseline_site/'ColorCorrectionPipeline/acceleration.py')
old=importlib.util.module_from_spec(spec); sys.modules[module_name]=old; spec.loader.exec_module(old)
owner=ColorCorrection().load_bundle(args.models/'full.ccpb')
models=[owner.models.model_gc,
    MonotonicToneModel(np.arange(0.,101.,20.),np.arange(0.,101.,20.)),
    MonotonicToneModel(np.array([0.,1.,2.,50.,100.]),np.array([0.,20.,20.,95.,100.]))]
out=ROOT/args.label; out.mkdir(exist_ok=True)
report={'byte_cube':[],'continuous':[],'files':[],'scope':'Candidate bulk execution versus unaccelerated recorded reference. Byte-cube exhaustive RGB differences; DE00 on every16th red plane plus each worst RGB point. No physical calibration truth is claimed.'}
report['source_sha256']={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (repo/'ColorCorrectionPipeline').rglob('*.py')}
if args.real_only: models=models[:1]
def de(a,b): return float(delta_e00(xyz_to_lab(rgb_to_xyz(a),white=D65),xyz_to_lab(rgb_to_xyz(b),white=D65)).max())
def save(): (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
axis=np.arange(256,dtype=np.float64)/255.
gb=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
with old.PreparedExecution(owner,{'mode':'direct'},cache=old.LutCache()) as baseline, owner.prepare_execution({'mode':'direct'},cache=LutCache()) as candidate:
    report['preparation']=candidate.preparation
    assert baseline.calibration_id==candidate.calibration_id
    for index,model in enumerate(models):
        started=time.perf_counter(); maximum=0.; max_de=0.; de_samples=0; q8=0; q16=0
        for red in range(256):
            rgb=np.column_stack((np.full(65536,red/255.),gb))
            if index==0:
                expected,_,_=baseline._direct(rgb,backend=None)
                actual,_,info=candidate._direct(rgb,backend=candidate._direct_backend)
                assert info['GC_execution']['actual_backend']=='cpu-numba',info
            else:
                expected=apply_tone(rgb.reshape(1,-1,3),model).reshape(-1,3)
                actual,info=evaluate_tone(rgb,model,backend='numba',allow_fallback=False)
            difference=np.abs(actual-expected)
            error=float(difference.max()); maximum=max(maximum,error)
            assert error<=5e-12,(index,red,error)
            q8+=int(np.count_nonzero(np.rint(actual*255).astype(np.uint8)!=np.rint(expected*255).astype(np.uint8)))
            q16+=int(np.count_nonzero(np.rint(actual*65535).astype(np.uint16)!=np.rint(expected*65535).astype(np.uint16)))
            worst=int(np.argmax(difference.max(axis=1)))
            if red%16==0:
                max_de=max(max_de,de(actual,expected)); de_samples+=len(actual)
            else:
                max_de=max(max_de,de(actual[worst:worst+1],expected[worst:worst+1])); de_samples+=1
            if red%64==0: print('cube',index,red,error,flush=True)
        assert q8 == 0 and q16 == 0, (index, q8, q16)
        assert max_de <= 1e-8, (index, max_de)
        report['byte_cube'].append({'case':index,'kind':'real full pipeline' if index==0 else 'identity tone' if index==1 else 'plateau/steep tone','samples':256**3,'max_rgb':maximum,'max_de00_probes':max_de,'de00_samples':de_samples,'different_uint8_channels':q8,'different_uint16_channels':q16,'seconds':time.perf_counter()-started})
        save(); print('complete',report['byte_cube'][-1],flush=True)
    rng=np.random.default_rng(61927)
    for index in range(0 if args.real_only else 100):
        x=np.r_[0.,np.sort(rng.uniform(1,99,6)),100.]
        y=np.r_[0.,np.sort(rng.uniform(0,100,6)),100.]
        model=MonotonicToneModel(x,y)
        lab=np.column_stack((np.repeat(x,27),np.tile(np.repeat([-20.,0.,20.],3),len(x)*3),np.tile([-20.,0.,20.],len(x)*9)))
        near=[]
        for direction in (0.,np.inf,-np.inf):
            probe=lab.copy(); probe[:,0]=np.nextafter(probe[:,0],direction)
            near.append(xyz_to_rgb(lab_to_xyz(probe,white=D65),policy='clip'))
        rgb=np.vstack((rng.random((4096,3)),*near))
        expected=apply_tone(rgb.reshape(1,-1,3),model).reshape(-1,3)
        actual,info=evaluate_tone(rgb,model,backend='auto')
        error=float(np.abs(actual-expected).max()); assert error<=5e-12,error
        report['continuous'].append({'curve':index,'samples':len(rgb),'max_rgb':error,'max_de00':de(actual,expected),'backend':info['actual_backend']})
    save()
    for size in (() if args.real_only else ('512','1080p','4k')):
        path=args.inputs/(size+'.png')
        expected=baseline.predict_image(path).images['CC']; actual=candidate.predict_image(path).images['CC']
        error=float(np.abs(actual-expected).max()); assert error<=5e-12,error
        a=write_image_data(out/('baseline-'+size+'.png'),expected,dtype='uint8',allow_precision_loss=True,overwrite=True)
        b=write_image_data(out/('candidate-'+size+'.png'),actual,dtype='uint8',allow_precision_loss=True,overwrite=True)
        assert a.sha256 == b.sha256, size
        report['files'].append({'size':size,'shape':list(actual.shape),'max_rgb':error,'max_de00':de(actual.reshape(-1,3)[::64],expected.reshape(-1,3)[::64]),'png_byte_identical':a.sha256==b.sha256,'baseline_png_sha256':a.sha256,'candidate_png_sha256':b.sha256})
        save(); print('file',report['files'][-1],flush=True)
report['source_unchanged']=report['source_sha256']=={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (repo/'ColorCorrectionPipeline').rglob('*.py')}
assert report['source_unchanged']
save()
