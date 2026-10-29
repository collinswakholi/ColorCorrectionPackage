"""Untimed UI fast_mode comparison using the same frozen calibration per pair.

PNG8 errors are descriptive render differences, not physical accuracy claims.
Standard UI rendering still uses a CC LUT33; fast mode combines stages in LUT65.
"""
import argparse
import json
import os
import sys
from pathlib import Path

for variable in ['NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS']:
    os.environ[variable]='8'
sys.dont_write_bytecode=True

import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parent
UI=Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\CCP_New_UI')
NATIVE=UI/'_implementation/ui-docs-20261026/sandbox/harness/candidate-win-x64-v3'
sys.path.insert(0,str(UI/'api'))
from colorcorrect import Client,ApplyOptions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
    from ColorCorrectionPipeline.colorimetry import rgb_to_xyz,xyz_to_lab,delta_e00,D65
    benchmark=json.loads((args.benchmark/'benchmark.json').read_text())
    probes={}
    for row in benchmark['runs']:probes.setdefault((row['backend'],row['kind']),row['model'])
    x=np.random.default_rng(92026).integers(0,256,(512,512,3),dtype=np.uint8)
    x[0,:256]=np.arange(256,dtype=np.uint8)[:,None]
    for channel in range(3):
        x[1+channel,:256]=0;x[1+channel,:256,channel]=np.arange(256,dtype=np.uint8)
    image=BASE/'inputs/color-probe.png';Image.fromarray(x).save(image)
    data=dict(scope='Same saved UI model; PNG8 standard GC/WB plus CC-LUT33 versus combined LUT65 fast rendering; no refit. Common D65 CIEDE2000 between output pixels, not physical accuracy.',
        device_environment={key:os.environ.get(key) for key in ['CCP_ORT_DEVICE_ID','CCP_INFERENCE_PROFILE']},runtime={},cases=[])
    for backend in ['cpu','auto']:
        with Client(root=NATIVE,backend=backend,timeout=120) as client:
            detected=client.detect(BASE/'inputs/chart-1000.png')
            data['runtime'][backend]=dict(detection=detected.raw_reply,devices=client.engine.devices(),
                native_stderr=list(client.engine._stderr))
            for kind in ['cc_only','full']:
                model=client.load_calibration(probes[(backend,kind)])
                for name,path in [('common-chart',BASE/'inputs/chart-1000.png'),('color-probe',image)]:
                    rendered={};replies={};outputs={}
                    for fast in [False,True]:
                        result=model.apply(path,output_dir=args.output,options=ApplyOptions(fast_mode=fast))
                        outputs[str(fast)]=str(result.outputs[0]);replies[str(fast)]=result.raw_reply
                        rendered[fast]=np.asarray(Image.open(result.outputs[0]),dtype=np.float64)/255
                    a,b=rendered[False],rendered[True];difference=np.abs(a-b)
                    de=delta_e00(xyz_to_lab(rgb_to_xyz(a),white=D65),xyz_to_lab(rgb_to_xyz(b),white=D65))
                    row=dict(backend=backend,kind=kind,input=name,model=str(model.path),outputs=outputs,replies=replies,
                        rgb=dict(mean=float(difference.mean()),p95=float(np.percentile(difference,95)),maximum=float(difference.max())),
                        delta_e00=dict(mean=float(de.mean()),p95=float(np.percentile(de,95)),maximum=float(de.max())))
                    data['cases'].append(row)
                    (args.output/'accuracy.json').write_text(json.dumps(data,indent=2),encoding='utf8')
                    print(backend,kind,name,row['rgb'],row['delta_e00'],flush=True)


if __name__=='__main__':main()
