"""Raw fixed-LUT stress near transfer-function/gamut endpoints; not promotion."""
import argparse
import hashlib
import sys
from pathlib import Path
from bench_package_api import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--models-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.site))
    import numpy as np
    from ColorCorrectionPipeline import ColorCorrection
    from ColorCorrectionPipeline.acceleration import _measure
    from ColorCorrectionPipeline.core.execution import apply_lut
    from threadpoolctl import threadpool_limits
    controller=threadpool_limits(limits=8)
    def cube(axis):return np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    cases={'black_bytes_0_16':cube(np.arange(17)/255),'white_bytes_239_255':cube(np.arange(239,256)/255),
        'black_first4cells_33':cube(np.linspace(0,4/128,33)),'white_last4cells_33':cube(np.linspace(1-4/128,1,33)),
        'random_bytes_262144':np.random.default_rng(319067).integers(0,256,(262144,3)).astype(float)/255}
    data=dict(scope='Untimed fixed129 LUT stress; intentionally loose prepare budget only exposes table for measurement; these are raw errors, not a qualified profile',models={})
    for kind in ['cc_only','full']:
        cc=ColorCorrection();cc.load_bundle(args.models_from/f'{kind}.ccpb')
        prepared=cc.prepare_execution({'mode':'combined-lut','backend':'auto','grid_sizes':(129,),
            'max_rgb_error':1,'max_delta_e00':100})
        row={}
        for name,rgb in cases.items():
            expected=prepared._direct(rgb)[0];actual,backend=apply_lut(rgb,prepared._lut,backend='auto')
            error=_measure(actual,expected,prepared.semantics.output_encoding)
            ix=int(np.argmax(np.abs(actual-expected).max(axis=1)))
            row[name]=dict(error=error,input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                rgb_max_witness={'input':rgb[ix].tolist(),'direct':expected[ix].tolist(),'lut':actual[ix].tolist()})
            print(kind,name,error['max_rgb_error'],error['max_delta_e00'],flush=True)
        data['models'][kind]=row;write(args.output/'endpoint-stress.json',data);prepared.close()


if __name__=='__main__':main()
