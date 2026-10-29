"""Untimed dense stress and image errors; retains counterexamples to sample bounds."""
import argparse
import sys
from pathlib import Path

from bench_package_api import BASE,measured,stats,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--models-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-rgb-error',type=float,default=1/255);p.add_argument('--max-delta-e00',type=float,default=.5)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.site))
    import numpy as np
    from ColorCorrectionPipeline import ColorCorrection
    from ColorCorrectionPipeline.core.execution import apply_lut
    from ColorCorrectionPipeline.colorimetry import rgb_to_xyz,xyz_to_lab,delta_e00,D65
    from ColorCorrectionPipeline.io.image_data import read_image_data
    from threadpoolctl import threadpool_limits
    controller=threadpool_limits(limits=8)
    data=dict(scope='Untimed independent 67^3 cell-center RGB stress, then full image errors; same frozen P78 calibrations; explicit preview options; budgets are empirical, not all-domain guarantees',models={})
    options={'mode':'auto','backend':'auto','max_rgb_error':args.max_rgb_error,'max_delta_e00':args.max_delta_e00}
    def compare(actual,expected,source):
        difference=np.abs(actual-expected)
        flat_a=actual.reshape(-1,3);flat_b=expected.reshape(-1,3);flat_source=source.reshape(-1,3)
        de=np.empty(len(flat_a))
        for start in range(0,len(flat_a),65536):
            a,b=flat_a[start:start+65536],flat_b[start:start+65536]
            de[start:start+len(a)]=delta_e00(xyz_to_lab(rgb_to_xyz(a),white=D65),xyz_to_lab(rgb_to_xyz(b),white=D65))
        pixel=int(np.argmax(difference.reshape(-1,3).max(axis=1)));pixel_de=int(np.argmax(de))
        return dict(pixels=len(flat_a),rgb_mean=float(difference.mean()),rgb_p95=float(np.percentile(difference,95)),
            rgb_max=float(difference.max()),de_mean=float(de.mean()),de_p95=float(np.percentile(de,95)),de_max=float(de.max()),
            rgb_max_witness=dict(input=flat_source[pixel].tolist(),direct=flat_b[pixel].tolist(),lut=flat_a[pixel].tolist(),flat_pixel_index=pixel),
            de_max_witness=dict(input=flat_source[pixel_de].tolist(),direct=flat_b[pixel_de].tolist(),lut=flat_a[pixel_de].tolist(),flat_pixel_index=pixel_de))
    axis=(np.arange(67)+.5)/67
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    for kind in ['cc_only','full']:
        cc=ColorCorrection();cc.load_bundle(args.models_from/f'{kind}.ccpb')
        prepared=cc.prepare_execution(options)
        row=dict(preparation=prepared.preparation,options=options,warm_cache=[])
        for repetition in range(3):
            reused,elapsed=measured(lambda:cc.prepare_execution(options))
            row['warm_cache'].append(dict(ms=elapsed,cache_hit=reused.preparation['cache_hit']))
            reused.close()
        row['warm_cache_timing']=stats([r['ms'] for r in row['warm_cache']])
        if prepared._lut is None:
            row['error']='No LUT selected under requested options'
        else:
            expected=prepared._direct(grid)[0]
            actual,backend=apply_lut(grid,prepared._lut,backend='auto')
            row['dense_67_centers']=compare(actual,expected,grid)
            row['dense_67_centers']['backend']=backend
            row['images']={}
            for name in ['chart-1000','color-probe','512','1080p','4k']:
                image=read_image_data(BASE/'inputs'/f'{name}.png')
                source=image.pixels
                expected=prepared._direct(source)[0]
                actual=prepared.predict_image(image).images['CC']
                row['images'][name]=compare(actual,expected,source)
                print(kind,name,row['images'][name]['rgb_max'],row['images'][name]['de_max'],flush=True)
                data['models'][kind]=row;write(args.output/'stress.json',data)
        data['models'][kind]=row;write(args.output/'stress.json',data)
        print(kind,'dense',row.get('dense_67_centers'),flush=True)
        prepared.close()


if __name__=='__main__':main()
