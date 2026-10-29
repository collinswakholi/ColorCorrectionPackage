"""Final-source critical case after endpoint-audit hardening; no fitting rerun."""
import argparse
import hashlib
import sys
from pathlib import Path
from bench_package_api import BASE,measured,stats,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--models-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.site))
    import numpy as np
    import numba
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection
    from ColorCorrectionPipeline.io.image_data import read_image_data
    from ColorCorrectionPipeline.colorimetry import rgb_to_xyz,xyz_to_lab,delta_e00,D65
    from threadpoolctl import threadpool_limits,threadpool_info
    controller=threadpool_limits(limits=8)
    root=Path(package.__file__).parent
    source={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')}
    data=dict(source_sha256=source,threads=8,threadpools=threadpool_info(),
        scope='Final audit-v2 source; 4K real-photo GC/WB/CC; exact frozen P78 bundle; 7 alternating warm raw-float64 public API applications per mode; source ImageData preparation separately reported',profiles={},warm=[])
    cc=ColorCorrection();cc.load_bundle(args.models_from/'full.ccpb')
    options={'strict_default':{},'one_code':{'mode':'auto','grid_sizes':(129,),'max_rgb_error':1/255,'max_delta_e00':.5},
        'direct':{'mode':'direct','backend':'auto'},
        'relaxed':{'mode':'combined-lut','backend':'auto','grid_sizes':(129,),'max_rgb_error':.006,'max_delta_e00':1.0}}
    prepared={}
    for name,opt in options.items():
        obj,elapsed=measured(lambda:cc.prepare_execution(opt));prepared[name]=obj
        data['profiles'][name]=dict(preparation_ms=elapsed,preparation=obj.preparation)
        write(args.output/'final.json',data)
        print(name,obj.preparation['qualification']['selected_mode'],elapsed,flush=True)
    assert prepared['strict_default'].preparation['qualification']['selected_mode']=='direct'
    assert prepared['one_code'].preparation['qualification']['selected_mode']=='direct'
    assert prepared['relaxed'].preparation['qualification']['selected_mode']=='combined-lut'
    hits=[]
    for rep in range(3):
        reused,elapsed=measured(lambda:cc.prepare_execution(options['relaxed']))
        hits.append(dict(ms=elapsed,hit=reused.preparation['cache_hit']));reused.close()
    data['warm_cache']=dict(raw=hits,timing=stats([r['ms'] for r in hits]))
    image,setup=measured(lambda:read_image_data(BASE/'inputs/4k.png'));raw=image.pixels
    data['input_preparation_ms']=setup;data['input_metadata']=image.to_metadata()
    outputs={}
    for name in ['direct','relaxed']:
        result,elapsed=measured(lambda:prepared[name].predict_image(raw))
        data['profiles'][name]['first_ms']=elapsed
    for repetition in range(7):
        for name in (['direct','relaxed'] if repetition%2==0 else ['relaxed','direct']):
            result,elapsed=measured(lambda:prepared[name].predict_image(raw))
            outputs[name]=result.images['CC']
            data['warm'].append(dict(repetition=repetition,mode=name,ms=elapsed,
                numba_threads=numba.get_num_threads(),diagnostics=result.diagnostics))
        write(args.output/'final.json',data);print('repetition',repetition,flush=True)
    data['timing']={name:stats([r['ms'] for r in data['warm'] if r['mode']==name]) for name in ['direct','relaxed']}
    a,b=outputs['direct'],outputs['relaxed'];difference=np.abs(a-b)
    de=np.empty(a.shape[0]*a.shape[1]);fa=a.reshape(-1,3);fb=b.reshape(-1,3)
    for start in range(0,len(de),65536):
        aa,bb=fa[start:start+65536],fb[start:start+65536]
        de[start:start+len(aa)]=delta_e00(xyz_to_lab(rgb_to_xyz(aa),white=D65),xyz_to_lab(rgb_to_xyz(bb),white=D65))
    data['real_image_error']=dict(rgb_mean=float(difference.mean()),rgb_p95=float(np.percentile(difference,95)),
        rgb_max=float(difference.max()),de_mean=float(de.mean()),de_p95=float(np.percentile(de,95)),de_max=float(de.max()))
    data['source_unchanged']=source=={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')}
    data['numba_threads']=numba.get_num_threads();data['numba_threading_layer']=numba.threading_layer()
    write(args.output/'final.json',data)
    for obj in prepared.values():obj.close()
    print('timing',data['timing'],'error',data['real_image_error'],'source_unchanged',data['source_unchanged'],flush=True)


if __name__=='__main__':main()
