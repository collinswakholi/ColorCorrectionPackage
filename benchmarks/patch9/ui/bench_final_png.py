"""Final-source three-call end-to-end strict PNG qualification case."""
import argparse
import hashlib
import sys
from pathlib import Path
from bench_package_api import BASE,measured,stats,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--models-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.site))
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection
    from ColorCorrectionPipeline.io import write_image_data
    from threadpoolctl import threadpool_limits
    controller=threadpool_limits(limits=8)
    root=Path(package.__file__).parent
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')}
    cc=ColorCorrection();cc.load_bundle(args.models_from/'full.ccpb')
    options={'mode':'combined-lut','backend':'auto','grid_sizes':(129,),'max_rgb_error':.006,'max_delta_e00':1.0}
    prepared,cold=measured(lambda:cc.prepare_execution(options))
    assert prepared.preparation['qualification']['selected_mode']=='combined-lut'
    data=dict(source_sha256=hashes,threads=8,scope='Final source; common 4K PNG decode + final GC/WB/CC + public strict PNG8 write; atomic/fsync/readback/hash verification included; n=3 warm calls',
        cold_preparation_ms=cold,preparation=prepared.preparation,warm_cache=[],runs=[])
    for rep in range(3):
        reused,elapsed=measured(lambda:cc.prepare_execution(options));data['warm_cache'].append(dict(ms=elapsed,hit=reused.preparation['cache_hit']));reused.close()
    warmup,elapsed=measured(lambda:prepared.predict_image(str(BASE/'inputs/4k.png')))
    data['first_file_predict_ms']=elapsed;del warmup
    for repetition in range(3):
        def run():
            result,predict=measured(lambda:prepared.predict_image(str(BASE/'inputs/4k.png')))
            record,export=measured(lambda:write_image_data(args.output/f'{repetition}.png',result.images['CC'],dtype='uint8',allow_precision_loss=True))
            return dict(predict_ms=predict,strict_export_ms=export,export_record=vars(record),diagnostics=result.diagnostics)
        row,elapsed=measured(run);row.update(repetition=repetition,total_ms=elapsed);data['runs'].append(row)
        write(args.output/'final-png.json',data);print(repetition,elapsed,flush=True)
    data['timing']=stats([r['total_ms'] for r in data['runs']]);data['predict_timing']=stats([r['predict_ms'] for r in data['runs']]);data['export_timing']=stats([r['strict_export_ms'] for r in data['runs']])
    data['warm_cache_timing']=stats([r['ms'] for r in data['warm_cache']])
    data['source_unchanged']=hashes=={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')}
    write(args.output/'final-png.json',data);prepared.close();print(data['timing'],flush=True)


if __name__=='__main__':main()
