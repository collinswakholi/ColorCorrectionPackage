"""Compare frozen P78 and opt-in P9 package public APIs on UI benchmark inputs.

Fitting uses each package's real detector and shared canonical LabD50 references.
The separate shared-cell experiment fits precisely the UI report's 24 RGB means;
it deliberately does not claim parity between different fitting model families.
"""
import argparse
from collections.abc import Mapping
import contextlib
import io
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

for variable in ['NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS']:
    os.environ[variable]='8'

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
UI = Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\CCP_New_UI')


def measured(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns()-start)/1e6


def stats(values):
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=float(np.percentile(values,95)), raw_ms=values)


def jsonable(value):
    if isinstance(value,Mapping):return dict(value)
    if isinstance(value,Path): return str(value)
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path,data):
    path.write_text(json.dumps(data,indent=2,default=jsonable),encoding='utf8')


def native_target_linear(lab):
    """Reproduce verified UI color.rs constants, avoiding clipped display RGB.

    Native UI D50 uses x=.34567; package canonical D50 uses x=.3457. Keeping
    these separately explicit prevents interpreting a small standards table
    difference as a solver error.
    """
    def xyz(x,y):return np.array([x/y,1,(1-x-y)/y])
    d50=xyz(.34567,.35850);d65=xyz(.31271,.32902)
    B=np.array([[.8951,.2664,-.1614],[-.7502,1.7135,.0367],[.0389,-.0685,1.0296]])
    adaptation=np.linalg.inv(B)@np.diag((B@d65)/(B@d50))@B
    primaries=np.column_stack([xyz(.64,.33),xyz(.30,.60),xyz(.15,.06)])
    rgb_matrix=primaries@np.diag(np.linalg.solve(primaries,d65))
    fy=(lab[:,0]+16)/116
    f=np.column_stack([lab[:,1]/500+fy,fy,fy-lab[:,2]/200])
    xyz50=np.where(f>6/29,f**3,(116*f-16)/(24389/27))*d50
    return xyz50@adaptation.T@np.linalg.inv(rgb_matrix).T


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['p78','auto','direct','combined-lut'],default='p78')
    parser.add_argument('--repetitions',type=int,default=7)
    parser.add_argument('--max-rgb-error',type=float)
    parser.add_argument('--max-delta-e00',type=float)
    parser.add_argument('--label',help='Distinct evidence label for an explicit execution-budget profile')
    parser.add_argument('--typed-warm-case',action='store_true',help='Also amortize source ImageData validation before seven repeated applications')
    parser.add_argument('--models-from',type=Path,
        help='Load frozen P78 bundles for P9 runtime measurements without refitting')
    parser.add_argument('--probe-only',action='store_true',help='Functional shared-cell check, no performance qualification')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.site))
    start=time.perf_counter_ns()
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection, Config, ReferenceProfile, ImageData
    from ColorCorrectionPipeline.colorimetry import ColorSemantics, srgb_to_linear, linear_to_srgb
    from ColorCorrectionPipeline.fitting import fit_observations
    from ColorCorrectionPipeline.core.correction import predict_
    from ColorCorrectionPipeline.io import write_image_data
    from threadpoolctl import threadpool_info, threadpool_limits
    controller=threadpool_limits(limits=8)
    import_ms=(time.perf_counter_ns()-start)/1e6
    evidence=json.loads((BASE/'functional/functional.json').read_text())
    patches=evidence['primary_report']['report_json']['patches']
    reference=ReferenceProfile.from_values(
        [[p['reference_lab_d50'][key] for key in ['l','a','b']] for p in patches],
        color_space='lab',white='D50',scale=1,profile_id='ui-shared-classic',revision=1,
        names=tuple(p['name'] for p in patches),neutral_ids=tuple(f'r4c{i}' for i in range(1,7)))
    execution_options={'mode':args.mode,'backend':'auto'}
    if args.max_rgb_error is not None:execution_options['max_rgb_error']=args.max_rgb_error
    if args.max_delta_e00 is not None:execution_options['max_delta_e00']=args.max_delta_e00
    data=dict(site=str(args.site),imported_from=package.__file__,mode=args.label or args.mode,
        execution_options=execution_options,
        source_sha256={str(path.relative_to(Path(package.__file__).parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in Path(package.__file__).parent.rglob('*.py')},
        python=sys.version,platform=platform.platform(),import_ms=import_ms,
        threads=8,threadpools=threadpool_info(),
        fitting='Common Big photograph resized max1000px; package detector; explicit/modern/reproducible policies; UI canonical LabD50 references; degree1 encoded RGB withbias LS; noFFC/saturation/validation',
        export='Public strictwriter PNG uint8; atomicpublish+fsync+readback verification and hashes included',runs=[],models={})
    inputs={name:BASE/'inputs'/f'{name}.png' for name in ['512','1080p','4k']}
    # Exact same source means/reference triples as UI report, avoiding geometry
    # differences. UI degree1 fits linearRGB/intercept/ridge; package fits
    # encodedRGB/intercept/OLS. Report residuals rather than asserting equivalence.
    sem=ColorSemantics(version='explicit-v1')
    x=np.array([p['measured_srgb'] for p in patches])
    y=sem.reference_to_working_rgb(reference.canonical_lab_d50,clip=False)
    shared,shared_ms=measured(lambda:fit_observations(x,y,[f'r{i//6+1}c{i%6+1}' for i in range(24)],
        {'degree':1,'use_lut':False},'reproducible-v1'))
    shared_output=predict_(x,shared)
    ui_model=json.loads(Path(evidence['primary_model']).read_text())['correction']
    ui_coefficients=np.array(ui_model['coefficients']).reshape(3,3)
    ui_intercept=np.array(ui_model['intercept'])
    ui_output=linear_to_srgb(np.clip(srgb_to_linear(x)@ui_coefficients+ui_intercept,0,1))
    ui_report_output=np.array([p['corrected_srgb'] for p in patches])
    design=np.column_stack([srgb_to_linear(x),np.ones(len(x))])
    target_linear=native_target_linear(reference.canonical_lab_d50)
    numpy_ui_coefficients=np.linalg.solve(design.T@design+np.eye(4)*1e-5,design.T@target_linear)
    errors=np.abs(shared_output-ui_output)
    labs=reference.canonical_lab_d50
    data['shared_cells']=dict(fit_ms=shared_ms,x=x,y=y,package_output=shared_output,ui_output=ui_output,
        reference_display_rgb=np.array([p['reference_srgb'] for p in patches]),
        ui_target_linear=target_linear,package_target_encoded=y,
        native_d50_xy=[.34567,.35850],package_d50_xy=[.3457,.3585],
        ui_report_output=ui_report_output,ui_coefficients=ui_coefficients,ui_intercept=ui_intercept,
        reconstructed_ui_ridge_coefficients=numpy_ui_coefficients,
        reconstructed_ui_ridge_coefficient_max_error=np.max(np.abs(numpy_ui_coefficients-np.vstack([ui_coefficients,ui_intercept]))),
        rgb_abs_mean=errors.mean(),rgb_abs_p95=np.percentile(errors,95),rgb_abs_max=errors.max(),
        package_delta_e00=sem.delta_e00(np.clip(shared_output,0,1),labs,input_encoding='srgb'),
        ui_delta_e00=sem.delta_e00(ui_output,labs,input_encoding='srgb'),fit_report=shared.fit_report)
    write(args.output/'benchmark.json',data)
    if args.probe_only:
        print('Shared-cell functional probe complete; performance fields are not benchmark evidence',flush=True)
        return
    for kind in ['cc_only','full']:
        cc=ColorCorrection()
        cfg=Config(do_ffc=False,do_gc=kind=='full',do_wb=kind=='full',check_saturation=False,
            color_semantics={'version':'explicit-v1'},stage_policy='modern-v1',fitting_policy='reproducible-v1',
            reference_profile=reference,validation_options={'enabled':False},CC_kwargs={'degree':1,'use_lut':False},
            chart_detection={'device':'cpu'})
        if args.models_from:
            _,fit_ms=measured(lambda:cc.load_bundle(args.models_from/f'{kind}.ccpb'))
            fitted=(None,None,False)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                fitted,fit_ms=measured(lambda:cc.run(str(BASE/'inputs/chart-1000.png'),config=cfg))
            if fitted[2]:
                data['models'][kind]=dict(fit_ms=fit_ms,failed=cc.last_run)
                write(args.output/'benchmark.json',data)
                raise RuntimeError(str(cc.last_run))
        model_info=dict(fit_ms=fit_ms,numerical_report=cc.numerical_report,metadata=cc.models.metadata)
        if args.models_from:
            model_info['fit_ms_scope']='load frozen P78 bundle, no refit'
            model_info['source_bundle']=args.models_from/f'{kind}.ccpb'
        else:
            fit_times=[]
            for rep in range(args.repetitions):
                with contextlib.redirect_stdout(io.StringIO()):
                    fitted,elapsed=measured(lambda:cc.run(str(BASE/'inputs/chart-1000.png'),config=cfg))
                if fitted[2]:raise RuntimeError(str(cc.last_run))
                fit_times.append(elapsed)
            model_info['warm_detect_fit_render']=stats(fit_times)
        bundle=args.output/f'{kind}.ccpb'
        _,save_ms=measured(lambda:cc.save_bundle(bundle))
        loaded=ColorCorrection()
        _,load_ms=measured(lambda:loaded.load_bundle(bundle))
        model_info.update(bundle=bundle,save_bundle_ms=save_ms,load_bundle_ms=load_ms)
        model_info['warm_save_bundle']=stats([measured(lambda:cc.save_bundle(bundle))[1] for _ in range(args.repetitions)])
        model_info['warm_load_bundle']=stats([measured(lambda:loaded.load_bundle(bundle))[1] for _ in range(args.repetitions)])
        if args.mode=='p78':
            prepared=None
            model_info['preparation_ms']=0.0
        else:
            # Kept centralized so final public signature can be reconciled with
            # package implementation before any timings are collected.
            prepared,prep_ms=measured(lambda:loaded.prepare_execution(execution_options))
            model_info['preparation_ms']=prep_ms
            model_info['preparation_diagnostics']=prepared.preparation
        data['models'][kind]=model_info
        def apply(image):
            if prepared is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    return loaded.predict_image(image,stages='final')['CC']
            result=prepared.predict_image(image,stages='final')
            return result.images['CC']
        for size,path in inputs.items():
            image=np.asarray(Image.open(path),dtype=np.float64)/255
            first,first_ms=measured(lambda:apply(image))
            direct=np.asarray(first).copy()
            warm=[measured(lambda:apply(image))[1] for _ in range(args.repetitions)]
            row=dict(kind=kind,size=size,first_inmemory_ms=first_ms,inmemory=stats(warm))
            def file_apply(rep):
                output=apply(str(path))
                record=write_image_data(args.output/f'{kind}-{size}-{rep}.png',output,dtype='uint8',allow_precision_loss=True)
                return record
            record,first_file_ms=measured(lambda:file_apply('first'))
            file_times=[measured(lambda:file_apply(rep))[1] for rep in range(args.repetitions)]
            row.update(first_file_png_ms=first_file_ms,file_png=stats(file_times),first_output=record.path,
                input_shape=list(image.shape),output_dtype=str(direct.dtype),export_record=record.to_dict() if hasattr(record,'to_dict') else vars(record))
            # Exact source input deterministic on reload and across calls.
            reloaded=ColorCorrection();reloaded.load_bundle(bundle)
            expected=reloaded.predict_image(image,stages='final')['CC']
            err=np.abs(direct-expected)
            row['quality_vs_saved_direct']=dict(mean=err.mean(),p95=np.percentile(err,95),maximum=err.max())
            if args.typed_warm_case:
                typed,typed_setup=measured(lambda:ImageData.from_array(image))
                _,typed_first=measured(lambda:apply(typed))
                typed_times=[measured(lambda:apply(typed))[1] for _ in range(args.repetitions)]
                row['typed_image_data']=dict(preparation_ms=typed_setup,first_ms=typed_first,timing=stats(typed_times),
                    scope='Repeated identical immutable pixels; source validation/digests amortized; writable adapter copy remains')
            if prepared is not None:
                row['actual_diagnostics']=getattr(prepared.predict_image(image,stages='final'),'diagnostics',None)
            data['runs'].append(row)
            write(args.output/'benchmark.json',data)
            print(kind,size,'memory',row['inmemory']['median_ms'],'file',row['file_png']['median_ms'],flush=True)
        if prepared is not None:
            try:
                exports=[]
                for rep in range(args.repetitions):
                    exported,elapsed=measured(lambda:prepared.export_cube(args.output/f'{kind}.cube'))
                    exports.append(elapsed)
                model_info['cube_export']=dict(timing=stats(exports),record=exported)
            except Exception as exc:
                model_info['cube_export']=dict(error_type=type(exc).__name__,error=str(exc))
            prepared.close()
        else:
            model_info['cube_export']=dict(status='not available in P78 API')
        write(args.output/'benchmark.json',data)
    write(args.output/'benchmark.json',data)


if __name__=='__main__':main()
