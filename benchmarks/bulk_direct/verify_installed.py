"""Python3.8+ installed-wheel direct tone/backend/persistence probe."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import zipfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expect-numba',choices=['yes','no'],required=True)
    args=parser.parse_args()
    import numpy as np
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection,LutCache
    from ColorCorrectionPipeline.core import execution
    from ColorCorrectionPipeline.modern import apply_tone
    from ColorCorrectionPipeline.stage_policies import MonotonicToneModel
    assert Path(sys.prefix).resolve() in Path(package.__file__).resolve().parents
    assert sys.prefix!=sys.base_prefix
    expect=args.expect_numba=='yes'
    assert execution.accel.HAS_NUMBA is expect
    work=args.output.parent/args.output.stem
    work.mkdir(parents=True,exist_ok=True)
    model=MonotonicToneModel(np.array([0.,10.,30.,50.,80.,100.]),np.array([0.,20.,35.,65.,90.,100.]))
    rgb=np.random.default_rng(228).random((31,19,3))
    rgb[0,0],rgb[-1,-1]=0,1
    reference=apply_tone(rgb.reshape(1,-1,3),model).reshape(rgb.shape)
    actual,info=execution.evaluate_tone(rgb,model,backend='auto')
    error=float(np.abs(reference-actual).max())
    assert error<=5e-12
    assert info['actual_backend']==('cpu-numba' if expect else 'cpu-numpy'),info
    if not expect: assert np.array_equal(actual,reference)
    owner=ColorCorrection()
    owner.models.model_gc=model
    owner.models.model_wb=np.diag([.85,.9,.8])
    from ColorCorrectionPipeline.core.correction import Regressor_Model
    from ColorCorrectionPipeline.portable_inference import AffinePredictor
    cc=Regressor_Model()
    cc.degree,cc.portable_inference,cc.use_lut=1,True,False
    cc.model=AffinePredictor(np.eye(3,4,k=1)*.8,np.array([.01,.02,.03]))
    owner.models.model_cc=(cc,{},'ours')
    bundle=work/'tone.ccpb'
    owner.save_bundle(bundle)
    resumed=ColorCorrection().load_bundle(bundle)
    policy={'mode':'direct','validation_samples':64}
    with owner.prepare_execution(policy,cache=LutCache()) as before,resumed.prepare_execution(policy,cache=LutCache()) as after:
        first=before.predict_image(rgb)
        recorded,_,_=before._direct(rgb)
        pipeline_error=float(np.abs(recorded-first.images['CC']).max())
        assert pipeline_error<=5e-12
        if expect:
            assert first.diagnostics['backend']['pointwise_execution']=='owned-buffer-bulk-direct'
            assert first.diagnostics['backend']['CC']['calls_by_backend']=={'cpu-numba':1}
        else: assert np.array_equal(recorded,first.images['CC'])
        second=after.predict_image(rgb)
        assert np.array_equal(first.images['CC'],second.images['CC'])
        assert first.diagnostics['backend']['GC_execution']['actual_backend']==info['actual_backend']
        assert before.calibration_id==after.calibration_id
        source_id=before.calibration_id
        proof=before.preparation['qualification']
        source=first.images['CC'].copy()
    owner.models.save(str(work),'trusted-local')
    legacy=ColorCorrection()
    legacy.models.load(str(work),'trusted-local')
    with legacy.prepare_execution(policy,cache=LutCache()) as prepared:
        assert np.array_equal(prepared.predict_image(rgb).images['CC'],source)
    with zipfile.ZipFile(bundle) as archive:
        manifest=archive.read('manifest.json').decode()
        assert '_fused_tone' not in manifest and 'qualified-pointwise-' not in manifest and '_affine_inplace' not in manifest
    report={'python':platform.python_version(),'package':str(package.__file__),'numba':expect,
        'versions':{name:metadata.version(name) for name in ('numpy','scipy','colour-science')},
        'tone_backend':info,'max_rgb_reference_error':error,'bulk_pipeline_rgb_reference_error':pipeline_error,'bulk_execution_verified':expect,'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),
        'calibration_id':source_id,'bundle_reload_exact':True,'trusted_pickle_reload_exact':True,
        'runtime_kernel_state_not_serialized':True,'qualification':proof,'all_checks_passed':True}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({key:value for key,value in report.items() if key!='qualification'},indent=2))

if __name__=='__main__': main()
