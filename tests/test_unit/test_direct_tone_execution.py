"""Direct tone acceleration preserves scientific equations and ownership."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, LutCache, AccuracyBudgetError
from ColorCorrectionPipeline.core import execution
from ColorCorrectionPipeline.modern import apply_tone
from ColorCorrectionPipeline.stage_policies import MonotonicToneModel


def tone(kind=0):
    if kind == 1:
        return MonotonicToneModel(np.arange(0.,101.,20.), np.arange(0.,101.,20.))
    if kind == 2:
        return MonotonicToneModel(np.array([0.,1.,2.,50.,100.]),np.array([0.,20.,20.,95.,100.]))
    return MonotonicToneModel(np.array([0.,10.,30.,50.,80.,100.]),np.array([0.,20.,35.,65.,90.,100.]))


def require_numba():
    if not execution.accel.HAS_NUMBA:
        pytest.skip('Numba is optional')


@pytest.mark.parametrize('kind',[0,1,2])
@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(17,13,3),(201,3)])
def test_direct_tone_matches_reference_without_mutation(kind,backend,dtype,shape):
    if backend=='numba': require_numba()
    source=np.random.default_rng(906).random(shape).astype(dtype)[::-1]
    source.setflags(write=False)
    before=source.tobytes()
    expected=apply_tone(source.astype(np.float64).reshape(1,-1,3),tone(kind)).reshape(shape)
    actual,info=execution.evaluate_tone(source,tone(kind),backend=backend,allow_fallback=False)
    if backend=='numpy': np.testing.assert_array_equal(actual,expected)
    else: np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
    assert info['actual_backend']=='cpu-'+backend and info['precision']=='float64'
    assert info['LUT'] is False
    assert actual.flags.writeable and not np.shares_memory(source,actual)
    assert source.tobytes()==before


@pytest.mark.parametrize('kind',[0,1,2])
def test_pchip_matches_at_knots_adjacent_values_and_endpoints(kind):
    require_numba()
    model=tone(kind)
    knots,coefficients=execution._tone_parameters(model)
    values=np.r_[[-1.,0.,100.,101.],knots,np.nextafter(knots,-np.inf),np.nextafter(knots,np.inf)]
    actual=np.array([execution._tone_lightness(value,knots,coefficients) for value in values])
    np.testing.assert_allclose(actual,model.predict(values),rtol=1e-15,atol=3e-14)
    np.testing.assert_array_equal(actual[:4],[0.,0.,100.,100.])


def test_transfer_and_lab_threshold_neighborhoods():
    require_numba()
    from ColorCorrectionPipeline.colorimetry import xyz_to_rgb,white_xyz,D65
    eps=216./24389.
    levels=np.array([0.,np.nextafter(0.,1.),.04045,np.nextafter(.04045,0),np.nextafter(.04045,1),1.])
    values=np.stack(np.meshgrid(levels,levels,levels,indexing='ij'),axis=-1).reshape(-1,3)
    for threshold in (eps,np.nextafter(eps,0),np.nextafter(eps,1)):
        neutrals=xyz_to_rgb(np.tile(white_xyz(D65)*threshold,(7,1)),policy='clip')
        values=np.vstack((values,neutrals))
    actual,_=execution.evaluate_tone(values,tone(),backend='numba',allow_fallback=False)
    np.testing.assert_allclose(actual,apply_tone(values.reshape(1,-1,3),tone()).reshape(-1,3),rtol=0,atol=2e-12)


@pytest.mark.parametrize('value',[np.zeros(3),np.empty((0,3)),np.ones((1,4)),np.ones((1,1,1,3)),
    np.array([[np.nan,0,0]]),np.array([[np.inf,0,0]]),np.array([[-1.,0,0]]),np.array([[1.01,0,0]]),
    np.array([[0,1,0]],dtype=object),np.ones((1,3),dtype=complex)])
def test_invalid_tone_inputs_refused(value):
    with pytest.raises(ValueError): execution.evaluate_tone(value,tone())


def test_unavailable_backend_and_runtime_failure_are_honest(monkeypatch):
    rgb=np.array([[.3,.5,.8]])
    expected=apply_tone(rgb.reshape(1,-1,3),tone()).reshape(rgb.shape)
    monkeypatch.setattr(execution.accel,'HAS_NUMBA',False)
    actual,info=execution.evaluate_tone(rgb,tone(),backend='numba')
    np.testing.assert_array_equal(actual,expected)
    assert info['actual_backend']=='cpu-numpy' and 'unavailable' in info['fallback_reason']
    with pytest.raises(RuntimeError,match='unavailable'):
        execution.evaluate_tone(rgb,tone(),backend='numba',allow_fallback=False)
    monkeypatch.setattr(execution.accel,'HAS_NUMBA',True)
    def broken(*args): raise RuntimeError('tone backend failed')
    monkeypatch.setattr(execution,'_fused_tone',broken,raising=False)
    actual,info=execution.evaluate_tone(rgb,tone(),backend='numba')
    np.testing.assert_array_equal(actual,expected)
    assert 'tone backend failed' in info['fallback_reason']
    with pytest.raises(RuntimeError,match='tone backend failed'):
        execution.evaluate_tone(rgb,tone(),backend='numba',allow_fallback=False)


@pytest.mark.parametrize('override',['subclass','instance','class','narrow','steep'])
def test_overrides_and_ill_conditioned_models_keep_reference(monkeypatch,override):
    require_numba()
    model=tone()
    if override=='subclass':
        class Custom(MonotonicToneModel):
            def predict(self,values): return np.asarray(values)*.7
        model=Custom(model.measured_knots,model.target_knots)
    elif override=='instance': object.__setattr__(model,'predict',lambda values:np.asarray(values)*.7)
    elif override=='class': monkeypatch.setattr(MonotonicToneModel,'predict',lambda self,values:np.asarray(values)*.7)
    elif override=='narrow': model=MonotonicToneModel(np.array([0.,20.,20.00001,60.,100.]),np.array([0.,20.,21.,60.,100.]))
    else: model=MonotonicToneModel(np.array([0.,.1,20.,60.,100.]),np.array([0.,20.,21.,60.,100.]))
    rgb=np.random.default_rng(3).random((101,3))
    monkeypatch.setattr(execution,'_fused_tone',lambda *args:pytest.fail('unsupported model must not fuse'))
    actual,info=execution.evaluate_tone(rgb,model,backend='auto',allow_fallback=False)
    np.testing.assert_array_equal(actual,apply_tone(rgb.reshape(1,-1,3),model).reshape(rgb.shape))
    assert 'ill-conditioned' in info['fallback_reason'] and info['actual_backend']=='cpu-numpy'
    with pytest.raises(ValueError,match='no well-conditioned'):
        execution.evaluate_tone(rgb,model,backend='numba',allow_fallback=False)


def calibration():
    owner=ColorCorrection()
    owner.models.model_gc=tone()
    owner.models.model_wb=np.array([[.8,.01,.02],[.02,.9,.03],[.01,.02,.85]])
    return owner


def options(**values):
    return dict(mode='direct',backend='auto',validation_samples=64,chunk_pixels=17,**values)


def test_intermediates_and_stop_preserve_stage_contracts():
    require_numba()
    rgb=np.random.default_rng(108).random((7,11,3))
    with calibration().prepare_execution(options(),cache=LutCache()) as prepared:
        expected,stages,_=prepared._direct(rgb,backend=None,retain=('GC','WB','CC'))
        actual=prepared.predict_image(rgb,stages='all')
        for stage in ('GC','WB','CC'):
            np.testing.assert_allclose(actual.images[stage],stages[stage],rtol=0,atol=2e-12)
            assert actual.images[stage].flags.writeable
        assert actual.diagnostics['backend']['GC']=='explicit-D65-neutral-PCHIP'
        assert actual.diagnostics['backend']['GC_execution']['actual_backend']=='cpu-numba'
        stopped=prepared.predict_image(rgb,stages=('GC',))
        np.testing.assert_allclose(stopped.images['GC'],stages['GC'],rtol=0,atol=2e-12)
        assert stopped.diagnostics['backend']['WB']=='disabled'
        np.testing.assert_array_equal(actual.images['FFC'],rgb)


def test_loose_lut_budget_never_admits_material_direct_roundoff(monkeypatch):
    require_numba()
    real=execution._fused_tone
    def perturbed(*args): return np.clip(real(*args)+1e-7,0,1)
    monkeypatch.setattr(execution,'_fused_tone',perturbed)
    owner=calibration()
    with owner.prepare_execution(options(),cache=LutCache()) as prepared:
        assert prepared._direct_backend is None
        proof=prepared.preparation['qualification']
        assert proof['direct_backend_error']['max_rgb_error']>5e-12
        assert proof['direct_roundoff_limits']['max_rgb_error']==5e-12
        rgb=np.random.default_rng(66).random((3,4,3))
        expected,_,_=prepared._direct(rgb)
        np.testing.assert_array_equal(prepared.predict_image(rgb).images['CC'],expected)
    with pytest.raises(AccuracyBudgetError,match='direct backend'):
        owner.prepare_execution(options(allow_fallback=False),cache=LutCache())


def test_zero_budget_restores_exact_reference_execution():
    require_numba()
    with calibration().prepare_execution(options(max_rgb_error=0.,max_delta_e00=0.),cache=LutCache()) as prepared:
        assert prepared._direct_backend is None
        assert prepared.preparation['qualification']['direct_backend_error']['max_rgb_error']>0
        rgb=np.random.default_rng(731).random((4,7,3))
        expected,_,_=prepared._direct(rgb)
        np.testing.assert_array_equal(prepared.predict_image(rgb).images['CC'],expected)


def test_failed_qualification_backend_stays_pinned_after_runtime_recovers(monkeypatch):
    require_numba()
    real=execution._fused_tone
    def broken(*args): raise RuntimeError('temporary compiler failure')
    monkeypatch.setattr(execution,'_fused_tone',broken)
    with calibration().prepare_execution(options(max_rgb_error=0.,max_delta_e00=0.),cache=LutCache()) as prepared:
        assert prepared._direct_backend is None
        assert 'runtime failure' in prepared.preparation['qualification']['direct_backend_fallback']
        monkeypatch.setattr(execution,'_fused_tone',real)
        rgb=np.random.default_rng(948).random((9,3,3))
        expected,_,_=prepared._direct(rgb)
        np.testing.assert_array_equal(prepared.predict_image(rgb).images['CC'],expected)


def test_mixed_frame_backend_report_retains_earlier_failure(monkeypatch):
    require_numba()
    with calibration().prepare_execution(options(),cache=LutCache()) as prepared:
        real=execution._fused_tone
        calls=[]
        def once(*args):
            calls.append(1)
            if len(calls)==1: raise RuntimeError('first chunk failed')
            return real(*args)
        monkeypatch.setattr(execution,'_fused_tone',once)
        output=prepared.predict_image(np.random.default_rng(733).random((9,7,3)))
        info=output.diagnostics['backend']['GC_execution']
        assert info['backends_used']==('cpu-numba','cpu-numpy')
        assert info['runtime_fallback_occurred'] is True
        assert any('first chunk failed' in reason for reason in info['fallback_reasons'])


def test_cc_runtime_failure_is_also_pinned_to_verified_reference(monkeypatch):
    require_numba()
    from ColorCorrectionPipeline.core.correction import Regressor_Model
    from ColorCorrectionPipeline.portable_inference import AffinePredictor
    owner=ColorCorrection()
    model=Regressor_Model()
    model.degree,model.portable_inference,model.use_lut=1,True,False
    model.model=AffinePredictor(np.eye(3,4,k=1)*.8,np.zeros(3))
    owner.models.model_cc=(model,{},'ours')
    monkeypatch.setattr(execution,'_fused_affine',lambda *args:(_ for _ in ()).throw(RuntimeError('CC compiler failed')))
    with owner.prepare_execution(options(),cache=LutCache()) as prepared:
        assert prepared._direct_backend is None
        assert 'runtime failure' in prepared.preparation['qualification']['direct_backend_fallback']


@pytest.mark.parametrize('gains',[[.2,.8,1.],[1.1,1.5,.9],[2.,4.,.5]])
def test_positive_diagonal_wb_scale_is_bitwise_matrix_equivalent(gains):
    require_numba()
    owner=calibration()
    owner.models.model_wb=np.diag(gains)
    rgb=np.random.default_rng(992).random((19,23,3))
    rgb[0,0]=0.; rgb[-1,-1]=1.
    with owner.prepare_execution(options(),cache=LutCache()) as prepared:
        output=prepared.predict_image(rgb,stages=('GC','WB'))
        expected=np.clip(output.images['GC']@owner.models.model_wb,0,1)
        assert output.images['WB'].tobytes()==expected.tobytes()
        assert output.diagnostics['backend']['WB_execution']=='numpy-channel-gains'


def test_concurrent_and_saved_calibration_predictions(tmp_path):
    require_numba()
    owner=calibration()
    bundle=tmp_path/'tone.ccpb'
    first=owner.save_bundle(bundle)
    resumed=ColorCorrection().load_bundle(bundle)
    rgb=np.random.default_rng(808).random((13,9,3))
    with owner.prepare_execution(options(),cache=LutCache()) as before, resumed.prepare_execution(options(),cache=LutCache()) as after:
        expected=before.predict_image(rgb).images['CC']
        np.testing.assert_array_equal(after.predict_image(rgb).images['CC'],expected)
        with ThreadPoolExecutor(max_workers=2) as pool:
            copies=list(pool.map(lambda _:before.predict_image(rgb).images['CC'],range(4)))
        for output in copies: np.testing.assert_array_equal(output,expected)
        assert before.calibration_id==after.calibration_id
        owner.models.model_gc=tone(2)
        np.testing.assert_array_equal(before.predict_image(rgb).images['CC'],expected)
    assert bundle.read_bytes() and first


def test_sensitive_downstream_pipeline_retains_exact_reference_gc(monkeypatch):
    require_numba()
    owner=calibration()
    owner.models.model_wb=np.eye(3)*1000
    monkeypatch.setattr(execution,'_fused_tone',lambda *args:pytest.fail('sensitive downstream must not fuse GC'))
    with owner.prepare_execution(options(),cache=LutCache()) as prepared:
        rgb=np.random.default_rng(19).random((2,5,3))
        _,expected,_=prepared._direct(rgb,retain=('GC','CC'))
        actual=prepared.predict_image(rgb,stages=('GC','CC'))
        for name in expected: np.testing.assert_array_equal(actual.images[name],expected[name])
        assert 'downstream sensitivity' in actual.diagnostics['backend']['GC_execution']['fallback_reason']


def test_downstream_gain_guard_covers_polynomial_normalization_and_unknown_models():
    from ColorCorrectionPipeline.core.correction import Regressor_Model
    from ColorCorrectionPipeline.portable_inference import AffinePredictor,MLPPredictor
    model=Regressor_Model()
    model.degree,model.portable_inference=1,True
    model.model=AffinePredictor(np.array([[0,1,0,0],[0,0,2,0],[0,0,0,3.]]),np.zeros(3))
    assert execution._tone_downstream_bound(np.eye(3)*.5,(model,{},'ours'))==1.5
    model.model=AffinePredictor(np.ones((3,4)),np.zeros(3),scale=np.ones(4)*1e-8)
    assert execution._tone_downstream_bound(None,(model,{},'ours')) is None
    model.model=MLPPredictor([np.zeros((4,3))],[np.zeros(3)],activation='relu',output_activation='identity')
    assert execution._tone_downstream_bound(None,(model,{},'ours')) is None
    assert execution._tone_downstream_bound(None,(np.eye(3),{},'conv')) is None


def test_constant_cc_cannot_hide_gc_error_during_qualification(monkeypatch):
    require_numba()
    from ColorCorrectionPipeline.core.correction import Regressor_Model
    from ColorCorrectionPipeline.portable_inference import AffinePredictor
    owner=calibration()
    model=Regressor_Model()
    model.degree,model.portable_inference,model.use_lut=1,True,False
    model.model=AffinePredictor(np.zeros((3,4)),np.full(3,.5))
    owner.models.model_cc=(model,{},'ours')
    real=execution._fused_tone
    monkeypatch.setattr(execution,'_fused_tone',lambda *args:np.clip(real(*args)+1e-7,0,1))
    with owner.prepare_execution(options(),cache=LutCache()) as prepared:
        proof=prepared.preparation['qualification']
        assert proof['direct_backend_error']['max_rgb_error']==0
        assert proof['direct_GC_error']['qualification_max_encoded_rgb_error']>5e-12
        assert prepared._direct_backend is None


@pytest.mark.parametrize('linear',[False,True])
@pytest.mark.parametrize('ffc',[False,True])
def test_explicit_encoding_and_spatial_ffc_keep_stage_order(monkeypatch,linear,ffc):
    from ColorCorrectionPipeline import ImageData
    from ColorCorrectionPipeline.colorimetry import srgb_to_linear
    from ColorCorrectionPipeline.core.accel import apply_ffc_float
    from tests.test_unit.test_modern_pipeline import chart,config
    chart_image,reference_profile=chart(monkeypatch)
    owner=ColorCorrection()
    space='linear-srgb' if linear else 'srgb'
    cfg=config(reference_profile=reference_profile,do_gc=True,do_wb=True,do_cc=False,
        color_semantics={'version':'explicit-v1','input_encoding':space,'output_encoding':space})
    assert not owner.run(ImageData.from_array(chart_image),config=cfg)[2],owner.last_run
    rgb=np.random.default_rng(402).random(chart_image.shape)*.8
    source=srgb_to_linear(rgb) if linear else rgb
    if ffc:
        owner.models.model_ffc=np.full(chart_image.shape[:2],1.1)
        owner.models.metadata['enabled_stages']['do_ffc']=True
        owner.models.metadata['numerical_report']['FFC']={'gain_limits':[.5,2.], 'source_shape':list(chart_image.shape[:2])}
    with owner.prepare_execution({'mode':'direct','backend':'auto','validation_samples':64},cache=LutCache()) as prepared:
        _,working=prepared._pipeline._prepare_input(source,prepared.semantics)
        if ffc: working=apply_ffc_float(working,prepared._pipeline.models.model_ffc)
        _,reference,_=prepared._direct(working,retain=('GC','WB','CC'))
        reference['FFC']=working
        reference=prepared._pipeline._convert_outputs(reference)
        actual=prepared.predict_image(source,stages='all')
        for name in reference:
            np.testing.assert_allclose(actual.images[name],reference[name],rtol=0,atol=2e-12)
