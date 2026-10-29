"""Explicit execution numerical/backend contracts, including adversarial inputs."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ColorCorrectionPipeline.core import execution
from ColorCorrectionPipeline.core.correction import Regressor_Model, _predict_direct
from ColorCorrectionPipeline.portable_inference import AffinePredictor


def affine(degree=3, normalized=False):
    rng = np.random.default_rng(157)
    count = PolynomialFeatures(degree=degree).fit(np.zeros((1,3))).n_output_features_
    model = Regressor_Model()
    model.degree = degree
    model.portable_inference = True
    model.model = AffinePredictor(rng.normal(size=(3,count)),rng.normal(size=3),
        center=rng.normal(size=count) if normalized else None,
        scale=rng.uniform(.1,3,size=count) if normalized else None,
        kind="pls" if normalized else "linear")
    return model


@pytest.mark.parametrize("degree",[0,1,2,3,5,8])
@pytest.mark.parametrize("normalized",[False,True])
@pytest.mark.parametrize("backend",["numpy","numba"])
def test_affine_agrees_with_reference(degree,normalized,backend):
    if backend == "numba" and not execution.accel.HAS_NUMBA:
        pytest.skip("Numba optional")
    model = affine(degree,normalized)
    rgb = np.random.default_rng(44).random((8,9,3)).astype(np.float32)[::2,::-1]
    expected = _predict_direct(rgb,model)
    actual, info = execution.evaluate_direct(rgb,model,backend=backend,allow_fallback=False)
    np.testing.assert_allclose(actual,expected,rtol=4e-13,atol=4e-13)
    assert info["actual_backend"] == "cpu-"+backend
    assert info["precision"] == "float64"


def test_known_linear_reproducible_adapter():
    portable = affine(2)
    estimator = LinearRegression()
    estimator.coef_,estimator.intercept_ = portable.model.coefficients,portable.model.intercept
    portable.model,portable.portable_inference,portable.fitting_version = estimator,False,"reproducible-v1"
    rgb = np.array([[.2,.5,.8],[0,1,0.]])
    actual,info = execution.evaluate_direct(rgb,portable,backend="numpy")
    np.testing.assert_array_equal(actual,_predict_direct(rgb,portable))
    assert info["feature_count"] == 10


class BatchDependent:
    def __init__(self):
        self.calls = 0
    def predict(self,features):
        self.calls += 1
        return np.broadcast_to(features.mean(axis=0)[1:4],(len(features),3)).copy()


def test_unsupported_model_preserves_whole_batch_reference():
    model = affine(1)
    model.model = BatchDependent()
    rgb = np.random.default_rng(2).random((65537,3))
    actual,info = execution.evaluate_direct(rgb,model,backend="numpy")
    assert model.model.calls == 1
    np.testing.assert_array_equal(actual,np.broadcast_to(rgb.mean(axis=0),rgb.shape))
    assert info["actual_backend"] == "reference-direct"
    assert info["fallback_reason"]
    for backend in ("numpy","numba"):
        with pytest.raises(ValueError,match="no fused"):
            execution.evaluate_direct(rgb,model,backend=backend,allow_fallback=False)
    _, info = execution.evaluate_direct(rgb,model,backend="auto",allow_fallback=False)
    assert info["actual_backend"] == "reference-direct"


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_legacy_policy_preserves_exact_reference(dtype):
    model = affine(3)
    model.portable_inference = False
    rgb = np.random.default_rng(25).random((41,3)).astype(dtype)
    expected = _predict_direct(rgb,model)
    actual,info = execution.evaluate_direct(rgb,model,backend="auto")
    np.testing.assert_array_equal(actual,expected)
    assert info["actual_backend"] == "reference-direct"


def test_degree_outside_fused_bound_uses_reference():
    model = affine(9)
    rgb = np.array([[.4,.7,.2]])
    actual,info = execution.evaluate_direct(rgb,model,backend="numpy")
    np.testing.assert_array_equal(actual,_predict_direct(rgb,model))
    assert info["actual_backend"] == "reference-direct"


@pytest.mark.parametrize("operation",["direct","lut"])
def test_explicit_numba_absence_and_runtime_fallback(monkeypatch,operation):
    model = affine(1)
    rgb = np.array([[.2,.8,.4]])
    table = asymmetric_table()
    evaluate = (lambda **kw: execution.evaluate_direct(rgb,model,**kw)) if operation == "direct" else (lambda **kw: execution.apply_lut(rgb,table,**kw))
    monkeypatch.setattr(execution.accel,"HAS_NUMBA",False)
    _,info = evaluate(backend="numba")
    assert info["actual_backend"] == "cpu-numpy"
    assert "unavailable" in info["fallback_reason"]
    with pytest.raises(RuntimeError,match="unavailable"):
        evaluate(backend="numba",allow_fallback=False)
    monkeypatch.setattr(execution.accel,"HAS_NUMBA",True)
    def broken(*args):
        raise RuntimeError("backend exploded")
    monkeypatch.setattr(execution,"_fused_affine",broken,raising=False)
    monkeypatch.setattr(execution.accel,"_trilinear_interp",broken)
    _,info = evaluate(backend="numba")
    assert info["actual_backend"] == "cpu-numpy"
    assert "backend exploded" in info["fallback_reason"]
    with pytest.raises(RuntimeError,match="exploded"):
        evaluate(backend="numba",allow_fallback=False)


def asymmetric_table(size=7):
    rng = np.random.default_rng(7)
    return rng.normal(size=(size,size,size,3))


def test_lut_backend_consistency_at_boundaries_and_asymmetric_cells():
    rgb = np.vstack([np.random.default_rng(9).random((601,3)),np.zeros((1,3)),np.ones((1,3)),
                     [[0,1,.3],[1,.5,0],[.5,0,1]]])
    table = asymmetric_table()
    reference,_ = execution.apply_lut(rgb,table,backend="numpy")
    actual,info = execution.apply_lut(rgb,table,backend="numba")
    np.testing.assert_allclose(actual,reference,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(actual[601],table[0,0,0])
    np.testing.assert_array_equal(actual[602],table[-1,-1,-1])
    assert info["method"] == "trilinear"


@pytest.mark.parametrize("operation",["direct","lut"])
@pytest.mark.parametrize("backend",["numpy","numba"])
def test_domain_reject_and_explicit_clip(operation,backend):
    rgb = np.array([[[-.3,.2,1.4],[1,0,.1]]])
    model = affine(2)
    table = asymmetric_table()
    evaluate = (lambda data,**kw: execution.evaluate_direct(data,model,**kw)) if operation == "direct" else (lambda data,**kw: execution.apply_lut(data,table,**kw))
    with pytest.raises(ValueError,match="outside"):
        evaluate(rgb,backend=backend)
    actual,info = evaluate(rgb,backend=backend,domain="clip")
    expected,_ = evaluate(np.clip(rgb,0,1),backend=backend)
    np.testing.assert_array_equal(actual,expected)
    assert info["input_was_clipped"]
    assert info["domain_policy"] == "clip"


@pytest.mark.parametrize("rgb",[np.empty((0,3)),np.zeros(3),np.zeros((2,4)),np.zeros((1,1,1,3)),
    [[np.nan,0,1]],[[0,np.inf,1]],np.array([[0,1,0]],dtype=object),np.ones((1,3),dtype=complex)])
def test_invalid_rgb_rejected(rgb):
    with pytest.raises(ValueError):
        execution.evaluate_direct(rgb,affine())
    with pytest.raises(ValueError):
        execution.apply_lut(rgb,asymmetric_table())


@pytest.mark.parametrize("table",[np.ones((2,3,2,3)),np.ones((1,1,1,3)),np.ones((2,2,2,4)),np.full((2,2,2,3),np.nan),
                                 np.zeros((2,2,2,3),dtype=int)])
def test_invalid_lut_rejected(table):
    with pytest.raises(ValueError):
        execution.apply_lut(np.ones((1,3)),table)


def test_parameter_and_backend_validation():
    model = affine(1)
    model.model.scale = np.zeros(4)
    with pytest.raises(ValueError,match="positive"):
        execution.evaluate_direct(np.zeros((1,3)),model)
    with pytest.raises(ValueError,match="backend"):
        execution.evaluate_direct(np.zeros((1,3)),affine(),backend="gpu")
    with pytest.raises(TypeError,match="boolean"):
        execution.evaluate_direct(np.zeros((1,3)),affine(),allow_fallback=1)
    with pytest.raises(ValueError,match="domain"):
        execution.apply_lut(np.zeros((1,3)),asymmetric_table(),domain="extrapolate")


def test_complex_affine_is_not_silently_truncated_to_real():
    model = affine(1)
    model.model.coefficients = model.model.coefficients.astype(complex)+.1j
    with pytest.raises(ValueError,match="real numeric"):
        execution.evaluate_direct(np.zeros((1,3)),model)
