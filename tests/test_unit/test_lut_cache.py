"""Derived LUT cache ownership, complete invalidation and synchronization."""
import copy
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading
import time

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ColorCorrectionPipeline.core import correction
from ColorCorrectionPipeline.core.accel import build_3d_lut
from ColorCorrectionPipeline.core.lut_cache import ensure_lut, inference_identity, prime_lut_identity, persistence_lut
from ColorCorrectionPipeline.core.correction import Regressor_Model


def configured():
    model = Regressor_Model()
    model.degree = 1
    model.portable_inference = True
    model._lut_grid_size = 5
    model.lut_min_pixels = 1
    estimator = LinearRegression()
    estimator.coef_ = np.column_stack([np.zeros(3),np.eye(3)])
    estimator.intercept_ = np.zeros(3)
    model.model = estimator
    model._lut_predict_fn = model._predict_for_lut
    return model


def test_parallel_requests_build_exactly_once():
    model = configured()
    barrier = threading.Barrier(8)
    count = []
    def builder(callback,grid_size):
        count.append(1)
        time.sleep(.025)
        return build_3d_lut(callback,grid_size)
    def request(_):
        barrier.wait()
        return ensure_lut(model,builder)
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(request,range(8)))
    assert len(count) == 1
    assert model._lut_cache_info["status"] == "hit"
    assert model._lut_cache_info["entry_limit"] == 1


@pytest.mark.parametrize("mutation",["coef","intercept","grid","degree","portable","policy"])
def test_full_producer_settings_invalidate(mutation):
    model = configured()
    calls = []
    def builder(callback,grid_size):
        calls.append(1)
        return build_3d_lut(callback,grid_size)
    assert ensure_lut(model,builder)
    before = model._lut_identity
    if mutation == "coef": model.model.coef_[2,3] += .017
    elif mutation == "intercept": model.model.intercept_[2] += .03
    elif mutation == "grid": model._lut_grid_size = 7
    elif mutation == "degree":
        model.degree = 2
        model.model.coef_ = np.pad(model.model.coef_,((0,0),(0,6)))
    elif mutation == "portable": model.portable_inference = False
    else: model.fitting_version = "reproducible-v1"
    assert ensure_lut(model,builder)
    assert len(calls) == 2
    assert model._lut_identity != before
    assert model._lut.shape == (model._lut_grid_size,)*3+(3,)


def test_hash_covers_unsampled_coefficient():
    model = configured()
    model.degree = 8
    model.model.coef_ = np.zeros((3,165))
    before,_ = inference_identity(model)
    model.model.coef_[1,137] = np.nextafter(0.,1.)
    after,_ = inference_identity(model)
    assert before != after


def test_owned_immutable_cache_and_replacement_invalidates():
    model = configured()
    shared = np.ones((5,5,5,3))
    calls = []
    def builder(callback,grid_size):
        calls.append(1)
        return shared
    assert ensure_lut(model,builder)
    shared.fill(2)
    np.testing.assert_array_equal(model._lut,np.ones_like(shared))
    with pytest.raises(ValueError): model._lut[0,0,0,0] = 9
    model._lut = np.zeros_like(shared)
    assert ensure_lut(model,builder)
    assert len(calls) == 2
    np.testing.assert_array_equal(model._lut,shared)


@pytest.mark.parametrize("operation",[pickle.loads,copy.deepcopy])
def test_runtime_locks_not_serialized_and_restored_lut_exact(operation):
    model = configured()
    assert ensure_lut(model,build_3d_lut)
    expected = model._lut.copy()
    state = model.__getstate__()
    assert not any(key in state for key in ("_lut_lock","_lut_identity","_lut_table_ref","_lut_cache_info"))
    restored = operation(pickle.dumps(model)) if operation is pickle.loads else operation(model)
    np.testing.assert_array_equal(restored._lut,expected)
    assert restored._lut_predict_fn.__self__ is restored
    restored.model.intercept_[0] = .25
    assert ensure_lut(restored,build_3d_lut)
    np.testing.assert_allclose(restored._lut[...,0],expected[...,0]+.25,rtol=0,atol=1e-15)


def test_unknown_callback_rebuilds_honestly_each_call():
    model = configured()
    offset = [0.]
    model._lut_predict_fn = lambda rgb: rgb+offset[0]
    assert ensure_lut(model,build_3d_lut)
    offset[0] = .3
    assert ensure_lut(model,build_3d_lut)
    assert model._lut_identity is None
    assert "opaque" in model._lut_cache_info["identity_unavailable_reason"]
    np.testing.assert_allclose(model._lut[0,0,0],.3)


def test_failed_build_memo_reset_and_changed_producer_retry():
    model = configured()
    calls = []
    def broken(callback,grid_size):
        calls.append(1)
        raise RuntimeError("expected test failure")
    assert not ensure_lut(model,broken)
    assert not ensure_lut(model,broken)
    assert len(calls) == 1
    assert "expected test failure" in model._lut_cache_info["reason"]
    model.clear_lut_cache()
    assert not ensure_lut(model,broken)
    assert len(calls) == 2
    model.model.intercept_[0] += .1
    assert ensure_lut(model,build_3d_lut)
    assert model._lut_failed_identity is None


def test_mutation_during_build_not_published():
    model = configured()
    def changes_producer(callback,grid_size):
        table = build_3d_lut(callback,grid_size)
        model.model.coef_[0,1] = 2
        return table
    assert not ensure_lut(model,changes_producer)
    assert model._lut is None
    assert "changed during" in model._lut_cache_info["reason"]
    assert ensure_lut(model,build_3d_lut)
    assert model._lut[-1,0,0,0] == 2


def test_disable_and_reenable_discards_old_cache():
    model = configured()
    assert ensure_lut(model,build_3d_lut)
    model.use_lut = False
    assert not ensure_lut(model,build_3d_lut)
    assert model._lut is None
    model.use_lut = True
    assert ensure_lut(model,build_3d_lut)


@pytest.mark.parametrize("failure",["nan","shape","integer","bad_grid"])
def test_invalid_builder_values_not_published(failure):
    model = configured()
    if failure == "bad_grid": model._lut_grid_size = True
    def builder(callback,grid_size):
        if failure == "nan": return np.full((5,5,5,3),np.nan)
        if failure == "shape": return np.zeros((5,5,4,3))
        return np.zeros((5,5,5,3),dtype=int)
    assert not ensure_lut(model,builder)
    assert model._lut is None
    assert model._lut_cache_info["status"] == "failed"


def test_prime_restored_values_and_no_callback_execution():
    model = configured()
    model._lut = np.full((5,5,5,3),.73)
    model._lut_predict_fn = lambda rgb: pytest.fail("priming called prediction")
    prime_lut_identity(model)
    np.testing.assert_array_equal(model._lut,.73*np.ones((5,5,5,3)))
    assert not model._lut.flags.writeable


def test_legacy_static_lut_compatibility():
    model = Regressor_Model()
    model._lut_grid_size = 2
    model._lut = np.ones((2,2,2,3))
    assert ensure_lut(model,lambda *args: pytest.fail("static table should not rebuild"))
    assert model._lut_cache_info["status"] == "static-table"


def test_predict_default_direct_unchanged_when_lut_disabled(monkeypatch):
    model = configured()
    model.use_lut = False
    rgb = np.random.default_rng(2).random((5,3))
    expected = correction._predict_direct(rgb,model)
    monkeypatch.setattr(correction,"build_3d_lut",lambda *args,**kw: pytest.fail("disabled LUT"))
    np.testing.assert_array_equal(correction.predict_(rgb,model),expected)


@pytest.mark.parametrize("mutation",["coefficient","replacement"])
def test_stale_lut_is_not_persisted_or_rebuilt_on_original(mutation):
    model = configured()
    assert ensure_lut(model,build_3d_lut)
    saved_table, saved_identity = model._lut,model._lut_identity
    if mutation == "coefficient":
        model.model.coef_[0,1] = .4
    else:
        model._lut = np.full_like(model._lut,.9)
        saved_table = model._lut
    assert persistence_lut(model) is None
    restored = pickle.loads(pickle.dumps(model))
    assert restored._lut is None
    assert model._lut is saved_table
    assert model._lut_identity == saved_identity
    assert ensure_lut(restored,build_3d_lut)
    expected = .4 if mutation == "coefficient" else 1.
    assert restored._lut[-1,0,0,0] == expected


def test_current_static_and_unversioned_tables_preserved_for_persistence():
    model = configured()
    model._lut = build_3d_lut(model._lut_predict_fn,model._lut_grid_size)
    assert persistence_lut(model) is model._lut
    prime_lut_identity(model)
    assert persistence_lut(model) is model._lut
    model.model,model._lut_predict_fn = None,None
    assert persistence_lut(model) is model._lut


def test_opaque_callback_table_omitted_from_saved_state():
    model = configured()
    model._lut_predict_fn = lambda rgb: rgb+.5
    assert ensure_lut(model,build_3d_lut)
    original = model._lut
    assert persistence_lut(model) is None
    assert model.__getstate__()["_lut"] is None
    assert model._lut is original


def test_bundle_omits_stale_table_and_uses_current_producer():
    from ColorCorrectionPipeline.models import MyModels
    from ColorCorrectionPipeline.bundle import dumps_models, loads_models
    model = configured()
    assert ensure_lut(model,build_3d_lut)
    stale = model._lut
    model.model.coef_[1,2] = .35
    models = MyModels()
    models.model_cc = (model,{"mtd":"linear","degree":1},"ours")
    restored = loads_models(dumps_models(models)).model_cc[0]
    assert model._lut is stale
    assert restored._lut is None
    assert ensure_lut(restored,build_3d_lut)
    assert restored._lut[0,-1,0,1] == .35
