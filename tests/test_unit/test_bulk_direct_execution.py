"""Whole-frame pointwise execution must preserve ownership and recovery."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ColorCorrectionPipeline import AccuracyBudgetError, ColorCorrection, LutCache
from ColorCorrectionPipeline.core import execution
from ColorCorrectionPipeline.core.correction import Regressor_Model
from ColorCorrectionPipeline.portable_inference import AffinePredictor
from ColorCorrectionPipeline.stage_policies import MonotonicToneModel


def calibration(degree=2, normalized=False, wb=True, cc=True):
    owner = ColorCorrection()
    owner.models.model_gc = MonotonicToneModel(
        np.array([0., 10., 30., 50., 80., 100.]),
        np.array([0., 20., 35., 65., 90., 100.]))
    owner.models.model_wb = np.diag([.9, 1.1, .8]) if wb else None
    if cc:
        from sklearn.preprocessing import PolynomialFeatures
        count = PolynomialFeatures(degree=degree).fit(np.zeros((1, 3))).n_output_features_
        rng = np.random.default_rng(610)
        model = Regressor_Model()
        model.degree, model.portable_inference, model.use_lut = degree, True, False
        model.mtd = 'pls' if normalized else 'linear'
        model.model = AffinePredictor(rng.normal(0, .003, (3, count)), np.full(3, .4),
            center=rng.normal(0, .1, count) if normalized else None,
            scale=rng.uniform(.7, 2., count) if normalized else None,
            kind='pls' if normalized else 'linear')
        owner.models.model_cc = (model, {}, 'ours')
    return owner


def prepared(owner=None, **options):
    if not execution.accel.HAS_NUMBA:
        pytest.skip('Numba is optional')
    return (owner or calibration()).prepare_execution(dict(
        mode='direct', backend='auto', validation_samples=64, chunk_pixels=17, **options), cache=LutCache())


@pytest.mark.parametrize('degree', [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize('normalized', [False, True])
def test_bulk_affine_matches_recorded_reference(degree, normalized):
    source = np.random.default_rng(948).random((11, 23, 3))[::-1, ::-1]
    source.setflags(write=False)
    before = source.tobytes()
    with prepared(calibration(degree, normalized)) as handle:
        expected, _, _ = handle._direct(source)
        actual = handle.predict_image(source)
        np.testing.assert_allclose(actual.images['CC'], expected, rtol=0, atol=5e-12)
        assert actual.images['CC'].flags.writeable
        assert not np.shares_memory(source, actual.images['CC'])
        assert execution.bulk_direct_plan(handle._pipeline.models, backend='auto', stop='CC')
        info = actual.diagnostics['backend']
        assert info['pointwise_execution'] == 'owned-buffer-bulk-direct'
        assert info['CC']['calls_by_backend'] == {'cpu-numba': 1}
        assert info['CC']['domain_policy'] == 'reject'
        assert info['CC']['polynomial_degree'] == degree
        assert info['CC']['feature_policy'] == 'portable/reproducible CPU float64'
        assert source.tobytes() == before


@pytest.mark.parametrize('wb,cc', [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize('stop', ['GC', 'WB', 'CC'])
def test_bulk_intermediates_and_disabled_stages_are_independent(wb, cc, stop):
    source = np.random.default_rng(932).random((9, 7, 3))
    source[0, 0] = 0.
    source[-1, -1] = 1.
    with prepared(calibration(wb=wb, cc=cc)) as handle:
        names = ('GC', 'WB', 'CC')[:('GC', 'WB', 'CC').index(stop) + 1]
        _, reference, _ = handle._direct(source, retain=names, stop=stop)
        result = handle.predict_image(source, stages=names)
        for name in names:
            np.testing.assert_allclose(result.images[name], reference[name], rtol=0, atol=5e-12)
            assert result.images[name].flags.writeable
        for index, name in enumerate(names):
            for other in names[index + 1:]:
                assert not np.shares_memory(result.images[name], result.images[other])
        assert result.diagnostics['backend']['GC_execution']['calls_by_backend'] == {'cpu-numba': 1}
        before = {key: value.copy() for key, value in result.images.items()}
        result.images[names[-1]][:] = .2
        for name in names[:-1]:
            np.testing.assert_array_equal(result.images[name], before[name])


@pytest.mark.parametrize('failure', ['raise', 'nonfinite'])
@pytest.mark.parametrize('strict', [False, True])
def test_partially_written_bulk_cc_never_contaminates_fallback(monkeypatch, failure, strict):
    source = np.random.default_rng(119).random((7, 11, 3))
    before = source.copy()
    with prepared(allow_fallback=not strict) as handle:
        _, reference, _ = handle._direct(source, retain=('GC', 'WB', 'CC'))
        def broken(flat, *args):
            flat[:len(flat)//2] = .01
            if failure == 'raise':
                raise RuntimeError('CC partially written')
            flat[-1, -1] = np.inf
        monkeypatch.setattr(execution, '_affine_inplace', broken)
        if strict:
            with pytest.raises((RuntimeError, FloatingPointError)):
                handle.predict_image(source, stages=('GC', 'WB', 'CC'))
        else:
            result = handle.predict_image(source, stages=('GC', 'WB', 'CC'))
            for name in reference:
                np.testing.assert_array_equal(result.images[name], reference[name])
            info = result.diagnostics['backend']['GC_execution']
            assert info['actual_backend'] == 'reference-direct'
            assert info['runtime_fallback_occurred']
            assert info['fallback_reasons']
            assert result.diagnostics['backend']['pointwise_execution'] == 'reference-after-bulk-runtime-failure'
        np.testing.assert_array_equal(source, before)


def test_bulk_failure_during_qualification_pins_reference(monkeypatch):
    if not execution.accel.HAS_NUMBA:
        pytest.skip('Numba is optional')
    real = execution._affine_inplace
    def broken(flat, *args):
        flat[:] = .9
        raise RuntimeError('qualification failure')
    monkeypatch.setattr(execution, '_affine_inplace', broken)
    with prepared() as handle:
        assert handle._direct_backend is None
        monkeypatch.setattr(execution, '_affine_inplace', real)
        source = np.random.default_rng(31).random((11, 3, 3))
        reference, _, _ = handle._direct(source)
        np.testing.assert_array_equal(handle.predict_image(source).images['CC'], reference)
    monkeypatch.setattr(execution, '_affine_inplace', broken)
    with pytest.raises(RuntimeError, match='qualification failure'):
        prepared(allow_fallback=False)


def test_bulk_qualification_does_not_accept_perturbed_output(monkeypatch):
    if not execution.accel.HAS_NUMBA:
        pytest.skip('Numba is optional')
    real = execution._affine_inplace
    def perturbed(flat, *args):
        real(flat, *args)
        flat += 1e-7
    monkeypatch.setattr(execution, '_affine_inplace', perturbed)
    with prepared() as handle:
        assert handle._direct_backend is None
        assert handle.preparation['qualification']['direct_backend_error']['max_rgb_error'] > 5e-12
    with pytest.raises(AccuracyBudgetError):
        prepared(allow_fallback=False)


def test_bulk_reload_snapshot_and_parallel_calls(tmp_path):
    owner = calibration(normalized=True)
    path = tmp_path/'bulk.ccpb'
    owner.save_bundle(path)
    saved = path.read_bytes()
    resumed = ColorCorrection().load_bundle(path)
    source = np.random.default_rng(83).random((13, 17, 3))
    with prepared(owner) as before, prepared(resumed) as after:
        expected = before.predict_image(source).images['CC']
        np.testing.assert_array_equal(after.predict_image(source).images['CC'], expected)
        owner.models.model_wb[:] = 0.
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda _: before.predict_image(source).images['CC'], range(4)))
        for output in outputs:
            np.testing.assert_array_equal(output, expected)
        assert before.calibration_id == after.calibration_id
    assert path.read_bytes() == saved


@pytest.mark.parametrize('backend', ['numpy', 'auto'])
def test_bulk_plan_rejects_unavailable_or_reference_backend(monkeypatch, backend):
    monkeypatch.setattr(execution.accel, 'HAS_NUMBA', False)
    assert execution.bulk_direct_plan(calibration().models, backend=backend, stop='CC') is None


def test_scalar_rgb_reductions_still_reject_nonfinite_and_preserve_clip():
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='finite'):
            execution._rgb(np.array([[0., value, 1.]]), 'clip')
    source = np.array([[-2., .2, 3.]], dtype=np.longdouble)
    actual, clipped = execution._rgb(source, 'clip')
    np.testing.assert_array_equal(actual, [[0., .2, 1.]])
    assert clipped and actual.dtype == np.float64
