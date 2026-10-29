"""Adversarial qualification of opt-in neutral and observation policies."""

import copy
import pickle

import numpy as np
import pytest

from ColorCorrectionPipeline.stage_policies import (
    StagePolicy, MonotonicToneModel, fit_monotonic_tone, fit_neutral_wb,
    saturation_selection,
)


CELLS = tuple("cell-%d" % i for i in range(8))
NEUTRALS = CELLS[2:]
POLICY = StagePolicy(version="modern-v1")


def tone_data():
    return np.array([75, 25, 12, 25, 40, 55, 70, 85.]), np.array([10, 90, 15, 30, 45, 60, 75, 90.])


def test_monotonic_model_preserves_endpoints_dense_ramp_and_uses_only_neutrals():
    x, y = tone_data()
    model, report = fit_monotonic_tone(x, y, CELLS, NEUTRALS, policy=POLICY)
    ramp = np.linspace(-10, 110, 12001)
    prediction = model.predict(ramp)
    assert np.isfinite(prediction).all()
    assert np.diff(prediction).min() >= 0
    assert prediction[0] == 0 and prediction[-1] == 100
    np.testing.assert_allclose(model.predict(x[2:]), y[2:], atol=1e-12)
    assert report["endpoint_assumptions"] == [[0, 0], [100, 100]]
    x[:2] = [99, 1]
    y[:2] = [1000, np.nan]  # colored targets are not GC observations
    model_b, _ = fit_monotonic_tone(x, y, CELLS, NEUTRALS, policy=POLICY)
    np.testing.assert_array_equal(model_b.predict(ramp), prediction)


@pytest.mark.parametrize("x", [[10, 10, 10], [10, 12, 14], [10, 90], [0, 100, 50]])
def test_endpoint_anchors_cannot_hide_insufficient_measured_neutrals(x):
    cells = tuple("n%d" % i for i in range(len(x)))
    with pytest.raises(ValueError, match="measured|usable"):
        fit_monotonic_tone(x, np.linspace(10, 90, len(x)), cells, cells, policy=POLICY)


def test_duplicate_conflicting_targets_and_reversed_neutrals_fail():
    for x, y in (([10, 10, 40, 70], [10, 30, 45, 80]), ([10, 40, 70], [10, 80, 45])):
        cells = tuple("n%d" % i for i in range(len(x)))
        with pytest.raises(ValueError, match="contradict"):
            fit_monotonic_tone(x, y, cells, cells, policy=POLICY)


def test_agreeing_duplicates_are_aggregated_and_counted_honestly():
    cells = ("a", "b", "c", "d")
    model, report = fit_monotonic_tone([10, 10, 40, 70], [10, 10.5, 45, 80], cells, cells, policy=POLICY)
    assert report["measured_neutral_count"] == 4
    assert report["distinct_measured_lightness_count"] == 3
    assert model.predict(10) == 10.25


def test_clipped_original_neutral_does_not_train_tone_and_is_reported():
    x, y = tone_data()
    clipped = np.zeros((8, 3))
    clipped[4, 1] = 1 / 81
    model, report = fit_monotonic_tone(x, y, CELLS, NEUTRALS, clipped_fractions=clipped, policy=POLICY)
    assert CELLS[4] in report["excluded_cell_ids"]
    x[4], y[4] = 99, 0
    again, _ = fit_monotonic_tone(x, y, CELLS, NEUTRALS, clipped_fractions=clipped, policy=POLICY)
    np.testing.assert_array_equal(model.predict(np.arange(101)), again.predict(np.arange(101)))
    assert report["saturation"]["estimated_values"] is False


@pytest.mark.parametrize("operation", [lambda x: x, pickle.loads, copy.deepcopy])
def test_tone_knots_are_owned_and_immutable_after_copy_and_pickle(operation):
    x, y = tone_data()
    model, _ = fit_monotonic_tone(x, y, CELLS, NEUTRALS, policy=POLICY)
    transformed = operation(pickle.dumps(model)) if operation is pickle.loads else operation(model)
    x[:] = 0
    for array in (transformed.measured_knots, transformed.target_knots):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_allclose(transformed.predict([0, 100]), [0, 100])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_tone_application_fails(bad):
    model = MonotonicToneModel([0, 10, 40, 70, 100], [0, 15, 45, 80, 100])
    with pytest.raises(ValueError, match="finite"):
        model.predict([bad])


def test_wb_robust_neutral_ratio_headroom_and_colored_independence():
    target = np.tile([.2, .3, .4], (8, 1))
    measured = target / [2, 1.5, .75]
    measured[2] = [.15, .2, .5]  # one outlying neutral does not set median gains
    diagonal, report = fit_neutral_wb(measured, target, CELLS, NEUTRALS, policy=POLICY)
    np.testing.assert_allclose(np.diag(diagonal), [.95 * 2, .95 * 1.5, .95 * .75])
    measured[:2] = 0
    target[:2] = 1
    again, _ = fit_neutral_wb(measured, target, CELLS, NEUTRALS, policy=POLICY)
    np.testing.assert_array_equal(again, diagonal)
    assert report["headroom"] == .95


def test_nonfinite_measured_neutrals_are_excluded_without_invented_measurements():
    x, y = tone_data()
    x[2] = np.nan
    _, report = fit_monotonic_tone(x, y, CELLS, NEUTRALS, policy=POLICY)
    assert report["excluded_cell_ids"] == [CELLS[2]]
    assert report["measured_neutral_count"] == 5


def test_wb_identity_requires_explicit_unit_headroom():
    data = np.tile([.2, .3, .4], (8, 1))
    diagonal, _ = fit_neutral_wb(data, data, CELLS, NEUTRALS, policy=StagePolicy(version="modern-v1", wb_headroom=1))
    np.testing.assert_array_equal(diagonal, np.eye(3))


@pytest.mark.parametrize("measured", [np.zeros((8, 3)), np.full((8, 3), .001), np.full((8, 3), np.nan)])
def test_wb_rejects_dark_or_invalid_neutrals_before_ratio(measured):
    with pytest.raises(ValueError, match="two usable"):
        fit_neutral_wb(measured, np.full((8, 3), .3), CELLS, NEUTRALS, policy=POLICY)


def test_wb_rejects_unbounded_gains_and_insufficient_unclipped_neutrals():
    with pytest.raises(ValueError, match="max_gain"):
        fit_neutral_wb(np.full((8, 3), .02), np.ones((8, 3)), CELLS, NEUTRALS, policy=POLICY)
    clip = np.ones(8)
    clip[-1] = 0
    with pytest.raises(ValueError, match="two usable"):
        fit_neutral_wb(np.full((8, 3), .3), np.full((8, 3), .3), CELLS, NEUTRALS, clipped_fractions=clip, policy=POLICY)


def test_saturation_report_does_not_exclude_or_transform_pixels():
    mask, report = saturation_selection(CELLS, [.1] * 8, StagePolicy(version="modern-v1", saturation="report"))
    assert mask.all()
    assert report["flagged_cell_ids"] == list(CELLS)
    assert report["excluded_cell_ids"] == []
    assert report["pixel_transform"] == "identity"


@pytest.mark.parametrize("bad", [-.1, 1.1, np.nan, np.inf])
def test_invalid_clipping_fractions_fail(bad):
    with pytest.raises(ValueError):
        saturation_selection(CELLS, [bad] * 8, POLICY)


@pytest.mark.parametrize("options", [{"version":"future"}, {"wb_headroom":0}, {"wb_headroom":1.1}, {"gc_min_neutrals":2}, {"gc_min_neutrals":True}, {"gc_min_span":np.nan}, {"saturation":"recover"}, {"unexpected":True}])
def test_policy_options_are_strict_and_roundtrip(options):
    assert StagePolicy.from_value(POLICY.to_dict()) == POLICY
    with pytest.raises((TypeError, ValueError)):
        StagePolicy.from_value(options)
