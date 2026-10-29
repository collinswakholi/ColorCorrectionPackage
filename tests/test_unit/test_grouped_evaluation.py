"""Validation must account for every measured observation and failed fold."""
import numpy as np
import pytest

from ColorCorrectionPipeline.evaluation import grouped_cc_validation


def data(repeats=3):
    x = np.random.default_rng(8).random((12, 3))
    groups = ["cell-%02d" % i for i in range(12)]
    return np.repeat(x, repeats, axis=0), np.repeat(x, repeats, axis=0), np.repeat(groups, repeats)


def test_groups_never_leak_and_anchors_only_train():
    x, y, groups = data()
    x = np.vstack((x, [[0, 0, 0]]))
    y = np.vstack((y, [[0, 0, 0]]))
    groups = np.append(groups, "synthetic-black")
    calls = []
    def fit_predict(xt, yt, gt, test):
        assert "synthetic-black" in gt
        calls.append((xt, yt, gt, test))
        return test.copy()
    report = grouped_cc_validation(x, y, groups, fit_predict,
             synthetic_mask=np.array([False] * 36 + [True]), n_splits=4)
    assert report["status"] == "available"
    assert report["scored_observations"] == report["expected_observations"] == 36
    assert report["scored_groups"] == 12
    assert report["mean"] == 0
    assert "upstream stages fixed" in report["scope"]
    for fold in report["folds"]:
        assert not set(fold["test_groups"]) & set(fold["training_groups"])
        assert 36 not in fold["test_indices"]
    assert len(calls) == 4


def test_excluded_groups_never_reach_training_or_test():
    x, y, groups = data()
    eligible = groups != groups[0]
    def fit_predict(xt, yt, gt, test):
        assert groups[0] not in gt
        return test
    report = grouped_cc_validation(x, y, groups, fit_predict, eligible=eligible)
    assert report["expected_observations"] == 36
    assert report["excluded_observations"] == 3
    assert report["scored_observations"] == 33
    assert report["excluded_groups"] == 1


@pytest.mark.parametrize("failure", ["raise", "nan", "shape", "metricnan", "metricshape"])
def test_failed_fold_not_silently_dropped(failure):
    x, y, groups = data()
    def fit_predict(xt, yt, gt, test):
        if failure == "raise":
            raise RuntimeError("deliberate failure")
        if failure == "nan":
            return test * np.nan
        if failure == "shape":
            return test[:1]
        return test
    def metric(expected, prediction, indices):
        return np.full(len(expected), np.nan) if failure == "metricnan" else np.array([0])
    report = grouped_cc_validation(x, y, groups, fit_predict,
                metric=metric if failure.startswith("metric") else None)
    assert report["status"] == "unavailable"
    assert report["mean"] is None
    assert report["failed_observations"] == 36
    assert report["failed_groups"] == 12
    assert report["scored_observations"] == 0


def test_canonical_metric_receives_original_indices():
    x, y, groups = data()
    received = []
    def metric(expected, prediction, indices):
        np.testing.assert_array_equal(expected, y[indices])
        received.extend(indices.tolist())
        return np.ones(len(expected)) * 2
    report = grouped_cc_validation(x, y, groups, lambda a, b, c, d: d,
                metric=metric, metric_name="CIEDE2000 canonical Lab D50")
    assert sorted(received) == list(range(36))
    assert report["mean"] == 2
    assert report["metric"] == "CIEDE2000 canonical Lab D50"


def test_insufficient_groups_unavailable_with_full_accounting():
    x, y, groups = data()
    report = grouped_cc_validation(x[:6], y[:6], groups[:6], lambda a, b, c, d: d, n_splits=4)
    assert report["status"] == "unavailable"
    assert report["failed_observations"] == 6
    assert report["expected_groups"] == report["failed_groups"] == 2
    assert report["folds"] == []


def test_validation_rng_scoped_and_repeats_weighted_by_group():
    x, y, groups = data(1)
    state = np.random.get_state()
    report = grouped_cc_validation(x, y, groups, lambda a, b, c, d: d + .02)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    xx = np.vstack((x, np.repeat(x[:1], 30, axis=0)))
    yy = np.vstack((y, np.repeat(y[:1], 30, axis=0)))
    gg = np.append(groups, [groups[0]] * 30)
    repeated = grouped_cc_validation(xx, yy, gg, lambda a, b, c, d: d + .02)
    assert repeated["mean"] == pytest.approx(report["mean"], abs=1e-16)


@pytest.mark.parametrize("kind", ["eligibility", "synthetic", "target"])
def test_invalid_group_assignments_fail_early(kind):
    x, y, groups = data()
    kwargs = {}
    if kind == "eligibility":
        mask = np.ones(len(x), bool)
        mask[0] = False
        kwargs["eligible"] = mask
    elif kind == "synthetic":
        mask = np.zeros(len(x), bool)
        mask[0] = True
        kwargs["synthetic_mask"] = mask
    else:
        y[0] += .01
    with pytest.raises(ValueError):
        grouped_cc_validation(x, y, groups, lambda a, b, c, d: d, **kwargs)


def test_one_failed_fold_invalidates_aggregate_without_losing_successes():
    x, y, groups = data()
    calls = []
    def predict(a, b, c, d):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("one fold failed")
        return d
    report = grouped_cc_validation(x, y, groups, predict)
    assert report["status"] == "unavailable"
    assert report["mean"] is None
    assert report["scored_observations"] == 27
    assert report["failed_observations"] == 9
    assert report["scored_groups"] == 9
    assert report["failed_groups"] == 3


def test_all_groups_excluded_are_unavailable_with_complete_accounting():
    x, y, groups = data()
    def forbidden(*args):
        raise AssertionError("excluded rows must never be fitted")
    report = grouped_cc_validation(x, y, groups, forbidden, eligible=np.zeros(len(x), bool))
    assert report["status"] == "unavailable"
    assert report["scored_observations"] == report["failed_observations"] == 0
    assert report["excluded_observations"] == report["expected_observations"] == 36
