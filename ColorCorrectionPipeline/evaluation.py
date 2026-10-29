"""Grouped, completely accounted CC-only held-out-color validation.

Upstream GC/WB/FFC remain fixed. This is neither whole-pipeline validation nor
independent-capture accuracy, and must not be represented as either.
"""
import numpy as np

from .fitting import _paired_data, _integer


def grouped_cc_validation(X, Y, groups, fit_predict, *, n_splits=4, random_state=42,
                          eligible=None, synthetic_mask=None, metric=None,
                          metric_name="encoded-RGB Euclidean distance"):
    """Fit each training fold and score its excluded measured groups.

    ``fit_predict(Xtrain, Ytrain, groupstrain, Xtest)`` must fit a fresh estimator.
    ``metric(expected, predicted, original_indices)`` returns one finite scalar per
    test row. Original indices permit use of unclipped canonical reference truth.
    Synthetic rows are training-only. Excluded rows are absent from every fold.
    Targets from held-out groups never reach the fitting callback. One failed
    prediction/metric invalidates the aggregate; failures are never dropped.
    Each group is equally weighted in the reported mean irrespective of repeats.
    """
    X, Y, groups, synthetic = _paired_data(X, Y, groups, synthetic_mask)
    n_splits = _integer(n_splits, "n_splits", 2)
    random_state = _integer(random_state, "random_state", 0, 2**32 - 1)
    allowed = np.ones(len(X), bool) if eligible is None else np.asarray(eligible)
    if allowed.shape != (len(X),) or allowed.dtype.kind != "b":
        raise ValueError("eligible must be a boolean vector with one entry per row")
    for group in set(groups.tolist()):
        row = groups == group
        if np.any(synthetic[row] != synthetic[row][0]):
            raise ValueError("groups cannot mix measured and synthetic rows")
        if np.any(allowed[row] != allowed[row][0]):
            raise ValueError("eligibility must be uniform within each cell group")
        if not np.allclose(Y[row], Y[row][0], atol=1e-12, rtol=0):
            raise ValueError("a cell group has inconsistent reference targets")
    measured = ~synthetic
    evaluated = measured & allowed
    excluded = measured & ~allowed
    identities = sorted(set(groups[evaluated].tolist()))
    report = {"schema_version": 1, "scope": "CC-only held-out colors; upstream stages fixed",
              "status": "unavailable", "reason": None, "metric": metric_name,
              "expected_observations": int(measured.sum()), "scored_observations": 0,
              "excluded_observations": int(excluded.sum()), "failed_observations": 0,
              "expected_groups": len(set(groups[measured].tolist())), "scored_groups": 0,
              "excluded_groups": len(set(groups[excluded].tolist())), "failed_groups": 0,
              "synthetic_rows_train_only": int((synthetic & allowed).sum()),
              "random_state": random_state, "n_splits": n_splits, "folds": [],
              "mean": None, "median": None, "p95": None,
              "weighting": "equal canonical-cell groups; repeated rows averaged within group",
              "excluded_indices": np.flatnonzero(excluded).tolist(),
              "independent_capture": False, "whole_pipeline": False}
    if len(identities) < n_splits:
        report.update(reason="insufficient independent eligible groups for requested folds",
                      failed_observations=int(evaluated.sum()), failed_groups=len(identities))
        return report
    rng = np.random.default_rng(random_state)
    order = np.array(identities)[rng.permutation(len(identities))]
    group_errors = []
    for fold_id, test_groups in enumerate(np.array_split(order, n_splits)):
        test = evaluated & np.isin(groups, test_groups)
        training = allowed & ~np.isin(groups, test_groups)
        training_indices, test_indices = np.flatnonzero(training), np.flatnonzero(test)
        train_groups = sorted(set(groups[training & measured].tolist()))
        fold = {"fold": fold_id, "training_groups": train_groups, "test_groups": sorted(test_groups.tolist()),
                "training_indices": training_indices.tolist(), "test_indices": test_indices.tolist(),
                "expected_observations": int(test.sum()), "scored_observations": 0,
                "failed_observations": 0, "status": "failed", "reason": None,
                "group_scores": {}}
        try:
            if len(train_groups) < 2:
                raise ValueError("insufficient measured training groups")
            prediction = np.asarray(fit_predict(X[training].copy(), Y[training].copy(), groups[training].copy(), X[test].copy()), dtype=np.float64)
            if prediction.shape != Y[test].shape or not np.all(np.isfinite(prediction)):
                raise ValueError("held-out prediction shape or finiteness is invalid")
            errors = (np.linalg.norm(Y[test] - prediction, axis=1) if metric is None else
                      np.asarray(metric(Y[test].copy(), prediction, test_indices.copy()), dtype=np.float64))
            if errors.shape != (int(test.sum()),) or not np.all(np.isfinite(errors)) or np.any(errors < 0):
                raise ValueError("held-out metric must return one finite nonnegative value per row")
            for group in test_groups:
                value = float(errors[groups[test] == group].mean())
                fold["group_scores"][str(group)] = value
                group_errors.append(value)
            fold.update(status="scored", scored_observations=int(test.sum()))
            report["scored_observations"] += int(test.sum())
            report["scored_groups"] += len(test_groups)
        except Exception as exc:
            fold.update(reason=type(exc).__name__ + ": " + str(exc), failed_observations=int(test.sum()))
            report["failed_observations"] += int(test.sum())
            report["failed_groups"] += len(test_groups)
        report["folds"].append(fold)
    if report["failed_observations"]:
        report["reason"] = "one or more folds failed; no aggregate score is reported"
    else:
        report.update(status="available", mean=float(np.mean(group_errors)),
                      median=float(np.median(group_errors)), p95=float(np.percentile(group_errors, 95)))
    assert report["expected_observations"] == report["scored_observations"] + report["excluded_observations"] + report["failed_observations"]
    assert report["expected_groups"] == report["scored_groups"] + report["excluded_groups"] + report["failed_groups"]
    return report
