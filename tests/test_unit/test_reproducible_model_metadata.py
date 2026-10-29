"""New fitting labels cannot conceal incompatible inference parameters on load."""
import copy
import pickle

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, MyModels
from .test_modern_pipeline import chart, config


def fitted(monkeypatch, conventional=False):
    image, reference = chart(monkeypatch)
    cc = ColorCorrection()
    parameters = {"cc_method": "conv", "degree": 1} if conventional else {"degree": 1}
    assert not cc.run(image, config=config(reference_profile=reference, do_gc=False,
                         do_wb=False, CC_kwargs=parameters))[2], cc.last_run
    return cc.models


def rejected_without_activation(model, tmp_path):
    (tmp_path / "corrupt.pkl").write_bytes(pickle.dumps(model))
    active = MyModels()
    active.model_wb = np.eye(3) * .5
    previous = active.model_wb
    with pytest.raises(OSError):
        active.load(str(tmp_path), "corrupt")
    assert active.model_wb is previous
    assert active.metadata == {}


@pytest.mark.parametrize("corruption", ["legacy-estimator", "unknown-method", "short-tuple",
    "options-degree", "estimator-degree", "missing-fit-report", "missing-effective-report",
    "missing-options", "legacy-policy-modern-estimator"])
def test_reproducible_cc_model_and_metadata_must_agree(monkeypatch, tmp_path, corruption):
    model = copy.deepcopy(fitted(monkeypatch))
    estimator, options, method = model.model_cc
    if corruption == "legacy-estimator":
        estimator.fitting_version = "legacy-v1"
    elif corruption == "unknown-method":
        model.model_cc = (estimator, options, "not-a-method")
    elif corruption == "short-tuple":
        model.model_cc = (estimator, options)
    elif corruption == "options-degree":
        options["degree"] = 2
    elif corruption == "estimator-degree":
        estimator.degree = 2
    elif corruption == "missing-fit-report":
        del estimator.fit_report
    elif corruption == "missing-effective-report":
        model.metadata["numerical_report"]["CC"]["fit"].pop("effective_parameters")
    elif corruption == "missing-options":
        model.model_cc = (estimator, {}, method)
    else:
        model.metadata["fitting_policy"] = {"version": "legacy-v1"}
        model.metadata["stage_policy"] = {"version": "legacy-v1"}
    rejected_without_activation(model, tmp_path)


@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "basis-degree", "missing-effective-report"])
def test_conventional_cc_basis_and_matrix_must_agree(monkeypatch, tmp_path, corruption):
    model = copy.deepcopy(fitted(monkeypatch, conventional=True))
    matrix, options, method = model.model_cc
    if corruption == "shape":
        model.model_cc = (np.ones((3, 1)), options, method)
    elif corruption == "nonfinite":
        broken = matrix.copy()
        broken[0, 0] = np.nan
        model.model_cc = (broken, options, method)
    elif corruption == "basis-degree":
        options["degree"] = 2
    else:
        model.metadata["numerical_report"]["CC"]["fit"].pop("effective_parameters")
    rejected_without_activation(model, tmp_path)
