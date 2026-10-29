"""A real near-black counterexample must fail the strengthened sampled audit."""
import json
from pathlib import Path
import numpy as np

from ColorCorrectionPipeline import ColorCorrection, LutCache, MonotonicToneModel
from ColorCorrectionPipeline.core.correction import Regressor_Model
from ColorCorrectionPipeline.portable_inference import AffinePredictor


def calibration():
    values = json.loads((Path(__file__).parents[1]/"fixtures/pointwise_lut_stress.json").read_text())
    model = Regressor_Model()
    model.degree, model.mtd, model.use_lut = values["cc"]["degree"], "linear", False
    model.portable_inference = True
    # The fixture's zero center/unit scale are neutral. Linear bundles encode
    # this convention with None; explicit normalization belongs to PLS.
    assert not np.any(values["cc"]["center"]) and np.all(np.array(values["cc"]["scale"]) == 1)
    model.model = AffinePredictor(values["cc"]["coefficients"], values["cc"]["intercept"])
    cc = ColorCorrection()
    cc.models.model_gc = MonotonicToneModel(np.array(values["tone"]["measured_knots"]), np.array(values["tone"]["target_knots"]))
    cc.models.model_wb = np.array(values["wb_matrix"])
    cc.models.model_cc = model, {}, "ours"
    return cc


def test_near_black_byte_cube_rejects_previously_accepted_preview():
    cc = calibration()
    witness = np.array([[[3., 1., 0.], [3., 1., 1.]]])/255
    actual = cc.predict_image(witness)["CC"]
    np.testing.assert_allclose(actual, [[[0., 3.891275425605221e-5, .0049850505770031445],
                                      [0., .00036026849670269825, .009539662942978012]]], atol=2e-14, rtol=0)
    with cc.prepare_execution({"mode": "combined-lut", "backend": "numpy", "grid_sizes": (129,),
        "max_rgb_error": 1/255, "max_delta_e00": .5, "validation_samples": 64,
        "stress_grid_size": 3}, cache=LutCache()) as prepared:
        proof = prepared.preparation["qualification"]
        assert proof["audit"]["max_rgb_error"] < 1/255
        assert proof["audit"]["max_delta_e00"] < .5
        assert proof["stress"]["max_rgb_error"] > .005
        assert proof["stress"]["max_delta_e00"] > .84
        assert proof["selected_mode"] == "direct" and "dense stress" in proof["reason"]
        np.testing.assert_allclose(prepared.predict_image(witness).images["CC"], actual, atol=2e-14, rtol=0)
