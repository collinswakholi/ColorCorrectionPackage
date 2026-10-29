"""Legacy artifacts must not opt into new numerical approximations on load."""

import numpy as np

from ColorCorrectionPipeline.core.correction import Regressor_Model, _predict_direct, fit_model, predict_


def test_pre_lut_state_keeps_direct_predictions_for_large_images():
    x = np.random.default_rng(12).random((80, 3))
    model = fit_model(x, x ** 3, {"mtd": "linear", "degree": 3, "use_lut": False})
    state = model.__getstate__()
    for key in ("use_lut", "lazy_lut", "lut_min_pixels", "_lut", "_lut_grid_size", "_lut_predict_fn"):
        state.pop(key, None)
    restored = Regressor_Model.__new__(Regressor_Model)
    restored.__setstate__(state)
    assert restored.use_lut is False
    assert restored._lut_predict_fn is None
    image = np.random.default_rng(13).random((4097, 3))
    np.testing.assert_array_equal(predict_(image, restored), _predict_direct(image, restored))
