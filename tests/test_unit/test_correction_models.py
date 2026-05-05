"""Unit tests for correction model compatibility paths."""

import numpy as np

from ColorCorrectionPipeline.core.correction import fit_model, predict_


def test_custom_torch_model_fits_with_current_scheduler_api():
    rng = np.random.default_rng(123)
    detected = rng.random((32, 3))
    reference = np.clip(0.85 * detected + np.array([0.03, 0.02, 0.01]), 0.0, 1.0)

    model = fit_model(
        detected,
        reference,
        {
            "mtd": "custom",
            "degree": 1,
            "hidden_layers": [8],
            "max_iterations": 2,
            "batch_size": 8,
            "patience": 1,
            "random_state": 0,
            "tol": 1e-5,
        },
    )
    predicted = predict_(detected[:4].reshape(-1, 1, 3), model).reshape(-1, 3)

    assert predicted.shape == (4, 3)
    assert np.isfinite(predicted).all()