"""Unit tests for correction model compatibility paths."""

import numpy as np

from ColorCorrectionPipeline.core.correction import estimate_gamma_profile, fit_model, predict_


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


def test_custom_torch_disables_batch_norm_for_tiny_training_split():
    detected = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]], dtype=np.float64)
    reference = np.clip(detected * 0.9 + 0.02, 0.0, 1.0)

    model = fit_model(
        detected,
        reference,
        {
            "mtd": "custom",
            "degree": 1,
            "hidden_layers": [4],
            "max_iterations": 1,
            "batch_size": 16,
            "use_batch_norm": True,
            "use_lut": False,
        },
    )

    assert model.model.use_batch_norm is False


def test_lut_is_lazy_and_skipped_for_small_predictions():
    detected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    reference = detected.copy()

    model = fit_model(
        detected,
        reference,
        {
            "mtd": "linear",
            "degree": 1,
            "lazy_lut": True,
            "lut_grid_size": 5,
            "lut_min_pixels": 10,
        },
    )

    assert model._lut is None
    predict_(detected[:2], model)
    assert model._lut is None

    image = np.tile(detected[None, :4, :], (4, 1, 1))
    predicted = predict_(image, model)
    assert predicted.shape == image.shape
    assert model._lut is not None


def test_gamma_profile_uses_identity_when_chart_detection_fails(monkeypatch):
    image = np.full((4, 4, 3), 0.5, dtype=np.float64)
    ref = np.tile(np.array([[0.2, 0.3, 0.4]]), (24, 1))
    illuminant = np.array([0.31271, 0.32902])

    monkeypatch.setattr(
        "ColorCorrectionPipeline.core.correction.extract_neutral_patches",
        lambda *args, **kwargs: (None, None),
    )

    coeffs, corrected, metrics = estimate_gamma_profile(
        image,
        ref,
        illuminant,
        get_deltaE=False,
    )

    np.testing.assert_array_equal(coeffs, np.array([1, 0]))
    np.testing.assert_array_equal(corrected, image)
    assert metrics == {}
