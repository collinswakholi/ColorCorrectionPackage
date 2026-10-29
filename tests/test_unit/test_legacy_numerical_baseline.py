"""Numerical snapshots that patches 0--2 must preserve.

These exercise explicitly supplied transforms, independently of chart detector
selection. A new detector can change fitted observations; that does not justify
an unreviewed change to applying the same saved parameters.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection
from ColorCorrectionPipeline.core.color_spaces import extrapolate_if_sat_image
from ColorCorrectionPipeline.core.correction import predict_image
from ColorCorrectionPipeline.core.utils import compute_diag


@pytest.fixture(scope="module")
def baseline():
    path = Path(__file__).parents[1] / "fixtures" / "legacy_numeric_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_classic_reference_values_preserve_recorded_baseline(baseline):
    pipeline = ColorCorrection()
    reference = pipeline.get_reference_values()

    assert reference.index.tolist() == baseline["reference_names"]
    np.testing.assert_allclose(
        reference.values, baseline["reference_rgb"], rtol=0, atol=1e-10
    )
    np.testing.assert_array_equal(
        pipeline.REF_ILLUMINANT, baseline["reference_illuminant"]
    )


def test_applying_existing_gamma_parameters_preserves_baseline(baseline):
    corrected = predict_image(
        np.asarray(baseline["gamma_input"]),
        np.asarray(baseline["gamma_coefficients"]),
    )

    # Fast Lab conversion can differ slightly between supported CPU backends.
    np.testing.assert_allclose(
        corrected, baseline["gamma_output"], rtol=0, atol=2e-6
    )


def test_matched_white_balance_preserves_legacy_headroom(baseline):
    samples = np.asarray(baseline["matched_wb_input"])

    np.testing.assert_allclose(
        compute_diag(samples, samples),
        baseline["matched_wb_diagonal"],
        rtol=0,
        atol=1e-14,
    )


def test_legacy_saturation_scaling_is_not_silently_redefined(baseline):
    corrected, _, _ = extrapolate_if_sat_image(
        np.asarray(baseline["saturation_input"]),
        np.asarray(baseline["reference_rgb"]),
    )

    np.testing.assert_allclose(
        corrected, baseline["saturation_output"], rtol=0, atol=1e-14
    )
