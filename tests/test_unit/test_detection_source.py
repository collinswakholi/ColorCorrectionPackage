"""Optional diagnostics cannot choose different fitting geometry."""

import numpy as np
import pytest

from ColorCorrectionPipeline.detection import bind_detection_source, detection_context, get_detection_diagnostics, model_chart_samples
from ColorCorrectionPipeline.detection import charts


def test_bound_original_pixels_choose_geometry_but_current_stage_is_sampled(monkeypatch):
    original = np.full((8, 8, 3), 50, dtype=np.uint8)
    transformed = np.full((8, 8, 3), 150, dtype=np.uint8)
    quad = np.array([[2., 2.], [5., 2.], [5., 5.], [2., 5.]])
    chart = {"type": 0, "patches": np.repeat(quad[None], 24, axis=0), "corners": quad, "dims": [3, 3]}
    forwarded = []

    def infer(image, *args):
        forwarded.append(image.copy())
        return [chart], {"provider": "fixture"}

    monkeypatch.setattr(charts, "_forward", infer)
    with detection_context({"backend": "model", "model_path": "fixture.onnx"}):
        bind_detection_source(original)
        original[:] = 0  # owned source remains independent
        values = model_chart_samples(transformed, 4)
        np.testing.assert_array_equal(forwarded[0], 50)
        np.testing.assert_allclose(values, 150 / 255)
        assert get_detection_diagnostics()[0]["geometry_source"] == "source_image"
        with pytest.raises(RuntimeError, match="already owns"):
            bind_detection_source(transformed)


def test_binding_requires_a_context():
    with pytest.raises(RuntimeError, match="detection_context"):
        bind_detection_source(np.zeros((8, 8, 3), dtype=np.uint8))
