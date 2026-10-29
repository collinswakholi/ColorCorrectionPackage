"""Contract, geometry and failure tests independent of learned weights.

Photograph inference is an additional integration check, not mocked here.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from ColorCorrectionPipeline.detection import (
    ChartDetectionError,
    DetectionOptions,
    ModelUnavailableError,
    NoChartDetectedError,
    UnsupportedChartError,
    detection_context,
    get_detection_diagnostics,
    model_chart_samples,
    model_extract_color_charts,
)
from ColorCorrectionPipeline.detection import charts as detection


def _fixture(chart_type=0, rotation=0):
    rows, cols = detection._LAYOUTS[chart_type]
    h = np.array([[240., 8., 35.], [10., 170., 40.], [.02, .03, 1.]])
    if rotation:
        theta = np.deg2rad(rotation)
        turn = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
        # Rotate around the 320x260 image center without losing the chart.
        before = np.array([[1., 0., -160.], [0., 1., -130.], [0, 0, 1.]])
        after = np.array([[1., 0., 160.], [0., 1., 130.], [0, 0, 1.]])
        h = after @ turn @ before @ h
    centers = detection._warp(detection._anchor_points(rows, cols), h)
    diag = np.hypot(240 / cols, 170 / rows) * .7
    anchors = [{"type": chart_type, "anchor": i, "center": center, "diag": diag, "score": .95} for i, center in enumerate(centers)]
    chart = detection._assemble_one(anchors, chart_type, .75, 320, 260)
    return chart, anchors


@pytest.mark.parametrize("kwargs", [
    {"backend": "typo"}, {"confidence": 0}, {"confidence": float("nan")},
    {"iou": 1.01}, {"sampling_inset": 0}, {"max_detections": True},
    {"max_detections": 1001}, {"allow_mcc_fallback": "false"},
    {"model_path": 42}, {"recall_model_path": ""},
])
def test_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DetectionOptions(**kwargs)


def test_options_do_not_alias_caller_settings():
    source = {"backend": "model", "model_path": "first.onnx"}
    options = DetectionOptions.from_value(source)
    source["model_path"] = "second.onnx"
    assert options.model_path == "first.onnx"


def test_preprocessing_is_rgb_nchw_and_reversible_coordinates():
    bgr = np.zeros((20, 40, 3), dtype=np.uint8)
    bgr[:] = [10, 20, 30]
    tensor, gain, left, top = detection._letterbox(bgr, 80)
    assert tensor.shape == (1, 3, 80, 80)
    np.testing.assert_allclose(tensor[0, :, 30, 20], np.array([30, 20, 10]) / 255)
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.full(3, 114 / 255))
    assert (gain, left, top) == (2, 0, 20)


@pytest.mark.parametrize("layout", ["nc", "cn", "bnc", "bcn"])
def test_output_layouts_and_letterbox_decode(layout):
    rows = np.zeros((2, 29), dtype=np.float32)
    rows[:, :4] = [[30, 50, 10, 12], [90, 70, 10, 12]]
    rows[0, 4] = .9
    rows[1, 5] = .8
    output = {"nc": rows, "cn": rows.T, "bnc": rows[None], "bcn": rows.T[None]}[layout]
    decoded = detection._decode(output, DetectionOptions(), gain=2, left=10, top=20)
    assert len(decoded) == 2
    np.testing.assert_allclose(decoded[0]["center"], [10, 15])
    assert decoded[0]["anchor"] == 0
    assert decoded[1]["anchor"] == 1


@pytest.mark.parametrize("output", [np.zeros((1, 28, 10)), np.zeros((2, 29, 10)), np.full((29, 2), np.nan), np.full((29, 2), np.inf)])
def test_malformed_outputs_fail_honestly(output):
    with pytest.raises(ModelUnavailableError):
        detection._decode(output, DetectionOptions())


def test_oriented_nms_preserves_distinct_anchors():
    rows = np.zeros((3, 29), dtype=np.float32)
    rows[:, :4] = [[20, 20, 20, 10], [20.1, 20, 20, 10], [60, 20, 20, 10]]
    rows[0, 4], rows[1, 4], rows[2, 5] = .9, .8, .7
    decoded = detection._decode(rows, DetectionOptions())
    assert [d["anchor"] for d in decoded] == [0, 1]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_semantic_cell_order_survives_rotation(rotation):
    chart, anchors = _fixture(rotation=rotation)
    assert chart is not None
    assert chart["patches"].shape == (24, 4, 2)
    # Index0 remains the learned TL semantic cell; it is not re-sorted by
    # visual position after a 180/270-degree rotation.
    np.testing.assert_allclose(chart["patches"][0].mean(axis=0), anchors[0]["center"], atol=1)
    np.testing.assert_allclose(chart["patches"][23].mean(axis=0), anchors[2]["center"], atol=1)


def test_collinear_or_outside_geometry_is_rejected():
    _, anchors = _fixture()
    bad = [dict(a, center=np.array([a["anchor"] * 10., 30.])) for a in anchors]
    assert detection._assemble_one(bad, 0, .75, 320, 260) is None
    outside = [dict(a, center=a["center"] - [200, 0]) for a in anchors]
    assert detection._assemble_one(outside, 0, .75, 320, 260) is None


def test_synthetic_cells_sample_rgb_order_and_legacy_scale(monkeypatch):
    chart, _ = _fixture()
    image = np.zeros((260, 320, 3), dtype=np.uint8)
    for i, quad in enumerate(chart["patches"]):
        # Paint a physical cell larger than the inset sampling region. Painting
        # only the exact inset would intentionally blend its border with black.
        painted = quad.mean(axis=0) + (quad - quad.mean(axis=0)) * 1.2
        cv2.fillConvexPoly(image, np.round(painted).astype(np.int32), (i + 10, i + 20, i + 30))
    monkeypatch.setattr(detection, "_forward", lambda *a: ([chart], {"provider": "fixture"}))
    with detection_context({"backend": "model", "model_path": "fixture.onnx"}):
        samples, _, dims = model_extract_color_charts(image)
        assert samples[0].shape == (24, 3)
        np.testing.assert_allclose(samples[0][0], [30, 20, 10], atol=1)
        extended = model_chart_samples(image, 9).reshape(24, 9, 3)
        assert extended.shape == (24, 9, 3)
        np.testing.assert_allclose(extended[0].mean(axis=0), np.array([30, 20, 10]) / 255, atol=.01)
        assert len(dims) == 2


def test_run_context_reuses_geometry_and_resamples_changed_pixels(monkeypatch):
    chart, _ = _fixture()
    calls = []
    monkeypatch.setattr(detection, "_forward", lambda *a: (calls.append(1) or [chart], {"provider": "fixture"}))
    with detection_context({"backend": "model", "model_path": "fixture.onnx"}):
        first = model_chart_samples(np.full((260, 320, 3), 100, np.uint8), 4)
        second = model_chart_samples(np.full((260, 320, 3), 200, np.uint8), 4)
        assert len(calls) == 1
        np.testing.assert_allclose(second, first * 2)
        with pytest.raises(ChartDetectionError, match="dimensions changed"):
            model_chart_samples(np.full((261, 320, 3), 200, np.uint8), 4)


def test_no_lattice_and_unsupported_layout_are_distinct(monkeypatch):
    image = np.zeros((260, 320, 3), np.uint8)
    monkeypatch.setattr(detection, "_forward", lambda *a: ([], {}))
    with detection_context({"backend": "model", "model_path": "fixture.onnx"}):
        with pytest.raises(NoChartDetectedError):
            model_chart_samples(image, 1)
    assert get_detection_diagnostics()[-1]["error_type"] == "NoChartDetectedError"
    chart, _ = _fixture(chart_type=1)
    monkeypatch.setattr(detection, "_forward", lambda *a: ([chart], {}))
    with detection_context({"backend": "model", "model_path": "fixture.onnx"}):
        with pytest.raises(UnsupportedChartError, match="spyder"):
            model_chart_samples(image, 1)


def test_fallback_requires_explicit_permission_and_is_reported(monkeypatch):
    image = np.zeros((260, 320, 3), np.uint8)
    monkeypatch.setattr(detection, "_forward", lambda *a: ([], {}))
    with detection_context({"backend": "model", "model_path": "fixture.onnx", "allow_mcc_fallback": True}):
        with pytest.warns(RuntimeWarning, match="explicit MCC fallback"):
            assert model_chart_samples(image, 1) is None
        assert get_detection_diagnostics()[-1]["status"] == "fallback"
        assert model_chart_samples(image, 1) is None


def test_explicit_mcc_never_loads_model(monkeypatch):
    def forbidden(*a):
        pytest.fail("MCC route attempted model inference")
    monkeypatch.setattr(detection, "_forward", forbidden)
    with detection_context({"backend": "mcc", "model_path": "fixture.onnx"}):
        assert model_chart_samples(np.zeros((10, 10, 3), np.uint8), 1) is None
        assert get_detection_diagnostics()[-1]["backend"] == "mcc"


def test_missing_bundled_model_fails_in_both_auto_and_model(monkeypatch, tmp_path):
    monkeypatch.delenv("CCP_CHART_MODEL", raising=False)
    monkeypatch.setattr(detection, "DEFAULT_MODEL_PATH", tmp_path / "absent.onnx")
    image = np.zeros((10, 10, 3), np.uint8)
    for backend in ("auto", "model"):
        with detection_context({"backend": backend}):
            with pytest.raises(ModelUnavailableError, match="does not exist"):
                model_chart_samples(image, 1)


def test_environment_model_path_takes_precedence_over_bundle(monkeypatch):
    chart, _ = _fixture()
    paths = []
    monkeypatch.setenv("CCP_CHART_MODEL", "environment.onnx")
    monkeypatch.setattr(detection, "_forward", lambda image, path, options: (paths.append(path) or [chart], {}))
    with detection_context():
        model_chart_samples(np.zeros((260, 320, 3), np.uint8), 1)
    assert paths[0].name == "environment.onnx"


def test_context_settings_and_diagnostics_are_thread_local(monkeypatch):
    chart, _ = _fixture()
    monkeypatch.setattr(detection, "_forward", lambda image, path, options: ([chart], {"model": str(path)}))
    def run(name):
        with detection_context({"backend": "model", "model_path": name}):
            model_chart_samples(np.zeros((260, 320, 3), np.uint8), 1)
            result = get_detection_diagnostics()
            assert Path(result[-1]["model"]).name == name
            result[-1]["model"] = "mutated"
            assert Path(get_detection_diagnostics()[-1]["model"]).name == name
            return name
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(run, ["one.onnx", "two.onnx"])) == ["one.onnx", "two.onnx"]


def test_cache_identity_includes_unsampled_pixels():
    from ColorCorrectionPipeline.core.utils import _ChartCache
    a = np.zeros((64, 256, 3), np.uint8)
    b = a.copy()
    b[1, 1] = 255
    assert _ChartCache._hash(a) != _ChartCache._hash(b)


def test_utils_routes_model_without_creating_mcc(monkeypatch):
    import ColorCorrectionPipeline.detection as api
    from ColorCorrectionPipeline.core import utils
    monkeypatch.setattr(api, "model_extract_color_charts", lambda *a, **k: ([np.full((24, 3), 128.)], np.zeros((10, 10, 3), np.uint8), [2, 3]))
    monkeypatch.setattr(utils, "_create_detector", lambda: pytest.fail("MCC should not execute"))
    rgb, _, dims = utils.extract_color_chart(np.zeros((10, 10, 3), np.uint8), get_patch_size=True)
    assert rgb[0, 0] == 128
    assert dims == [2, 3]


def test_legacy_empty_detector_return_is_handled(monkeypatch):
    from ColorCorrectionPipeline.core import utils
    monkeypatch.setattr(utils, "_create_detector", lambda: object())
    monkeypatch.setattr(utils, "detect_refine_charts", lambda *a, **k: None)
    assert utils._extract_color_chart_mcc(np.zeros((10, 10, 3), np.uint8)) == (None, None, None)


def test_mcc_no_chart_returns_empty_without_assertion():
    from ColorCorrectionPipeline.core import utils
    class EmptyDetector:
        def process(self, *a, **k):
            return False
        def getListColorChecker(self):
            return []
    assert utils.detect_refine_charts(EmptyDetector(), np.zeros((10, 10, 3), np.uint8), 1) == []
