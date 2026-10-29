"""Valid inference provenance survives no-chart and unsupported-geometry outcomes."""

import numpy as np
import pytest

from ColorCorrectionPipeline.detection import charts


def details():
    return {"provider": "CPUExecutionProvider", "runtime_version": "fixture-runtime",
            "model": "fixture.onnx", "sha256": "fixture-model-hash", "cpu_fallback": True,
            "selected_device": "cpu", "execution_verified": True,
            "attempts": [{"device": "cuda", "status": "failed", "reason": "driver fixture"},
                         {"device": "cpu", "status": "succeeded"}]}


def test_no_chart_retains_forward_provider_and_cpu_recovery_reason(monkeypatch):
    evidence = details()
    monkeypatch.setattr(charts, "_forward", lambda *args: ([], evidence))
    with charts.detection_context():
        with pytest.raises(charts.NoChartDetectedError):
            charts.model_chart_samples(np.zeros((8, 8, 3), dtype=np.uint8), 1)
        recorded = charts.get_detection_diagnostics()[-1]
        assert recorded["status"] == "failed"
        assert recorded["error_type"] == "NoChartDetectedError"
        assert recorded["forward_completed"] is True
        assert recorded["provider"] == evidence["provider"]
        assert recorded["runtime_version"] == evidence["runtime_version"]
        assert recorded["sha256"] == evidence["sha256"]
        assert recorded["cpu_fallback"] is True
        assert "driver fixture" in recorded["attempts"][0]["reason"]
        recorded["attempts"][0]["reason"] = "caller mutation"
        assert charts.get_detection_diagnostics()[-1]["attempts"][0]["reason"] == "driver fixture"


def test_unsupported_chart_retains_successful_inference_details(monkeypatch):
    evidence = details()
    monkeypatch.setattr(charts, "_forward", lambda *args: ([{"type": 1}], evidence))
    with charts.detection_context():
        with pytest.raises(charts.UnsupportedChartError):
            charts.model_chart_samples(np.zeros((8, 8, 3), dtype=np.uint8), 1)
        recorded = charts.get_detection_diagnostics()[-1]
        assert recorded["provider"] == "CPUExecutionProvider"
        assert recorded["model"] == "fixture.onnx"
        assert recorded["forward_completed"] is True
        assert recorded["error_type"] == "UnsupportedChartError"


def test_decode_failure_carries_completed_inference_details(monkeypatch):
    def broken(*args):
        raise ValueError("decode fixture error")

    monkeypatch.setattr(charts, "infer_model", lambda *args: ([np.zeros((1, 29, 1))], (1, 0, 0), details()))
    monkeypatch.setattr(charts, "_decode", broken)
    with charts.detection_context():
        with pytest.raises(charts.ModelUnavailableError):
            charts.model_chart_samples(np.zeros((8, 8, 3), dtype=np.uint8), 1)
        recorded = charts.get_detection_diagnostics()[-1]
        assert recorded["provider"] == "CPUExecutionProvider"
        assert recorded["forward_completed"] is True
        assert "decode fixture error" in recorded["reason"]
