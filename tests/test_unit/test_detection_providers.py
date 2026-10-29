"""Provider policy/failure tests independent of a particular graphics driver."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ColorCorrectionPipeline.detection import DetectionOptions
from ColorCorrectionPipeline.detection import providers as p
from ColorCorrectionPipeline.detection import charts as charts_module


@pytest.mark.parametrize("system,expected", [
    ("Windows", ["cuda", "directml", "webgpu", "cpu"]),
    ("Linux", ["cuda", "webgpu", "cpu"]),
    ("Darwin", ["webgpu", "coreml", "cpu"]),
])
def test_auto_policy_tries_acceleration_before_cpu(system, expected):
    plans = p.provider_plans(system=system)
    assert [plan.device for plan in plans] == expected
    assert plans[-1].provider == "CPUExecutionProvider"


@pytest.mark.parametrize("device", ["cpu", "cuda", "directml", "vulkan", "webgpu", "metal", "coreml"])
def test_explicit_device_cpu_recovery_is_controlled(device):
    assert len(p.provider_plans(device, False)) == 1
    plans = p.provider_plans(device, True)
    assert plans[0].device == device
    assert len(plans) == (1 if device == "cpu" else 2)


@pytest.mark.parametrize("system,backend", [("Windows", "D3D12"), ("Linux", "Vulkan"), ("Darwin", "Metal")])
def test_webgpu_maps_to_native_graphics_backend(system, backend):
    assert p.provider_plans("webgpu", False, system)[0].graphics_backend == backend
    assert p.provider_plans("vulkan", False, system)[0].provider == "WebGpuExecutionProvider"


@pytest.mark.parametrize("changes", [
    {"device": "typo"}, {"device_id": -1}, {"device_id": True}, {"device_id": 64},
    {"allow_cpu_fallback": "true"}, {"cpu_threads": 0}, {"cpu_threads": 65},
    {"webgpu_library": ""}, {"webgpu_library": 42}, {"chart_index": -1}, {"chart_index": True},
])
def test_invalid_provider_options_are_rejected(changes):
    with pytest.raises(ValueError):
        DetectionOptions(**changes)


class Runtime:
    def __init__(self, plan, failure=None):
        self.input_size = 1280
        self.plan = plan
        self.details = {"registered_providers": [plan.provider]}
        self.execution_counts = {plan.provider: 3} if plan.device != "cpu" else None
        self.failure = failure

    def forward(self, tensor):
        if self.failure:
            raise RuntimeError(self.failure)
        return [np.ones((1, 29, 1), dtype=np.float32)]


def fixture_runtime(monkeypatch, tmp_path, fail_gpu=None):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"trusted fixture bytes")
    ort = SimpleNamespace(__version__="fixture", get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(p, "_import_ort", lambda: ort)
    monkeypatch.setattr(p.platform, "system", lambda: "Linux")
    calls = []

    def load(path, digest, plan, *args):
        calls.append(plan)
        return Runtime(plan, fail_gpu if plan.device != "cpu" else None)

    monkeypatch.setattr(p, "_load_session", load)
    return model, calls


def test_gpu_success_reports_execution_evidence(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path)
    _, geometry, detail = p.infer_model(model, DetectionOptions(device="cuda"), lambda size: (np.zeros((1,)), "geometry"))
    assert geometry == "geometry"
    assert detail["provider"] == "CUDAExecutionProvider"
    assert detail["execution_evidence"] == "node_profile"
    assert detail["node_provider_counts"] == {"CUDAExecutionProvider": 3}
    assert detail["cpu_fallback"] is False
    assert len(calls) == 1


def test_runtime_gpu_failure_retries_cpu_and_keeps_reason(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path, "GPU memory fixture failure")
    _, _, detail = p.infer_model(model, DetectionOptions(device="cuda"), lambda size: (np.zeros((1,)), None))
    assert [plan.device for plan in calls] == ["cuda", "cpu"]
    assert detail["provider"] == "CPUExecutionProvider"
    assert detail["cpu_fallback"] is True
    assert "GPU memory fixture failure" in detail["attempts"][0]["reason"]


def test_initialization_gpu_failure_retries_cpu(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path)
    previous = p._load_session

    def load(path, digest, plan, *args):
        if plan.device != "cpu":
            raise p.ProviderUnavailableError("missing GPU driver fixture")
        return previous(path, digest, plan, *args)

    monkeypatch.setattr(p, "_load_session", load)
    _, _, detail = p.infer_model(model, DetectionOptions(device="cuda"), lambda size: (np.zeros((1,)), None))
    assert detail["cpu_fallback"] is True
    assert "missing GPU driver" in detail["attempts"][0]["reason"]


def test_disabling_cpu_recovery_propagates_provider_attempts(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path, "GPU fixture failure")
    with pytest.raises(p.ProviderUnavailableError) as caught:
        p.infer_model(model, DetectionOptions(device="cuda", allow_cpu_fallback=False), lambda size: (np.zeros((1,)), None))
    assert len(calls) == 1
    assert caught.value.attempts[0]["device"] == "cuda"


def test_invalid_gpu_outputs_can_recover_on_cpu(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path)
    validation_calls = []

    def validate(output):
        validation_calls.append(1)
        if len(validation_calls) == 1:
            raise ValueError("nonfinite GPU output fixture")

    _, _, detail = p.infer_model(model, DetectionOptions(device="cuda"), lambda size: (np.zeros((1,)), None), validate)
    assert detail["selected_device"] == "cpu"
    assert detail["attempts"][0]["type"] == "ValueError"


def test_full_model_content_is_in_the_session_key(monkeypatch, tmp_path):
    model, _ = fixture_runtime(monkeypatch, tmp_path)
    digests = []

    def load(path, digest, plan, *args):
        digests.append(digest)
        return Runtime(plan)

    monkeypatch.setattr(p, "_load_session", load)
    options = DetectionOptions(device="cpu")
    p.infer_model(model, options, lambda size: (np.zeros((1,)), None))
    model.write_bytes(b"different fixture data")
    p.infer_model(model, options, lambda size: (np.zeros((1,)), None))
    assert digests[0] != digests[1]


def test_profile_requires_actual_gpu_nodes_and_rejects_cpu_nodes(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}]))
    session = SimpleNamespace(end_profiling=lambda: str(profile))
    with pytest.raises(p.ProviderUnavailableError, match="did not observe"):
        p._profile_execution(session, str(tmp_path), "CUDAExecutionProvider")


def test_profile_reports_gpu_execution_counts(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"cat": "Node", "args": {"provider": "DmlExecutionProvider"}}] * 2))
    session = SimpleNamespace(end_profiling=lambda: str(profile))
    assert p._profile_execution(session, str(tmp_path), "DmlExecutionProvider") == {"DmlExecutionProvider": 2}


def test_profile_rejects_mixed_execution(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"cat": "Node", "args": {"provider": provider}} for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")]))
    session = SimpleNamespace(end_profiling=lambda: str(profile))
    with pytest.raises(p.ProviderUnavailableError, match="executed CPU nodes"):
        p._profile_execution(session, str(tmp_path), "CUDAExecutionProvider")


def test_session_creation_must_register_requested_provider(monkeypatch):
    class Settings:
        def add_session_config_entry(self, *args):
            pass

    session = SimpleNamespace(disable_fallback=lambda: None, get_providers=lambda: ["CPUExecutionProvider"])
    fake = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                           SessionOptions=Settings, InferenceSession=lambda *args, **kwargs: session)
    monkeypatch.setattr(p, "_import_ort", lambda: fake)
    p.clear_provider_cache()
    with pytest.raises(p.ProviderUnavailableError, match="session registered"):
        p._load_session("fixture.onnx", "digest", p._CUDA, 0, 4, None, "fake")


def test_explicit_cpu_does_not_probe_gpu_plugins(monkeypatch, tmp_path):
    model, calls = fixture_runtime(monkeypatch, tmp_path)
    _, _, detail = p.infer_model(model, DetectionOptions(device="cpu"), lambda size: (np.zeros((1,)), None))
    assert [plan.device for plan in calls] == ["cpu"]
    assert detail["cpu_fallback"] is False


def test_automatic_directml_prefers_hardware_adapter_with_more_video_memory(monkeypatch):
    monkeypatch.setattr(p, "directml_adapters", lambda: ({"index": 0, "dedicated_video_bytes": 128}, {"index": 1, "dedicated_video_bytes": 16000}))
    index, metadata = p._directml_selection(None)
    assert index == metadata["index"] == 1


def test_explicit_directml_index_is_preserved(monkeypatch):
    def forbidden():
        raise AssertionError("explicit index must not enumerate")
    monkeypatch.setattr(p, "directml_adapters", forbidden)
    assert p._directml_selection(3) == (3, None)


def test_failed_automatic_directml_enumeration_reports_default_selection(monkeypatch):
    def unavailable():
        raise OSError("DXGI fixture failure")
    monkeypatch.setattr(p, "directml_adapters", unavailable)
    index, metadata = p._directml_selection(None)
    assert index == 0
    assert "DXGI fixture failure" in metadata["enumeration_error"]


def test_default_device_selection_is_automatic():
    assert DetectionOptions().device == "auto"
    assert DetectionOptions().device_id is None


def fake_webgpu(monkeypatch, tmp_path):
    library = tmp_path / "provider.so"
    library.write_bytes(b"fixture plugin bytes")
    device = SimpleNamespace(ep_name="WebGpuExecutionProvider", device=SimpleNamespace(
        type="GPU", vendor="fixture", device_id=7, metadata={"Description": "fixture GPU", "Discrete": "1"}))
    registrations, selected = [], []
    ort = SimpleNamespace(register_execution_provider_library=lambda *a: registrations.append(a), get_ep_devices=lambda: [device])
    settings = SimpleNamespace(add_provider_for_devices=lambda *a: selected.append(a))
    monkeypatch.setattr(p.platform, "system", lambda: "Linux")
    monkeypatch.setattr(p, "_registered_libraries", {})
    monkeypatch.setattr(p, "_webgpu_context", None)
    return ort, settings, str(library), registrations, selected


def test_native_vulkan_uses_real_plugin_device_api(monkeypatch, tmp_path):
    ort, settings, library, registrations, selected = fake_webgpu(monkeypatch, tmp_path)
    plan = p.provider_plans("vulkan", False)[0]
    detail = p._prepare_webgpu(ort, settings, plan, None, library)
    assert len(registrations) == 1
    assert selected[0][1]["dawnBackendType"] == "Vulkan"
    assert selected[0][1]["deviceId"] == "0"
    assert "adapterIndex" not in selected[0][1]
    assert detail["enumerated_gpu_devices"][0]["vendor"] == "fixture"


def test_explicit_dawn_adapter_index_is_not_context_id(monkeypatch, tmp_path):
    ort, settings, library, registrations, selected = fake_webgpu(monkeypatch, tmp_path)
    plan = p.provider_plans("vulkan", False)[0]
    p._prepare_webgpu(ort, settings, plan, 2, library)
    assert selected[0][1]["adapterIndex"] == "2"
    assert selected[0][1]["deviceId"] == "0"


def test_webgpu_backend_change_is_rejected_instead_of_misreported(monkeypatch, tmp_path):
    ort, settings, library, registrations, selected = fake_webgpu(monkeypatch, tmp_path)
    p._prepare_webgpu(ort, settings, p.provider_plans("vulkan", False)[0], None, library)
    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    with pytest.raises(p.ProviderUnavailableError, match="restart the process"):
        p._prepare_webgpu(ort, settings, p.provider_plans("metal", False)[0], None, library)
    assert len(selected) == 1


def test_webgpu_rejects_missing_plugin_device_api(monkeypatch, tmp_path):
    ort, settings, library, registrations, selected = fake_webgpu(monkeypatch, tmp_path)
    with pytest.raises(p.ProviderUnavailableError, match="plugin execution-provider APIs"):
        p._prepare_webgpu(SimpleNamespace(), settings, p.provider_plans("vulkan", False)[0], None, library)


def test_shared_session_serializes_forwards():
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def run(*args):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["active"], state["maximum"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return [np.zeros((1,))]

    runtime = p._Session(SimpleNamespace(run=run), "image", 1280, p._CPU, {})
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(runtime.forward, [np.zeros((1,))] * 3))
    assert state["maximum"] == 1


def test_failed_session_is_not_run_again():
    calls = []

    def broken(*args):
        calls.append(1)
        raise RuntimeError("provider fixture crashed")

    runtime = p._Session(SimpleNamespace(run=broken), "image", 1280, p._CPU, {})
    with pytest.raises(RuntimeError, match="provider fixture crashed"):
        runtime.forward(np.zeros((1,)))
    with pytest.raises(p.ProviderUnavailableError, match="provider fixture crashed"):
        runtime.forward(np.zeros((1,)))
    assert len(calls) == 1
    assert runtime.session is None


def test_selected_face_is_shared_by_regular_and_extended_sampling(monkeypatch):
    def face(x):
        quad = np.array([[x, 1.], [x + 2, 1.], [x + 2, 3.], [x, 3.]])
        return {"type": 0, "patches": np.repeat(quad[None], 24, axis=0), "corners": quad, "dims": [2, 2]}
    image = np.full((8, 10, 3), 50, dtype=np.uint8)
    image[:, 5:] = 180
    monkeypatch.setattr(charts_module, "_forward", lambda *args: ([face(1), face(6)], {}))
    with charts_module.detection_context({"chart_index": 1}):
        regular, _, _ = charts_module.model_extract_color_charts(image)
        extended = charts_module.model_chart_samples(image, 4)
        np.testing.assert_allclose(regular, 180)
        np.testing.assert_allclose(extended, 180 / 255)
        assert charts_module.get_detection_diagnostics()[0]["selected_chart_index"] == 1


def test_missing_selected_face_does_not_change_to_mcc(monkeypatch):
    quad = np.array([[1., 1.], [3., 1.], [3., 3.], [1., 3.]])
    face = {"type": 0, "patches": np.repeat(quad[None], 24, axis=0), "corners": quad, "dims": [2, 2]}
    monkeypatch.setattr(charts_module, "_forward", lambda *args: ([face], {}))
    with charts_module.detection_context({"chart_index": 1, "allow_mcc_fallback": True}):
        with pytest.raises(charts_module.ChartSelectionError, match="chart_index"):
            charts_module.model_chart_samples(np.zeros((8, 10, 3), dtype=np.uint8), 1)
        assert charts_module.get_detection_diagnostics()[-1]["status"] == "failed"
