"""GPU-first ONNX chart execution with explicit, observable CPU recovery.

ORT provider registration is not proof that it ran a graph. A GPU session is
accepted only after its first forward supplies matching per-node profile events.
The model/geometry algorithm is shared by every provider.
"""

import copy
import hashlib
import importlib
import json
import os
import platform
import shutil
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


class ProviderUnavailableError(RuntimeError):
    """No requested inference provider could complete a verified forward."""


@dataclass(frozen=True)
class ProviderPlan:
    device: str
    provider: str
    graphics_backend: Optional[str] = None


_CPU = ProviderPlan("cpu", "CPUExecutionProvider")
_CUDA = ProviderPlan("cuda", "CUDAExecutionProvider")
_DML = ProviderPlan("directml", "DmlExecutionProvider")
_COREML = ProviderPlan("coreml", "CoreMLExecutionProvider")
_registration_lock = threading.RLock()
_registered_libraries: Dict[str, str] = {}
_webgpu_context = None
_vulkan_loader = None


def _import_ort():
    return importlib.import_module("onnxruntime")


def provider_plans(device="auto", allow_cpu_fallback=True, system=None):
    """Return a platform policy; availability is checked separately per attempt."""
    system = system or platform.system()
    native_backend = "D3D12" if system == "Windows" else "Metal" if system == "Darwin" else "Vulkan"
    webgpu = ProviderPlan("webgpu", "WebGpuExecutionProvider", native_backend)
    named = {
        "cpu": _CPU, "cuda": _CUDA, "directml": _DML, "coreml": _COREML,
        "vulkan": ProviderPlan("vulkan", "WebGpuExecutionProvider", "Vulkan"),
        "metal": ProviderPlan("metal", "WebGpuExecutionProvider", "Metal"),
        "webgpu": webgpu,
    }
    if device == "auto":
        choices = [_CUDA, _DML, webgpu] if system == "Windows" else [webgpu, _COREML] if system == "Darwin" else [_CUDA, webgpu]
    elif device in named:
        choices = [named[device]]
    else:
        raise ValueError("device must be auto, cpu, cuda, directml, vulkan, webgpu, metal or coreml")
    if choices != [_CPU] and allow_cpu_fallback:
        choices.append(_CPU)
    return choices


def validate_device_options(device, device_id, allow_cpu_fallback, cpu_threads, webgpu_library):
    provider_plans(device, allow_cpu_fallback)
    if device_id is not None and (isinstance(device_id, bool) or not isinstance(device_id, int) or not 0 <= device_id <= 63):
        raise ValueError("device_id must be None (automatic) or an integer from 0 to 63")
    if not isinstance(allow_cpu_fallback, bool):
        raise ValueError("allow_cpu_fallback must be boolean")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 64:
        raise ValueError("cpu_threads must be an integer from 1 to 64")
    if webgpu_library is not None and (not isinstance(webgpu_library, (str, os.PathLike)) or not str(webgpu_library).strip()):
        raise ValueError("webgpu_library must be a nonempty path")


def _model_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def directml_adapters():
    """Enumerate bounded DXGI hardware adapters without WMI or driver copies."""
    if platform.system() != "Windows":
        return ()
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                    ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

    class Description(ctypes.Structure):
        _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
                    ("device", ctypes.c_uint32), ("subsystem", ctypes.c_uint32),
                    ("revision", ctypes.c_uint32), ("video_memory", ctypes.c_size_t),
                    ("system_memory", ctypes.c_size_t), ("shared_memory", ctypes.c_size_t),
                    ("luid", LUID), ("flags", ctypes.c_uint32)]

    def method(interface, index, result, *arguments):
        table = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])

    def release(interface):
        if interface:
            method(interface, 2, ctypes.c_uint32)(interface)

    factory = ctypes.c_void_p()
    library = ctypes.WinDLL("dxgi.dll", winmode=0x800)
    create = library.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int32
    iid = GUID.from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    status = create(ctypes.byref(iid), ctypes.byref(factory))
    if status < 0:
        raise ProviderUnavailableError("CreateDXGIFactory1 failed: 0x%08x" % (status & 0xffffffff))
    results = []
    try:
        enumerate_adapter = method(factory, 12, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            status = enumerate_adapter(factory, index, ctypes.byref(adapter))
            if (status & 0xffffffff) == 0x887a0002:  # DXGI_ERROR_NOT_FOUND
                break
            if status < 0:
                raise ProviderUnavailableError("DXGI EnumAdapters1 failed: 0x%08x" % (status & 0xffffffff))
            try:
                description = Description()
                status = method(adapter, 10, ctypes.c_int32, ctypes.POINTER(Description))(adapter, ctypes.byref(description))
                if status < 0:
                    raise ProviderUnavailableError("DXGI GetDesc1 failed")
                if not description.flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE
                    results.append({"index": index, "description": description.description,
                                    "vendor_id": description.vendor, "device_id": description.device,
                                    "dedicated_video_bytes": description.video_memory})
            finally:
                release(adapter)
    finally:
        release(factory)
    return tuple(results)


def _directml_selection(device_id):
    if device_id is not None:
        return device_id, None
    try:
        adapters = directml_adapters()
        chosen = max(adapters, key=lambda item: item["dedicated_video_bytes"]) if adapters else None
        return (chosen["index"], dict(chosen)) if chosen else (0, {"index": 0, "reason": "DXGI reported no hardware descriptors"})
    except Exception as exc:
        return 0, {"index": 0, "enumeration_error": str(exc)}


def _plugin_path(explicit=None):
    configured = explicit or os.environ.get("CCP_WEBGPU_LIBRARY")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        try:
            plugin = importlib.import_module("onnxruntime_ep_webgpu")
            path = Path(plugin.get_library_path()).resolve()
        except ImportError as exc:
            raise ProviderUnavailableError("WebGPU plugin is not installed (onnxruntime-ep-webgpu)") from exc
    if not path.is_file():
        raise ProviderUnavailableError("WebGPU provider library does not exist: %s" % path)
    return path


def _device_details(device):
    hardware = device.device
    return {
        "type": str(hardware.type), "vendor": hardware.vendor,
        "device_id": hardware.device_id, "metadata": dict(hardware.metadata),
    }


def _prepare_webgpu(ort, settings, plan, device_id, explicit_library):
    global _webgpu_context, _vulkan_loader
    system = platform.system()
    if plan.graphics_backend == "Metal" and system != "Darwin":
        raise ProviderUnavailableError("Metal requires macOS")
    if plan.graphics_backend == "D3D12" and system != "Windows":
        raise ProviderUnavailableError("D3D12 requires Windows")
    if plan.graphics_backend == "Vulkan" and system == "Darwin":
        raise ProviderUnavailableError("native macOS WebGPU uses Metal; select metal or coreml")
    if not all(hasattr(ort, name) for name in ("register_execution_provider_library", "get_ep_devices")) or not hasattr(settings, "add_provider_for_devices"):
        raise ProviderUnavailableError("WebGPU requires ONNX Runtime plugin execution-provider APIs")
    library = _plugin_path(explicit_library)
    with _registration_lock:
        if system == "Windows" and plan.graphics_backend == "Vulkan" and _vulkan_loader is None:
            import ctypes
            # Search the installed system loader, not the application/CWD.
            _vulkan_loader = ctypes.WinDLL("vulkan-1.dll", winmode=0x800)
        key = str(library)
        if key not in _registered_libraries:
            registration = "ccp_chart_webgpu_" + hashlib.sha256(key.encode()).hexdigest()[:12]
            ort.register_execution_provider_library(registration, key)
            _registered_libraries[key] = registration
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"
                   and "GPU" in str(d.device.type).upper()
                   and not any("virtual" in str(k).lower() and str(v) == "1" for k, v in dict(d.device.metadata).items())]
        if not devices:
            raise ProviderUnavailableError("WebGPU reported no non-virtual GPU device")
        signature = (key, plan.graphics_backend, device_id)
        if _webgpu_context is not None and _webgpu_context != signature:
            raise ProviderUnavailableError("WebGPU backend/adapter already initialized; restart the process to change it")
        # A failed initializer can also leave a native context alive. Never reuse
        # it with different graphics backend/device settings in this process.
        _webgpu_context = signature
        # ORT's registration descriptor and Dawn's physical adapter enumeration
        # are separate. Prefer a discrete descriptor; adapterIndex controls Dawn.
        devices.sort(key=lambda d: dict(d.device.metadata).get("Discrete", "0") != "1")
        options = {"enableGraphCapture": "0", "powerPreference": "high-performance", "deviceId": "0"}
        if device_id is not None:
            options["adapterIndex"] = str(device_id)
        if plan.graphics_backend in ("D3D12", "Vulkan"):
            options["dawnBackendType"] = plan.graphics_backend
        settings.add_provider_for_devices([devices[0]], options)
        return {"library": key, "options": options,
                "enumerated_gpu_devices": [_device_details(d) for d in devices],
                "adapter_selection": "Dawn high-performance automatic selection or explicit adapterIndex; ORT descriptor is not independent adapter proof"}


def _profile_execution(session, profile_directory, expected_provider):
    try:
        filename = session.end_profiling()
        data = json.loads(Path(filename).read_text(encoding="utf8"))
        counts = Counter(event.get("args", {}).get("provider") for event in data
                         if event.get("cat") == "Node" and event.get("args", {}).get("provider"))
        if counts.get(expected_provider, 0) == 0:
            raise ProviderUnavailableError("profiling did not observe execution by %s" % expected_provider)
        if counts.get("CPUExecutionProvider", 0):
            raise ProviderUnavailableError("GPU attempt executed CPU nodes despite the no-fallback policy")
        return dict(counts)
    finally:
        shutil.rmtree(profile_directory, ignore_errors=True)


@dataclass
class _Session:
    session: Any
    input_name: str
    input_size: int
    plan: ProviderPlan
    details: Dict[str, Any]
    profile_directory: Optional[str] = None
    execution_counts: Optional[dict] = None
    unusable_reason: Optional[str] = None
    lock: Any = field(default_factory=threading.RLock)

    def invalidate(self, error):
        """Release a failed session before another provider is attempted."""
        with self.lock:
            self.unusable_reason = str(error)
            if self.profile_directory is not None:
                try:
                    self.session.end_profiling()
                except Exception:
                    pass
                shutil.rmtree(self.profile_directory, ignore_errors=True)
                self.profile_directory = None
            self.session = None

    def forward(self, tensor):
        # DirectML forbids concurrent Run calls on one session. Serialization also
        # makes first-forward profiling and runtime fallback deterministic.
        with self.lock:
            if self.unusable_reason:
                raise ProviderUnavailableError(self.unusable_reason)
            try:
                output = self.session.run(None, {self.input_name: tensor})
                if self.plan.device != "cpu" and self.execution_counts is None:
                    directory, self.profile_directory = self.profile_directory, None
                    self.execution_counts = _profile_execution(self.session, directory, self.plan.provider)
                return output
            except Exception as exc:
                self.invalidate(exc)
                raise


@lru_cache(maxsize=4)
def _load_session(path, digest, plan, device_id, cpu_threads, webgpu_library, runtime_version):
    ort = _import_ort()
    available = list(ort.get_available_providers())
    if plan.provider != "WebGpuExecutionProvider" and plan.provider not in available:
        raise ProviderUnavailableError("%s is not advertised by installed ONNX Runtime" % plan.provider)
    if plan.device == "directml" and platform.system() != "Windows":
        raise ProviderUnavailableError("DirectML requires Windows")
    if plan.device == "coreml" and platform.system() != "Darwin":
        raise ProviderUnavailableError("CoreML requires macOS")
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = cpu_threads
    settings.inter_op_num_threads = 1
    details = {"requested_provider": plan.provider, "graphics_backend": plan.graphics_backend,
               "device_id": device_id, "runtime_version": runtime_version}
    profile_directory = None
    try:
        if plan.device != "cpu":
            settings.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            profile_directory = tempfile.mkdtemp(prefix="ccp-chart-profile-")
            settings.enable_profiling = True
            settings.profile_file_prefix = str(Path(profile_directory) / "execution")
        provider_options = {}
        if plan.device == "directml":
            settings.enable_mem_pattern = False
            settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            selected_id, selection = _directml_selection(device_id)
            if selection is not None:
                details["automatic_adapter"] = selection
            details["device_id"] = selected_id
            provider_options = {"device_id": str(selected_id)}
        elif plan.device == "cuda":
            details["device_id"] = 0 if device_id is None else device_id
            provider_options = {"device_id": str(details["device_id"])}
        elif plan.device == "coreml":
            provider_options = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
            details["execution_scope"] = "CoreML provider observed; internal Apple GPU/ANE/CPU scheduling is managed by CoreML"
        if plan.provider == "WebGpuExecutionProvider":
            details.update(_prepare_webgpu(ort, settings, plan, device_id, webgpu_library))
            # A V2 device provider has already been appended to SessionOptions.
            session = ort.InferenceSession(path, sess_options=settings)
        else:
            providers = [(plan.provider, provider_options)] if provider_options else [plan.provider]
            session = ort.InferenceSession(path, sess_options=settings, providers=providers)
        if hasattr(session, "disable_fallback"):
            session.disable_fallback()
        registered = list(session.get_providers())
        if plan.provider not in registered:
            raise ProviderUnavailableError("requested %s, but session registered %s" % (plan.provider, registered))
        details["registered_providers"] = registered
        details["provider_options"] = provider_options
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("expected one float32 image input")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or shape[2] != shape[3] or shape[2] not in (1280, 1536):
            raise ValueError("expected input shape [1,3,1280,1280] or [1,3,1536,1536]")
        return _Session(session, inputs[0].name, int(shape[2]), plan, details, profile_directory)
    except Exception:
        if profile_directory:
            shutil.rmtree(profile_directory, ignore_errors=True)
        raise


def clear_provider_cache():
    """Release cached sessions; WebGPU backend/adapter still needs process restart."""
    _load_session.cache_clear()
    directml_adapters.cache_clear()


def infer_model(path, options, prepare_input, validate_output=None):
    """Run the first successful requested GPU, then CPU when policy permits.

    ``prepare_input(size)`` returns ``(tensor, geometry_metadata)``. Geometry and
    color interpretation remain outside the provider layer.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ProviderUnavailableError("chart model does not exist: %s" % path)
    ort = _import_ort()
    digest = _model_digest(path)
    version = str(getattr(ort, "__version__", "unknown"))
    library = options.webgpu_library or os.environ.get("CCP_WEBGPU_LIBRARY")
    if library:
        library = str(Path(library).expanduser().resolve())
    attempts = []
    for plan in provider_plans(options.device, options.allow_cpu_fallback):
        runtime = None
        try:
            runtime = _load_session(str(path), digest, plan, options.device_id,
                                    options.cpu_threads, library, version)
            tensor, geometry = prepare_input(runtime.input_size)
            output = runtime.forward(tensor)
            if validate_output is not None:
                validate_output(output)
            attempts.append({"device": plan.device, "provider": plan.provider, "status": "succeeded"})
            details = copy.deepcopy(runtime.details)
            details.update({"requested_device": options.device, "selected_device": plan.device,
                            "provider": plan.provider, "sha256": digest,
                            "model": str(path), "input_size": runtime.input_size,
                            "available_providers": list(ort.get_available_providers()),
                            "attempts": attempts, "cpu_fallback": plan.device == "cpu" and options.device != "cpu",
                            "execution_verified": True,
                            "execution_evidence": "node_profile" if plan.device != "cpu" else "explicit_cpu_only_session",
                            "node_provider_counts": copy.deepcopy(runtime.execution_counts or {})})
            return output, geometry, details
        except Exception as exc:
            if runtime is not None and hasattr(runtime, "invalidate"):
                runtime.invalidate(exc)
            attempts.append({"device": plan.device, "provider": plan.provider,
                             "status": "failed", "type": type(exc).__name__, "reason": str(exc)})
    error = ProviderUnavailableError("No usable chart execution provider: " + "; ".join(
        "%s: %s" % (attempt["device"], attempt["reason"]) for attempt in attempts))
    error.attempts = attempts
    raise error
