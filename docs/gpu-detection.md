# GPU chart detection and CPU recovery

Chart detection defaults to `device="auto"`. This is independent of the chart
algorithm selector `backend="auto"|"model"|"mcc"`; a GPU failure retries the same
ONNX model on CPU, not a different chart detector. MCC fallback remains a separate
explicit opt-in.

| Platform | Automatic attempt order |
| --- | --- |
| Windows | CUDA, DirectML, native WebGPU/D3D12, CPU |
| Linux | CUDA, native WebGPU/Vulkan, CPU |
| macOS | native WebGPU/Metal, CoreML, CPU |

Radeon can use DirectML on Windows and Vulkan through native WebGPU. Apple
Silicon can use Metal through native WebGPU or CoreML. A CUDA-enabled runtime
and its vendor libraries must be installed to use CUDA. Missing providers,
initialization failures, failed forwards and malformed output are recorded and
can recover to CPU. No provider or driver is downloaded during inference.

```python
config = Config(chart_detection={
    "device": "auto",
    "device_id": None,
    "allow_cpu_fallback": True,
})
```

`device` also accepts `cpu`, `cuda`, `directml`, `vulkan`, `webgpu`, `metal` and
`coreml`. An explicit GPU choice attempts that route, then CPU when
`allow_cpu_fallback=True`. Set it to `False` for a strict hardware qualification.
Explicit `cpu` does not probe GPU plugins. `cpu_threads` defaults to four and
is bounded to 1–64. `chart_index` selects one detected Classic face consistently
for both ordinary and extended sampling; out-of-range selections fail.
MCC compatibility uses index zero only, and cannot silently substitute a face
when a nonzero model face was requested.

`device_id=None` selects CUDA device zero, lets Dawn choose a high-performance
graphics adapter, and selects DirectML's hardware DXGI adapter with the most
dedicated video memory. If DXGI enumeration fails, DirectML index zero is tried
with that reason recorded. An explicit integer selects a provider-specific
adapter index. DXGI, CUDA and Dawn indices are different namespaces.

## Runtime installation

The native WebGPU provider is the official `onnxruntime-ep-webgpu` plugin paired
with a compatible ONNX Runtime. It uses Dawn to execute through D3D12, Vulkan or
Metal. There is no invented `VulkanExecutionProvider`. The provider library comes
from the installed plugin, an explicit `webgpu_library` path, or
`CCP_WEBGPU_LIBRARY`. Source checkout/runtime DLL paths are never searched.

DirectML uses the `onnxruntime-directml` distribution, and CUDA uses
`onnxruntime-gpu`, rather than co-installing conflicting ONNX Runtime wheel
variants in one environment. CoreML is provided by suitable official macOS
wheels. A CPU-only installation remains usable. See the package dependency
metadata for supported platform/Python markers and use a separate environment
when qualifying an alternate runtime.

The default native WebGPU context owns one backend/adapter per process. After
initializing D3D12, Vulkan or Metal, switching its backend or adapter requires a
new process. The code reports that condition and may use CPU recovery; it never
silently relabels an existing graphics context. `clear_provider_cache()` releases
cached model sessions but does not reset Dawn's process-wide context.

## Evidence and failure semantics

Each GPU attempt requests no CPU graph fallback, disables ORT's automatic runtime
fallback and serializes forwards on shared sessions (required by DirectML).
Its first successful forward must produce matching per-node ORT profiling events
and no CPU node events. A registered provider alone does not count as execution.

Diagnostics include requested/selected device, registered and available
providers, attempted failures, graphics backend, model content hash, effective
options and node provider counts. Model-session caches include the full model
hash and provider policy/options. A replaced model at the same path cannot
silently reuse a session for old bytes.

CPU recovery is explicitly identified. CoreML profiling proves execution by
CoreML, while Apple's runtime can internally schedule CPU, GPU or Neural Engine;
the report does not claim that ORT profiling distinguishes those internal units.
WebGPU ORT registration descriptors and Dawn's physical adapter selection are
also separate; diagnostics retain both rather than claiming descriptor identity
is independent physical-adapter proof.

The recorded local hardware checks used Windows, NVIDIA RTX4070TiSUPER/IntelUHD770,
ONNX Runtime1.31.1 with WebGPU plugin0.4.0, and separate DirectML runtime1.24.4.
One reviewed photograph produced the expected Classic chart through D3D12,
Vulkan and DirectML with GPU node events and zero CPU node events. Apple Silicon,
Radeon and CUDA hardware need their own host qualification. Driver-independent
unit tests exercise those selection/failure policies; they are not hardware tests.

Primary references:

- [Native WebGPU provider and plugin APIs](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
- [Official WebGPU plugin README](https://github.com/microsoft/onnxruntime/blob/main/plugin-ep-webgpu/python/onnxruntime_ep_webgpu/README.md)
- [DirectML execution requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [CoreML options and execution scope](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
