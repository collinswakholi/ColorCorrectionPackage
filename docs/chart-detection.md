# Model-based Classic24 detection

The default `auto` detector uses the bundled Opt2 1280 ONNX model. It runs with
ONNX Runtime with GPU-first provider selection and CPU fallback; it does not
require the Studio native engine, PyTorch for detection, or a model download. Model identity and notices
are recorded beside the bundled model. This is a different detection/sampling
algorithm from OpenCV MCC, so extracted patch values can change.
See [GPU selection](gpu-detection.md) for CUDA, DirectML, Vulkan, Metal/CoreML,
runtime requirements and verified-versus-unqualified hardware cases. Set
`device="cpu"` inside `chart_detection` to reproduce the earlier CPU detector.

`Config(chart_detection={...})` accepts `backend` (`auto`, `model`, or `mcc`),
`model_path`, optional `recall_model_path`, `confidence` (default .15), `iou`
(.7), `sampling_inset` (.75), `max_detections` (300), and
`allow_mcc_fallback` (false).
GPU options and `chart_index` are documented separately. CPU recovery retains
the same ONNX detector and is independent of optional MCC compatibility.

Model lookup precedence is the explicit `model_path`, `CCP_CHART_MODEL`, then
`ColorCorrectionPipeline/detection/models/chart_detector_1280.onnx`. No sibling
checkout is searched and no remote asset is downloaded. A missing bundled or
configured model raises `ModelUnavailableError` in both `auto` and `model`
modes. A broken installation cannot silently change detector algorithms.

Use `backend="mcc"` to reproduce the legacy detector. Opt in to
`allow_mcc_fallback=True` only when accepting a different detector on model
failure; a warning and diagnostic entry identify that fallback. Missing runtime,
malformed model output, no valid lattice, and unsupported chart layouts have
distinct errors. Model loading is for trusted ONNX assets.

Only Classic 4x6 references are supported by the Python calibration pipeline.
The model recognizes anchor classes for Classic, Spyder and Digital SG. The
latter two are never relabeled as Classic; an image with only unsupported
layouts fails clearly. Passport Classic faces share the Classic model layout.
Multi-face extraction returns Classic faces by confidence.

The model has 24 **anchor classes** (three layouts times eight named anchors).
Those classes are not the chart's 24 color cells. The decoder accepts the
specified 29-channel OBB output, performs class-aware oriented suppression,
assembles a homography from semantic anchor identities, and samples cells in
canonical row-major order. Rotation changes image coordinates, not cell IDs.
Degenerate, inconsistent, or out-of-frame sampling geometry is rejected.

One pipeline run owns one `detection_context`. The pipeline binds the original
source pixels as the geometry source, independently of optional metric switches.
Model geometry is detected once, then the same regions are resampled after correction stages. A context must not
be reused for a different source image; changed stage dimensions fail. Direct
utility contexts detect on their first sampled image unless they explicitly call
`bind_detection_source`. Utility users can use the same context for one image's stages:

```python
from ColorCorrectionPipeline.detection import detection_context, get_detection_diagnostics
from ColorCorrectionPipeline.core.utils import extract_color_chart

with detection_context({"backend": "model"}):
    rgb255, overlay, dimensions = extract_color_chart(bgr_uint8, get_patch_size=True)
    diagnostics = get_detection_diagnostics()
```

The legacy extraction boundary remains RGB values in `[0,255]`; the internal
extended-sample helper returns normalized RGB and repeats reference rows in the
same cell-major order. Sampling uses deterministic bilinear grids inside the
inset quadrilaterals. It does not imply independent color labels for repeated
pixels and does not establish physical color accuracy.

The Python assembly intentionally avoids the Studio detector's class-remapping
fallback. It uses conservative geometric checks and may reject cases accepted
by that fallback. Photograph integration checks should record image/model
hashes, chart counts, rotations, sampling shapes, timings and detector errors;
synthetic tests alone do not establish photographed-detector accuracy.
