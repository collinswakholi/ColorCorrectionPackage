"""GPU-first ONNX inference for the Opt2 oriented chart-anchor model.

The model layout, letterbox padding, ProbIoU and anchor identities follow the
maintainer-supplied ColorCorrect Studio detector. The assembly is conservative:
it never remaps another chart type to Classic, and requires valid measured
anchor geometry before sampling. No native engine, automatic download or
training package is needed. ONNX Runtime is a package dependency.
"""

import copy
import itertools
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .providers import infer_model, validate_device_options

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "chart_detector_1280.onnx"

class ChartDetectionError(RuntimeError):
    """Detection could not supply a valid chart for fitting."""


class ModelUnavailableError(ChartDetectionError):
    """The requested model, runtime, or model contract is unavailable."""


class NoChartDetectedError(ChartDetectionError):
    """Inference ran, but no valid chart lattice was found."""


class UnsupportedChartError(ChartDetectionError):
    """A chart was detected whose reference layout is unsupported."""


class ChartSelectionError(ChartDetectionError):
    """The requested face is unavailable; another algorithm cannot substitute it."""


@dataclass(frozen=True)
class DetectionOptions:
    backend: str = "auto"
    model_path: Optional[str] = None
    recall_model_path: Optional[str] = None
    confidence: float = 0.15
    iou: float = 0.7
    sampling_inset: float = 0.75
    allow_mcc_fallback: bool = False
    max_detections: int = 300
    chart_index: int = 0
    device: str = "auto"
    device_id: Optional[int] = None
    allow_cpu_fallback: bool = True
    cpu_threads: int = 4
    webgpu_library: Optional[str] = None

    def __post_init__(self):
        validate_device_options(self.device, self.device_id, self.allow_cpu_fallback,
                                self.cpu_threads, self.webgpu_library)
        if isinstance(self.chart_index, bool) or not isinstance(self.chart_index, int) or not 0 <= self.chart_index <= 63:
            raise ValueError("chart_index must be an integer from 0 to 63")
        if self.backend not in ("auto", "model", "mcc"):
            raise ValueError("chart detection backend must be auto, model, or mcc")
        if self.backend == "mcc" and self.chart_index != 0:
            raise ValueError("the MCC compatibility backend supports chart_index=0 only")
        for key in ("model_path", "recall_model_path"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, (str, os.PathLike)) or not str(value).strip()):
                raise ValueError("chart detection %s must be a nonempty path" % key)
        for key in ("confidence", "iou", "sampling_inset"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("chart detection %s must be finite in (0, 1]" % key)
        if not isinstance(self.allow_mcc_fallback, bool):
            raise ValueError("allow_mcc_fallback must be boolean")
        if isinstance(self.max_detections, bool) or not isinstance(self.max_detections, int) or not 4 <= self.max_detections <= 1000:
            raise ValueError("max_detections must be an integer from 4 to 1000")

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**copy.deepcopy(value))
        raise TypeError("chart detection options must be DetectionOptions or a dictionary")


@dataclass
class _Scope:
    options: DetectionOptions
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    geometry: Any = None
    shape: Any = None
    fallback: bool = False
    source_image: Any = None


_scope = ContextVar("ccp_chart_detection_scope", default=None)
_last_diagnostics = ContextVar("ccp_chart_detection_diagnostics", default=())


@contextmanager
def detection_context(options=None):
    """Scope settings and fixed model geometry to ONE source-image run.

Subsequent stage images resample the first successful chart geometry. Start a
new context for every independent image. Contexts are isolated across threads
and asynchronous tasks; nested scopes restore their parent's settings.
"""
    state = _Scope(DetectionOptions.from_value(options))
    token = _scope.set(state)
    try:
        yield state
    finally:
        _last_diagnostics.set(copy.deepcopy(state.diagnostics))
        _scope.reset(token)


def get_detection_diagnostics():
    state = _scope.get()
    return copy.deepcopy(state.diagnostics if state is not None else _last_diagnostics.get())


def bind_detection_source(image):
    """Choose geometry from original BGR pixels, independent of metric switches.

    The pipeline calls this once after loading its owned source pixels. Direct
    utility contexts may omit it and detect on their first sampled image.
    """
    state = _scope.get()
    if state is None:
        raise RuntimeError("bind_detection_source requires a detection_context")
    if state.source_image is not None or state.geometry is not None:
        raise RuntimeError("a detection context already owns a source image")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source geometry expects HxWx3 uint8 BGR")
    state.source_image = image.copy()


def _record(state, **values):
    state.diagnostics.append(values)
    _last_diagnostics.set(copy.deepcopy(state.diagnostics))


def _options_and_path(state):
    options = state.options
    configured = options.model_path or os.environ.get("CCP_CHART_MODEL")
    if not configured:
        configured = str(DEFAULT_MODEL_PATH)
    if options.backend == "mcc" or state.fallback:
        return options, None
    return options, Path(configured).expanduser().resolve()



def _letterbox(image, size):
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    # The Rust producer rounds positive .5 up, rather than Python's ties-to-even.
    nw, nh = int(math.floor(w * gain + 0.5)), int(math.floor(h * gain + 0.5))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image[:, :, ::-1], (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, gain, left, top


def _rows(output):
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2 or 29 not in output.shape:
        raise ModelUnavailableError("malformed chart output: expected [1,29,N] or [1,N,29]")
    if output.shape[0] == 29:
        output = output.T
    if output.shape[1] != 29 or output.shape[0] == 0 or not np.isfinite(output).all():
        raise ModelUnavailableError("chart output has invalid dimensions or nonfinite values")
    return np.asarray(output, dtype=np.float64)


def _covariance(boxes):
    aa, bb = boxes[:, 2] ** 2 / 12.0, boxes[:, 3] ** 2 / 12.0
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    return aa * c * c + bb * s * s, aa * s * s + bb * c * c, (aa - bb) * c * s


def _probiou(one, others):
    eps = 1e-7
    a1, b1, c1 = _covariance(one[None])
    a2, b2, c2 = _covariance(others)
    dx, dy = one[0] - others[:, 0], one[1] - others[:, 1]
    den = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) / den * .25
    t2 = (c1 + c2) * (-dx) * dy / den * .5
    det1, det2 = np.maximum(a1 * b1 - c1 * c1, 0), np.maximum(a2 * b2 - c2 * c2, 0)
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (4 * np.sqrt(det1 * det2) + eps) + eps) * .5
    bd = np.clip(t1 + t2 + t3, eps, 100)
    return 1 - np.sqrt(np.maximum(1 - np.exp(-bd) + eps, 0))


def _decode(output, options, gain=1.0, left=0, top=0):
    rows = _rows(output)
    classes = np.argmax(rows[:, 4:28], axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    indices = np.flatnonzero((scores >= options.confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0))
    indices = indices[np.argsort(-scores[indices], kind="stable")[:30000]]
    boxes = np.column_stack((rows[:, :4], rows[:, 28]))
    kept = []
    while len(indices) and len(kept) < options.max_detections:
        i = indices[0]
        kept.append(i)
        rest = indices[1:]
        same = classes[rest] == classes[i]
        suppressed = np.zeros(len(rest), dtype=bool)
        if same.any():
            suppressed[same] = _probiou(boxes[i], boxes[rest[same]]) > options.iou
        indices = rest[~suppressed]
    detections = []
    for i in kept:
        box = boxes[i].copy()
        box[:2] = (box[:2] - [left, top]) / gain
        box[2:4] /= gain
        if any(np.linalg.norm(box[:2] - det["center"]) < 3 for det in detections):
            continue
        detections.append({"type": int(classes[i]) // 8, "anchor": int(classes[i]) % 8,
                           "score": float(scores[i]), "center": box[:2],
                           "diag": float(np.hypot(box[2], box[3]))})
    return detections


_LAYOUTS = ((4, 6), (4, 6), (10, 14))
_TYPE_NAMES = ("classic_4x6", "spyder_4x6", "digital_sg_10x14")


def _anchor_points(rows, cols):
    cells = ((0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0),
             (0, cols // 2), (rows - 1, cols // 2), (rows // 2, 0), (rows // 2, cols // 2))
    return np.array([((c + .5) / cols, (r + .5) / rows) for r, c in cells], dtype=np.float64)


def _warp(points, homography):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), homography).reshape(-1, 2)


def _aabb_iou(a, b):
    amin, amax, bmin, bmax = a.min(axis=0), a.max(axis=0), b.min(axis=0), b.max(axis=0)
    inter = np.maximum(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0).prod()
    union = (amax - amin).prod() + (bmax - bmin).prod() - inter
    return float(inter / union) if union > 0 else 0


def _assemble_one(cluster, chart_type, inset, width, height):
    rows, cols = _LAYOUTS[chart_type]
    best = {}
    for det in cluster:
        if det["anchor"] not in best or det["score"] > best[det["anchor"]]["score"]:
            best[det["anchor"]] = det
    if len(best) < 4:
        return None
    anchors = _anchor_points(rows, cols)
    # All four semantic corners give orientation; otherwise choose a stable,
    # nondegenerate four-anchor subset. Never sort cells by visual position.
    choices = [(0, 1, 2, 3)] if all(i in best for i in range(4)) else itertools.combinations(sorted(best), 4)
    homography = None
    for ids in choices:
        src = anchors[list(ids)]
        if abs(cv2.contourArea(cv2.convexHull(src.astype(np.float32)))) < .03:
            continue
        dst = np.array([best[i]["center"] for i in ids], dtype=np.float32)
        candidate = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        if not np.isfinite(candidate).all() or abs(np.linalg.det(candidate)) < 1e-10:
            continue
        errors = np.linalg.norm(_warp(anchors[list(best)], candidate) - np.array([d["center"] for d in best.values()]), axis=1)
        if homography is None or np.median(errors) < best_error:
            homography, best_error = candidate, np.median(errors)
    if homography is None:
        return None
    denominators = np.array([0., homography[2, 0], homography[2, 0] + homography[2, 1], homography[2, 1]]) + homography[2, 2]
    if np.abs(denominators).min() < 1e-8 or (denominators > 0).any() != (denominators > 0).all():
        return None
    corners = _warp(((0, 0), (1, 0), (1, 1), (0, 1)), homography)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)):
        return None
    area = abs(cv2.contourArea(corners.astype(np.float32)))
    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if area < 64 or edges.min() <= 1e-6:
        return None
    if min(edges[0], edges[2]) / max(edges[0], edges[2]) < .62 or min(edges[1], edges[3]) / max(edges[1], edges[3]) < .62:
        return None
    cell = .25 * ((edges[0] + edges[2]) / cols + (edges[1] + edges[3]) / rows)
    if best_error > .45 * cell or np.linalg.norm(corners[0] - corners[2]) >= 4 * max(width, height):
        return None
    expected = np.mean([max(d["diag"], 1) for d in best.values()]) * np.hypot(rows, cols) / .7
    if not .2 * expected <= np.linalg.norm(corners[0] - corners[2]) <= 2.2 * expected:
        return None
    patches = []
    for row in range(rows):
        for col in range(cols):
            cell_quad = _warp(((col / cols, row / rows), ((col + 1) / cols, row / rows),
                               ((col + 1) / cols, (row + 1) / rows), (col / cols, (row + 1) / rows)), homography)
            center = cell_quad.mean(axis=0)
            patches.append(center + (cell_quad - center) * inset)
    patches = np.asarray(patches)
    if not np.isfinite(patches).all() or (patches[:, :, 0] < 0).any() or (patches[:, :, 0] > width - 1).any() or (patches[:, :, 1] < 0).any() or (patches[:, :, 1] > height - 1).any():
        return None
    return {"type": chart_type, "score": float(np.mean([d["score"] for d in best.values()])),
            "corners": corners, "patches": patches, "dims": [float((edges[0] + edges[2]) / (2 * cols)), float((edges[1] + edges[3]) / (2 * rows))]}


def _assemble(detections, options, width, height):
    charts = []
    for chart_type, (rows, cols) in enumerate(_LAYOUTS):
        remaining = [d for d in detections if d["type"] == chart_type]
        while len(remaining) >= 4:
            mids = [d for d in remaining if d["anchor"] >= 4]
            seed = max(mids or remaining, key=lambda d: d["score"])
            radius = .55 * max(seed["diag"], 1) * np.hypot(rows, cols) / .7
            # Avoid joining two faces of the same type when repeated semantic
            # anchors reveal a separation smaller than the broad radius.
            separations = [np.linalg.norm(a["center"] - b["center"]) for a, b in itertools.combinations(remaining, 2) if a["anchor"] == b["anchor"]]
            duplicate_names = len({a["anchor"] for a, b in itertools.combinations(remaining, 2) if a["anchor"] == b["anchor"]})
            if duplicate_names >= 3 and separations:
                separation = np.median(separations)
                if separation > 2 * np.median([max(d["diag"], 1) for d in remaining]):
                    radius = min(radius, .82 * separation)
            def bucket_at(center):
                bucket, ranks = {}, {}
                for det in remaining:
                    distance = np.linalg.norm(det["center"] - center)
                    if distance > radius:
                        continue
                    score = det["score"]
                    rank = score ** 2 / (distance + max(.35 * radius, 1)) + .15 * score if det["anchor"] < 4 else score / (distance + max(.35 * radius, 1)) - .05 * distance / max(radius, 1)
                    if det["anchor"] not in ranks or rank > ranks[det["anchor"]]:
                        bucket[det["anchor"]], ranks[det["anchor"]] = det, rank
                return bucket
            bucket = bucket_at(seed["center"])
            if len(bucket) >= 3:
                corner_dets = [d for i, d in bucket.items() if i < 4]
                center = np.mean([d["center"] for d in (corner_dets if len(corner_dets) >= 2 else bucket.values())], axis=0)
                refined = bucket_at(center)
                for i, det in bucket.items():
                    if i not in refined or (det["score"] >= .85 and det["score"] > refined[i]["score"] + .12):
                        refined[i] = det
                bucket = refined
            chart = _assemble_one(list(bucket.values()), chart_type, options.sampling_inset, width, height)
            if chart is not None:
                charts.append(chart)
                used = {id(d) for d in bucket.values()}
                remaining = [d for d in remaining if id(d) not in used]
            else:
                remaining = [d for d in remaining if d is not seed]
    kept = []
    for chart in sorted(charts, key=lambda c: -c["score"]):
        if not any(_aabb_iou(chart["corners"], old["corners"]) >= .35 for old in kept):
            kept.append(chart)
    return kept


def _forward(image, path, options):
    def prepare(size):
        tensor, gain, left, top = _letterbox(image, size)
        return tensor, (gain, left, top)

    def validate(outputs):
        if len(outputs) != 1:
            raise ModelUnavailableError("expected one oriented-anchor model output")
        _rows(outputs[0])

    details = None
    try:
        outputs, (gain, left, top), details = infer_model(path, options, prepare, validate)
        detections = _decode(outputs[0], options, gain, left, top)
    except Exception as exc:
        error = ModelUnavailableError("chart inference failed: %s" % exc)
        error.provider_attempts = getattr(exc, "attempts", [])
        error.inference_details = copy.deepcopy(details or {})
        error.forward_completed = details is not None
        raise error from exc
    h, w = image.shape[:2]
    details["anchors"] = len(detections)
    return _assemble(detections, options, w, h), details


def _model_charts(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("chart detection expects a nonempty HxWx3 uint8 BGR image")
    state = _scope.get() or _Scope(DetectionOptions())
    inference_details = {}
    forward_completed = False
    try:
        options, path = _options_and_path(state)
        if path is None:
            if options.chart_index != 0:
                raise ChartSelectionError("MCC fallback cannot preserve a nonzero selected chart_index")
            if not state.diagnostics:
                _record(state, backend="mcc", reason="explicit MCC" if options.backend == "mcc" else "explicit fallback after model failure")
            return None
        if state.geometry is not None:
            if image.shape[:2] != state.shape:
                raise ChartDetectionError("stage image dimensions changed within a chart detection context")
            return state.geometry
        source = state.source_image if state.source_image is not None else image
        if source.shape[:2] != image.shape[:2]:
            raise ChartDetectionError("stage image dimensions differ from the bound source image")
        charts, details = _forward(source, path, options)
        inference_details = copy.deepcopy(details)
        forward_completed = True
        if not charts and options.recall_model_path:
            charts, details = _forward(source, Path(options.recall_model_path).expanduser().resolve(), options)
            details["recall"] = True
            inference_details = copy.deepcopy(details)
            forward_completed = True
        if not charts:
            raise NoChartDetectedError("model inference found no valid chart lattice")
        classic = [chart for chart in charts if chart["type"] == 0]
        if not classic:
            raise UnsupportedChartError("detected %s; this pipeline currently requires Classic 4x6 references" % ", ".join(sorted({_TYPE_NAMES[c["type"]] for c in charts})))
        if options.chart_index >= len(classic):
            raise ChartSelectionError("requested chart_index %d but only %d Classic faces were detected" % (options.chart_index, len(classic)))
        _record(state, backend="model", status="detected", charts=len(classic), selected_chart_index=options.chart_index, geometry_source="source_image" if state.source_image is not None else "first_sampled_image", ignored_layouts=[_TYPE_NAMES[c["type"]] for c in charts if c["type"] != 0], **details)
        state.geometry, state.shape = classic, image.shape[:2]
        return classic
    except ChartDetectionError as exc:
        failure_details = copy.deepcopy(getattr(exc, "inference_details", None) or inference_details)
        failure_details.update(backend="model", status="failed", reason=str(exc),
                               error_type=type(exc).__name__, forward_completed=bool(forward_completed or getattr(exc, "forward_completed", False)),
                               provider_attempts=getattr(exc, "provider_attempts", []))
        _record(state, **failure_details)
        if state.options.allow_mcc_fallback and state.options.chart_index == 0 and not isinstance(exc, ChartSelectionError):
            import warnings
            warnings.warn("model chart detection failed; explicit MCC fallback: %s" % exc, RuntimeWarning, stacklevel=3)
            state.fallback = True
            _record(state, backend="mcc", status="fallback", reason=str(exc))
            return None
        raise


def _sample(image, patches, count):
    state = _scope.get()
    measurement = getattr(state, "measurement_image", None)
    if measurement is not None:
        if measurement.shape != image.shape:
            raise ChartDetectionError("measurement image dimensions differ from detector coordinates")
        from ..sampling import sample_quadrilaterals
        return sample_quadrilaterals(measurement, patches, count) * 255.0
    side = int(math.ceil(math.sqrt(count)))
    axis = np.linspace(0, 1, side) if side > 1 else np.array([.5])
    uu, vv = np.meshgrid(axis, axis)
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    if len(uv) > count:
        uv = uv[np.linspace(0, len(uv) - 1, count).round().astype(int)]
    rgb = image[:, :, ::-1].astype(np.float32)
    sampled = []
    for q in patches:
        u, v = uv[:, 0:1], uv[:, 1:2]
        points = (1 - v) * ((1 - u) * q[0] + u * q[1]) + v * ((1 - u) * q[3] + u * q[2])
        values = cv2.remap(rgb, points[:, 0].astype(np.float32).reshape(1, -1), points[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
        sampled.append(values.reshape(-1, 3).astype(np.float64))
    return np.asarray(sampled)


def model_extract_color_charts(image, n_charts=1):
    """Return legacy RGB [0,255] samples/drawing/dimensions, or None for MCC."""
    if isinstance(n_charts, bool) or not isinstance(n_charts, int) or n_charts < 1:
        raise ValueError("n_charts must be a positive integer")
    charts = _model_charts(image)
    if charts is None:
        return None
    index = (_scope.get().options if _scope.get() is not None else DetectionOptions()).chart_index
    chosen = charts[index:index + n_charts]
    drawing = image.copy()
    samples = []
    for chart in chosen:
        samples.append(_sample(image, chart["patches"], 81).mean(axis=1))
        cv2.polylines(drawing, [np.round(q).astype(np.int32) for q in chart["patches"]], True, (0, 255, 0), 1)
    return samples, drawing, chosen[0]["dims"]


def model_chart_samples(image, samples_per_patch):
    """Return deterministic normalized RGB samples in cell-major order, or None for MCC."""
    if isinstance(samples_per_patch, bool) or not isinstance(samples_per_patch, int) or not 1 <= samples_per_patch <= 4096:
        raise ValueError("model samples_per_patch must be an integer from 1 to 4096")
    charts = _model_charts(image)
    if charts is None:
        return None
    index = (_scope.get().options if _scope.get() is not None else DetectionOptions()).chart_index
    return _sample(image, charts[index]["patches"], samples_per_patch).reshape(-1, 3) / 255.0
