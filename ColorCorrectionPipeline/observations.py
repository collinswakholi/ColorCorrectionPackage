"""Adapt fixed detector geometry to immutable, reference-assigned observations."""

import numpy as np

from .contracts import ChartObservation, ObservationSet
from .sampling import sample_quadrilaterals


def bind_stage_measurements(rgb=None):
    """Use full-precision encoded sRGB for explicit-mode model sampling."""
    from .detection.charts import _scope
    state = _scope.get()
    if state is None:
        raise RuntimeError("stage measurements require a detection context")
    state.measurement_image = None if rgb is None else np.array(rgb, dtype=np.float64, copy=True)


def capture_observations(rgb, image_data, reference, stage, *, explicit=False, force=False):
    """Resample all available faces without changing their within-run identities."""
    from .detection import ChartDetectionError
    from .detection.charts import _scope, _model_charts, _sample
    state = _scope.get()
    if state is None:
        raise RuntimeError("observations require a detection context")
    bgr8 = np.clip(np.asarray(rgb)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
    if state.geometry is None and force:
        try:
            _model_charts(bgr8)
        except ChartDetectionError as exc:
            return ObservationSet((), image_data.digest, unavailable_reason=type(exc).__name__ + ": " + str(exc))
    if state.geometry is None:
        return ObservationSet((), image_data.digest, unavailable_reason="model geometry unavailable; no observations invented")
    faces = []
    last_detection = next((item for item in reversed(state.diagnostics) if item.get("status") == "detected"), {})
    detector_provenance = {key: last_detection[key] for key in (
        "sha256", "provider", "runtime_version", "graphics_backend", "cpu_fallback",
        "execution_verified", "execution_evidence", "node_provider_counts", "geometry_source") if key in last_detection}
    for index, geometry in enumerate(state.geometry):
        if explicit:
            samples = sample_quadrilaterals(rgb, geometry["patches"], 81)
        else:
            samples = _sample(bgr8, geometry["patches"], 81) / 255
        face = ChartObservation.from_samples(
            samples=samples, chart_id="capture-" + image_data.pixel_digest[:16], face_id="classic-%02d" % index,
            reference_digest=reference.digest, cell_ids=tuple("r%dc%d" % (row+1, col+1) for row in range(4) for col in range(6)),
            quadrilaterals=geometry["patches"], confidence=geometry["score"], image_shape=rgb.shape[:2],
            stage=stage, template_id="classic_4x6", layout=(4, 6),
            provenance={"identity_scope": "one source capture and fixed geometry; not physical chart serial identity",
                        "measurement_policy": "float64-bilinear" if explicit else "legacy-uint8-opencv-grid",
                        "detector": detector_provenance,
                        "detector_face_index": index})
        faces.append(face.validate_reference(reference))
    return ObservationSet(tuple(faces), image_data.digest).validate_reference(reference)
