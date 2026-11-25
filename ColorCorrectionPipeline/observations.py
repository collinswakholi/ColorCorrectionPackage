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


_FACE_PREFIX = {"classic_4x6": "classic", "spyder_4x6": "spyder", "digital_sg_10x14": "digital-sg"}


def capture_observations(rgb, image_data, reference, stage, *, explicit=False, force=False):
    """Resample all available faces without changing their within-run identities."""
    from .detection import ChartDetectionError
    from .detection.charts import _LAYOUTS, _TYPE_NAMES, _scope, _model_charts, _sample

    state = _scope.get()
    if state is None:
        raise RuntimeError("observations require a detection context")
    bgr8 = None
    if not explicit or (force and state.geometry is None):
        if explicit and state.source_image is not None and state.source_image.shape == rgb.shape:
            bgr8 = state.source_image
        else:
            bgr8 = np.clip(np.asarray(rgb)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
    if state.geometry is None and force:
        try:
            _model_charts(bgr8)
        except ChartDetectionError as exc:
            return ObservationSet(
                (), image_data.digest, unavailable_reason=type(exc).__name__ + ": " + str(exc)
            )
    if state.geometry is None:
        return ObservationSet(
            (),
            image_data.digest,
            unavailable_reason="model geometry unavailable; no observations invented",
        )
    faces = []
    last_detection = next(
        (item for item in reversed(state.diagnostics) if item.get("status") == "detected"), {}
    )
    detector_provenance = {
        key: last_detection[key]
        for key in (
            "sha256",
            "provider",
            "runtime_version",
            "graphics_backend",
            "cpu_fallback",
            "execution_verified",
            "execution_evidence",
            "node_provider_counts",
            "geometry_source",
        )
        if key in last_detection
    }
    for index, geometry in enumerate(state.geometry):
        sampling = {}
        if explicit:
            if state.options.sampling_policy in ("ui-compatible-v1", "ui-native-v1"):
                from .ui_native import points, sample_regions_float64
                from .ui_fitting import sample_regions

                count = state.options.samples_per_axis
                xy = None
                if state.options.samples_per_patch is not None:
                    count = state.options.samples_per_patch
                    seed = state.options.sampling_seed or 0
                    xy = points(geometry["patches"], count, seed)
                    sampling = {"sampling_pattern": "random-lcg32-v1", "sampling_seed": seed}
                sampler = (
                    sample_regions_float64
                    if getattr(state, "measurement_runtime", None) == "precision-f64-v1"
                    else sample_regions
                )
                samples = sampler(rgb, geometry["patches"], count, xy=xy)
            else:
                samples = sample_quadrilaterals(rgb, geometry["patches"], 81)
        else:
            samples = _sample(bgr8, geometry["patches"], 81) / 255
        template = _TYPE_NAMES[geometry.get("type", 0)]
        rows, cols = _LAYOUTS[geometry.get("type", 0)]
        face = ChartObservation.from_samples(
            samples=samples,
            chart_id="capture-" + image_data.pixel_digest[:16],
            face_id="%s-%02d" % (_FACE_PREFIX[template], index),
            reference_digest=reference.digest,
            cell_ids=tuple(
                "r%dc%d" % (row + 1, col + 1) for row in range(rows) for col in range(cols)
            ),
            quadrilaterals=geometry["patches"],
            confidence=geometry["score"],
            image_shape=rgb.shape[:2],
            stage=stage,
            template_id=template,
            layout=(rows, cols),
            provenance={
                **sampling,
                "identity_scope": "one source capture and fixed geometry; not physical chart serial identity",
                "measurement_policy": (
                    (
                        "ui-compatible-v1"
                        if state.options.sampling_policy == "ui-compatible-v1"
                        else "float64-bilinear"
                    )
                    if explicit
                    else "legacy-uint8-opencv-grid"
                ),
                "detector": detector_provenance,
                "detector_face_index": index,
            },
        )
        if explicit and state.options.sampling_policy == "ui-native-v1":
            from dataclasses import replace
            from .ui_native import ordered_mean

            face = replace(
                face,
                measured_rgb=ordered_mean(samples),
                provenance={
                    **dict(face.provenance),
                    "measurement_policy": getattr(state, "measurement_runtime", "ui-native-v1"),
                    "sample_reduction": "row-order-f64",
                },
            )
        faces.append(face.validate_reference(reference))
    return ObservationSet(tuple(faces), image_data.digest).validate_reference(reference)
