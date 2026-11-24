"""Modern-v1 stage fitting (GC, WB, saturation report) over fixed observations.

These helpers deliberately do not detect new geometry or manufacture
measurements from reference targets.
"""

import numpy as np

from .observations import capture_observations
from .stage_policies import (
    fit_monotonic_tone,
    fit_neutral_wb,
    saturation_selection,
)


def selected_face(owner, rgb, stage):
    observations = capture_observations(
        rgb, owner.image_data, owner.reference_profile, stage, explicit=True, force=True
    )
    index = owner._chart_index
    if index >= len(observations.observations):
        raise ValueError(
            "%s requires selected Classic24 face %d: %s"
            % (stage, index, observations.unavailable_reason or "face unavailable")
        )
    return observations.observations[index]


def prepare_source(owner, *, required):
    owner._source_face = None
    if not required:
        return
    face = selected_face(owner, owner.Image, "source")
    owner._source_face = face
    from .ui_fitting import sample_regions
    from .ui_native import face_sample_points, high_precision, sample_regions_float64

    sampler = sample_regions_float64 if high_precision(owner) else sample_regions
    count, xy = face_sample_points(face)
    samples = sampler(owner.Image, face.quadrilaterals, count, xy=xy)
    per_channel = np.mean(samples >= face.clipping_threshold, axis=1)
    owner.numerical_report["source_clipping"] = {
        "basis": "original sampled pixels at encoded RGB threshold; not sensor RAW evidence",
        "threshold": face.clipping_threshold,
        "cell_ids": list(face.cell_ids),
        "fractions_any_channel": face.clipped_fractions.tolist(),
        "fractions_per_channel": per_channel.tolist(),
        "source_observation_digest": face.digest,
    }


def clipping(owner, face):
    source = owner._source_face
    if source is None:
        raise ValueError("original chart observations were not captured")
    if source.face_id != face.face_id or source.cell_ids != face.cell_ids:
        raise ValueError("stage observations no longer match original chart cells")
    return np.maximum(source.clipped_fractions, face.clipped_fractions)


def fit_gamma(owner, image):
    face = selected_face(owner, image, "GC_training")
    from .ui_native import lab as native_lab, reference as native_reference, sample_ffc

    if owner.models.model_ffc is not None:
        face = sample_ffc(owner, face)
    lightness = native_lab(face.measured_rgb, scalar=True)[:, 0]
    reference = native_reference(owner.reference_profile)[2][:, 0]
    model, report = fit_monotonic_tone(
        lightness,
        reference,
        face.cell_ids,
        owner.reference_profile.neutral_ids,
        clipped_fractions=clipping(owner, face),
        policy=owner.stage_policy,
    )
    owner.models.model_gc = model
    owner.numerical_report["GC"] = report
    from .ui_native import render_fit_stage, runtime

    report["ui_runtime"] = runtime(owner)
    report["training_observation"] = face.to_dict()
    return render_fit_stage(owner, owner.Image, "GC"), {}, False


def fit_wb(owner, image):
    face = selected_face(owner, image, "WB_training")
    matrix, report = fit_neutral_wb(
        face.measured_rgb,
        owner.REFERENCE_RGB_PD.values,
        face.cell_ids,
        owner.reference_profile.neutral_ids,
        clipped_fractions=clipping(owner, face),
        policy=owner.stage_policy,
    )
    owner.models.model_wb = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(
        matrix.shape
    )
    owner.numerical_report["WB"] = report
    result = np.clip(image @ matrix, 0, 1)
    from .ui_native import high_precision

    if not high_precision(owner):
        from .ui_fitting import round_image

        result = round_image(result)
    return result, {}, False


def report_saturation(owner, image):
    face = selected_face(owner, image, "saturation")
    _, report = saturation_selection(
        face.cell_ids, clipping(owner, face), policy=owner.stage_policy
    )
    report["image_transform"] = "identity; observations are never reference-assisted recovery"
    owner.numerical_report["saturation"] = report
    return image, None, None
