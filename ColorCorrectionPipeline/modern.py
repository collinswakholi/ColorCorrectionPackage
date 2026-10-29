"""Opt-in numerical stage orchestration over fixed, full-precision observations.

Legacy entry points stay in pipeline.py. These helpers deliberately do not detect
new geometry or manufacture measurements from reference targets.
"""

import copy
import numpy as np

from .colorimetry import D65, rgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_rgb
from .observations import capture_observations
from .sampling import sample_quadrilaterals
from .stage_policies import fit_monotonic_tone, fit_neutral_wb, saturation_selection


def selected_face(owner, rgb, stage):
    observations = capture_observations(rgb, owner.image_data, owner.reference_profile,
                                        stage, explicit=True, force=True)
    index = owner._chart_index
    if index >= len(observations.observations):
        raise ValueError("%s requires selected Classic24 face %d: %s" %
                         (stage, index, observations.unavailable_reason or "face unavailable"))
    return observations.observations[index]


def prepare_source(owner, *, required):
    owner._source_face = None
    if not required:
        return
    face = selected_face(owner, owner.Image, "source")
    owner._source_face = face
    samples = sample_quadrilaterals(owner.Image, face.quadrilaterals, face.sample_count)
    per_channel = np.mean(samples >= face.clipping_threshold, axis=1)
    owner.numerical_report["source_clipping"] = {
        "basis": "original sampled pixels at encoded RGB threshold; not sensor RAW evidence",
        "threshold": face.clipping_threshold, "cell_ids": list(face.cell_ids),
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


def apply_tone(rgb, model):
    image = np.asarray(rgb, dtype=np.float64)
    lab = xyz_to_lab(rgb_to_xyz(image, encoding="srgb"), white=D65)
    lab[..., 0] = model.predict(lab[..., 0])
    return xyz_to_rgb(lab_to_xyz(lab, white=D65), encoding="srgb", policy="clip")


def fit_gamma(owner, image):
    face = selected_face(owner, image, "GC_training")
    lightness = xyz_to_lab(rgb_to_xyz(face.measured_rgb, encoding="srgb"), white=D65)[:, 0]
    reference = owner.color_semantics.reference_to_working_lab(owner.reference_profile.canonical_lab_d50)[:, 0]
    model, report = fit_monotonic_tone(lightness, reference, face.cell_ids,
                                     owner.reference_profile.neutral_ids,
                                     clipped_fractions=clipping(owner, face), policy=owner.stage_policy)
    owner.models.model_gc = model
    owner.numerical_report["GC"] = report
    return apply_tone(image, model), {}, False


def fit_wb(owner, image):
    face = selected_face(owner, image, "WB_training")
    matrix, report = fit_neutral_wb(face.measured_rgb, owner.REFERENCE_RGB_PD.values,
                                  face.cell_ids, owner.reference_profile.neutral_ids,
                                  clipped_fractions=clipping(owner, face), policy=owner.stage_policy)
    owner.models.model_wb = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)
    owner.numerical_report["WB"] = report
    return np.clip(image @ matrix, 0, 1), {}, False


def report_saturation(owner, image):
    face = selected_face(owner, image, "saturation")
    _, report = saturation_selection(face.cell_ids, clipping(owner, face), policy=owner.stage_policy)
    report["image_transform"] = "identity; observations are never reference-assisted recovery"
    owner.numerical_report["saturation"] = report
    return image, None, None


def fit_cc(owner, image, method, parameters):
    from .fitting import fit_observations, fit_conventional_observations
    from .evaluation import grouped_cc_validation
    from .core.correction import predict_
    import colour

    face = selected_face(owner, image, "CC_training")
    fractions = clipping(owner, face)
    eligible, selection = saturation_selection(face.cell_ids, fractions, policy=owner.stage_policy)
    x, y = face.measured_rgb, owner.REFERENCE_RGB_PD.values
    groups = np.asarray(face.cell_ids)
    synthetic = np.zeros(len(x), dtype=bool)
    if owner.fitting_policy.anchor_weight > 0:
        x = np.concatenate((x, np.array([[0., 0., 0.], [1., 1., 1.]])))
        y = np.concatenate((y, np.array([[0., 0., 0.], [1., 1., 1.]])))
        groups = np.r_[groups, ["synthetic-assumption:black", "synthetic-assumption:white"]]
        eligible = np.r_[eligible, [True, True]]
        synthetic = np.r_[synthetic, [True, True]]
    # The documented sample is one fixed region mean per independent chart cell.
    # No repeated means, random pseudo-observations, or implicit synthetic anchors.
    params = copy.deepcopy(parameters or {})
    for option in ("n_samples", "augment_extremes", "extreme_anchor_repeat", "shuffle_fit_data"):
        if option in params:
            raise ValueError("%s is a legacy sampling option; reproducible-v1 uses one mean per cell and FittingPolicy.anchor_weight" % option)
    display_keys = {"get_deltaE", "show", "cc_method"}
    params = {key: value for key, value in params.items() if key not in display_keys}
    if method.lower() == "ours":
        params.setdefault("degree", 1)
        params.setdefault("mtd", "linear")
        params.setdefault("use_lut", False)
        model = fit_observations(x[eligible], y[eligible], groups[eligible], params, owner.fitting_policy, synthetic_mask=synthetic[eligible])
        effective = copy.deepcopy(model.fit_report.get("effective_parameters", params))
        owner.models.model_cc = (model, effective, "ours")
        output = np.clip(predict_(image, model), 0, 1)
        fit_report = model.fit_report

        def fit_predict(a, b, g, test):
            fold_params = {**params, "use_lut": False}
            mask = np.array([cell.startswith("synthetic-assumption:") for cell in g])
            fitted = fit_observations(a, b, g, fold_params, owner.fitting_policy, synthetic_mask=mask)
            return np.clip(predict_(test, fitted), 0, 1)
    elif method.lower() == "conv":
        model, effective, fit_report = fit_conventional_observations(
            x[eligible], y[eligible], groups[eligible], params, owner.fitting_policy, synthetic_mask=synthetic[eligible])
        owner.models.model_cc = (model, effective, "conv")
        from .conventional import apply_conventional
        output = np.clip(apply_conventional(image, model, **effective), 0, 1)

        def fit_predict(a, b, g, test):
            mask = np.array([cell.startswith("synthetic-assumption:") for cell in g])
            fitted, options, _ = fit_conventional_observations(a, b, g, params, owner.fitting_policy, synthetic_mask=mask)
            return np.clip(apply_conventional(test, fitted, **options), 0, 1)
    else:
        raise ValueError("cc_method must be ours or conv")

    report = {"fit": fit_report, "selection": selection,
              "sampling": "one full-precision fixed-region mean per cell; no independent pixel repetitions"}
    report["anchor_assumptions"] = {"black": [0, 0, 0], "white": [1, 1, 1],
                                   "weight_per_anchor": owner.fitting_policy.anchor_weight,
                                   "measured": False, "validation_test_eligible": False}
    if owner.validation_options.get("enabled", True):
        reference_lab = owner.reference_profile.canonical_lab_d50
        # Match fold targets to canonical reference values, not clipped display swatches.
        def metric(expected, predicted, indices=None):
            if indices is None:
                raise ValueError("canonical CIEDE2000 requires original validation indices")
            return owner.color_semantics.delta_e00(predicted, reference_lab[np.asarray(indices)], input_encoding="srgb")

        report["validation"] = grouped_cc_validation(
            x, y, groups, fit_predict, eligible=eligible, synthetic_mask=synthetic,
            n_splits=owner.validation_options.get("n_splits", 4),
            random_state=owner.validation_options.get("random_state", owner.fitting_policy.random_state),
            metric=metric, metric_name="CIEDE2000 canonical reference at common evaluation white")
        report["validation"]["prediction_path"] = "direct estimator on held-out region means; not mean of corrected pixels or LUT output"
    else:
        report["validation"] = {"status": "unavailable", "reason": "disabled by caller", "scope": "CC-only conditional on fitted upstream stages"}
    owner.numerical_report["CC"] = report
    return output, {}, False
