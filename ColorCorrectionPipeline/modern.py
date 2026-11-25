"""Stage fitting (FFC, GC, WB, CC and the saturation report) for one calibration.

``StageFit`` owns the state that the stages of one run share: the selected chart
face, the original source observations and the FFC gain domain. The pipeline
creates one per run and calls its stage methods in order with the fitting
candidate (``owner``), whose public inputs they read and whose fitted models and
numerical report they fill.

Stages resample fixed chart observations; they never detect new geometry or
manufacture measurements from reference targets.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from . import ui_native
from .observations import capture_observations
from .stage_policies import fit_monotonic_tone, fit_neutral_wb, saturation_selection
from .ui_fitting import fit_ui, round_image, sample_regions


class StageFit:
    """Fit-time state shared between the stages of one run."""

    def __init__(self, chart_index=0):
        self._chart_index = chart_index
        self._source_face = None
        self._ffc_domain = "lightness"

    # -- shared observations ------------------------------------------------

    def _face(self, owner, rgb, stage):
        observations = capture_observations(
            rgb, owner.image_data, owner.reference_profile, stage, explicit=True, force=True
        )
        if self._chart_index >= len(observations.observations):
            raise ValueError(
                "%s requires selected chart face %d: %s"
                % (
                    stage,
                    self._chart_index,
                    observations.unavailable_reason or "face unavailable",
                )
            )
        return observations.observations[self._chart_index]

    def _clipping(self, face):
        source = self._source_face
        if source is None:
            raise ValueError("original chart observations were not captured")
        if source.face_id != face.face_id or source.cell_ids != face.cell_ids:
            raise ValueError("stage observations no longer match original chart cells")
        return np.maximum(source.clipped_fractions, face.clipped_fractions)

    def prepare_source(self, owner, *, required):
        """Record original-pixel clipping of the selected face before any stage runs."""
        self._source_face = None
        if not required:
            return
        face = self._face(owner, owner.Image, "source")
        self._source_face = face
        high = ui_native.high_precision(owner.fitting_policy)
        sampler = ui_native.sample_regions_float64 if high else sample_regions
        count, xy = ui_native.face_sample_points(face)
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

    # -- FFC/GC rendering ---------------------------------------------------

    def render(self, owner, image, stage):
        """Render a fitted FFC or GC stage over ``image`` as bundle prediction will.

        Fit metadata is committed only after all stages succeed, so this uses the
        resolved policy and this run's FFC domain with the same guarded renderer.
        """
        high = ui_native.high_precision(owner.fitting_policy)
        if not high and image.shape[0] * image.shape[1] >= 65536:
            from .ui_execution import try_execute

            result = try_execute(
                image,
                owner.models,
                (stage,),
                fit_policy=owner.fitting_policy.to_dict(),
                ffc_domain=self._ffc_domain,
            )
            if result is not None:
                return result[stage]
        return ui_native.apply_color(
            image,
            owner.models.model_ffc,
            owner.models.model_gc if stage == "GC" else None,
            rounded=not high,
            domain=self._ffc_domain,
        )

    def _sample_after_ffc(self, owner, face):
        """Measure a face through the fitted FFC gain at the sample positions."""
        count, xy = ui_native.face_sample_points(face)
        high = ui_native.high_precision(owner.fitting_policy)
        samples = (ui_native.sample_regions_float64 if high else sample_regions)(
            owner.Image, face.quadrilaterals, count, xy=xy
        )
        h, w = owner.Image.shape[:2]
        dtype = np.float64 if high else np.float32
        x = (np.clip(xy[..., 0], 0, w - 1) * (w - 1) / max(w - 1, 1)).astype(dtype)
        y = (np.clip(xy[..., 1], 0, h - 1) * (h - 1) / max(h - 1, 1)).astype(dtype)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = x - x0.astype(dtype)
        fy = y - y0.astype(dtype)
        g = owner.models.model_ffc.astype(dtype)
        gains = (
            g[y0, x0] * (1 - fy) * (1 - fx)
            + g[y1, x0] * fy * (1 - fx)
            + g[y0, x1] * (1 - fy) * fx
            + g[y1, x1] * fy * fx
        )
        corrected = ui_native.apply_color(
            samples, gains, scalar=True, rounded=False, domain=self._ffc_domain
        )
        clipped = np.mean(np.any(corrected >= face.clipping_threshold, axis=2), axis=1)
        return replace(
            face,
            measured_rgb=ui_native.ordered_mean(corrected),
            clipped_fractions=clipped,
            provenance={
                **dict(face.provenance),
                "measurement_policy": ui_native.runtime(owner.fitting_policy)
                + "/FFC-after-bilinear",
                "sample_reduction": "row-order-f64",
            },
        )

    # -- stages -------------------------------------------------------------

    def ffc(self, owner, image, parameters):
        from .ui_stages import fit_gain

        gain, quality, self._ffc_domain = fit_gain(
            owner.White_Image,
            owner.white_image_data,
            image.shape,
            parameters,
            high_precision_mode=ui_native.high_precision(owner.fitting_policy),
        )
        owner.models.model_ffc = gain
        owner.numerical_report["FFC"] = quality
        return self.render(owner, image, "FFC"), {}, False

    def gamma(self, owner, image):
        face = self._face(owner, image, "GC_training")
        if owner.models.model_ffc is not None:
            face = self._sample_after_ffc(owner, face)
        lightness = ui_native.lab(face.measured_rgb, scalar=True)[:, 0]
        reference = ui_native.reference(owner.reference_profile)[2][:, 0]
        model, report = fit_monotonic_tone(
            lightness,
            reference,
            face.cell_ids,
            owner.reference_profile.neutral_ids,
            clipped_fractions=self._clipping(face),
            policy=owner.stage_policy,
        )
        owner.models.model_gc = model
        owner.numerical_report["GC"] = report
        report["ui_runtime"] = ui_native.runtime(owner.fitting_policy)
        report["training_observation"] = face.to_dict()
        return self.render(owner, owner.Image, "GC"), {}, False

    def wb(self, owner, image):
        face = self._face(owner, image, "WB_training")
        matrix, report = fit_neutral_wb(
            face.measured_rgb,
            owner.REFERENCE_RGB_PD.values,
            face.cell_ids,
            owner.reference_profile.neutral_ids,
            clipped_fractions=self._clipping(face),
            policy=owner.stage_policy,
        )
        owner.models.model_wb = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(
            matrix.shape
        )
        owner.numerical_report["WB"] = report
        result = np.clip(image @ matrix, 0, 1)
        if not ui_native.high_precision(owner.fitting_policy):
            result = round_image(result)
        return result, {}, False

    def cc(self, owner, image, method, parameters):
        from .core.correction import Regressor_Model, chunked_cc, predict_
        from .evaluation import grouped_cc_validation

        face = self._face(owner, image, "CC_training")
        x = np.asarray(face.measured_rgb)
        truth = owner.reference_profile.canonical_lab_d50
        params = dict(parameters or {})
        params["cc_method"] = method
        high = ui_native.high_precision(owner.fitting_policy)
        # Fits use the native-v1 or precision-f64-v1 runtime (staged-v1 fitting was
        # removed in 3.0); both take the native linear-light reference targets.
        y = ui_native.reference(owner.reference_profile)[0]
        if high:
            params.setdefault("use_lut", False)
        predictor, report = fit_ui(x, y, params)
        report["details"]["ui_runtime"] = ui_native.HIGH_RUNTIME if high else ui_native.RUNTIME
        settings = report["effective_parameters"]
        model = Regressor_Model.from_ui_fit(predictor, report)
        owner.models.model_cc = (model, deepcopy(settings), "ours")
        numerical = {
            "fit": report,
            "selection": {
                "policy": "all measured cells",
                "source": "fixed UI sampling",
                "excluded_cell_ids": [],
                "reference_assisted_values": False,
                "independent_cells": len(x),
            },
            "sampling": "native float32 bilinear; inner .70; per-axis grid",
        }
        if owner.validation_options.get("enabled", True):

            def fit_predict(a, b, g, test):
                fold_settings = dict(settings)
                # The UI chooses batch size from each fold's actual training rows.
                fold_settings.pop("batch_size", None)
                fitted, _ = fit_ui(a, b, fold_settings)
                return fitted.predict_rgb(test)

            def metric(expected, predicted, indices=None):
                if indices is None:
                    raise ValueError("UI validation requires original cell indices")
                return owner.color_semantics.delta_e00(
                    predicted, truth[np.asarray(indices)], input_encoding="srgb"
                )

            numerical["validation"] = grouped_cc_validation(
                x,
                y,
                np.asarray(face.cell_ids),
                fit_predict,
                n_splits=owner.validation_options.get("n_splits", 4),
                random_state=owner.validation_options.get("random_state", 42),
                metric=metric,
                metric_name="CIEDE2000 at common evaluation white",
            )
        else:
            numerical["validation"] = {
                "status": "unavailable",
                "reason": "disabled by caller",
                "scope": "CC-only conditional on fitted upstream stages",
            }
        owner.numerical_report["CC"] = numerical
        if high or "basis" in settings:
            corrected = chunked_cc(image, owner.models.model_cc, 65536)
        else:
            corrected = predict_(image, model)
        corrected = np.clip(corrected, 0, 1)
        return (corrected if high else round_image(corrected)), {}, False

    def saturation(self, owner, image):
        face = self._face(owner, image, "saturation")
        _, report = saturation_selection(
            face.cell_ids, self._clipping(face), policy=owner.stage_policy
        )
        report["image_transform"] = "identity; observations are never reference-assisted recovery"
        owner.numerical_report["saturation"] = report
        return image, None, None
