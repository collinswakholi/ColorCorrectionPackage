"""Load-only compatibility for calibrations saved by 2.x and earlier.

3.0 removed the historical fitters (``legacy-v1``, ``reproducible-v1``, the
``ui-compatible-v1`` ``staged-v1`` runtime, API-1 configurations and the 1.x
``key_functions`` module). Models those fitters saved must keep loading and
applying exactly as before. The minimum code needed for that lives here, and
nothing in this package fits a new calibration:

- :func:`load_pickle` reads trusted model pickles, mapping class paths that no
  longer exist (``ColorCorrectionPipeline.key_functions.Regressor_Model`` and
  ``.CustomNN`` from 1.x pickles) onto their current inference classes.
- :func:`validate_saved_policies` checks the policy combination a saved
  calibration records, including the removed ``legacy-v1``/``reproducible-v1``/
  ``staged-v1`` recipes, without building a (fit-only) ``Config``.
- :mod:`.legacy_gc` applies legacy polynomial gamma (GC) coefficients.
- :mod:`.estimators` holds the load-only ``CustomNN`` and
  ``DeterministicTorchRegressor`` estimators (imports scikit-learn).

Records without metadata (written before 2.0) are read as ``legacy-v1`` color
semantics, stage and fitting policy: the dataclass defaults of ``ColorSemantics``,
``StagePolicy`` and ``FittingPolicy`` keep those historical values for that reason.

Importing this package must stay cheap: applying a saved bundle does not import
scikit-learn, pandas or torch.
"""

import pickle

__all__ = ["load_pickle", "validate_saved_policies", "REMOVED_CLASS_PATHS"]

# Class paths that 1.x pickles reference, mapped to their current definitions.
REMOVED_CLASS_PATHS = {
    ("ColorCorrectionPipeline.key_functions", "Regressor_Model"): (
        "ColorCorrectionPipeline.core.correction",
        "Regressor_Model",
    ),
    ("ColorCorrectionPipeline.key_functions", "CustomNN"): (
        "ColorCorrectionPipeline.core.correction",
        "CustomNN",
    ),
}
_REMOVED_MODULES = {module for module, _ in REMOVED_CLASS_PATHS}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        target = REMOVED_CLASS_PATHS.get((module, name))
        if target is not None:
            module, name = target
        elif module in _REMOVED_MODULES:
            raise pickle.UnpicklingError(
                "%s.%s was removed in ColorCorrectionPipeline 3.0 and is not part of a "
                "saved calibration" % (module, name)
            )
        return super().find_class(module, name)


def load_pickle(file):
    """Unpickle a trusted model file, remapping removed 1.x class paths.

    Unpickling can execute code; only load files from a trusted source.
    """
    return _CompatUnpickler(file).load()


def validate_saved_policies(
    semantics, stages, fitting, *, switches, image_options, validation_options
):
    """Reject policy combinations that no 2.x/3.0 fitter could have recorded.

    These are the capability gates 2.x applied by constructing a ``Config`` from
    saved metadata. 3.0 ``Config`` only describes new fits, so the rules for the
    removed recipes are kept here; current recipes get the same checks.
    """
    from ..colorimetry import D65
    from ..config import validate_image_options, validate_validation_options

    validate_validation_options(validation_options or {})
    validate_image_options(image_options)
    if fitting.version == "ui-compatible-v1":
        if semantics.adaptation != "Bradford" or semantics.fitting_domain != "linear-rgb":
            raise ValueError(
                "ui-compatible-v1 requires Bradford adaptation, linear-rgb fitting and UI sampling"
            )
        if fitting.ui_runtime == "native-v1" and semantics.input_encoding != "srgb":
            raise ValueError("native UI runtime requires sRGB input")
        if (
            switches["check_saturation"]
            or stages.saturation != "report"
            or fitting.anchor_weight
            or fitting.random_state != 42
            or fitting.condition_limit != 1e12
        ):
            raise ValueError(
                "ui-compatible-v1 requires check_saturation=False, report saturation, no anchors, seed 42 and the default condition_limit"
            )
    elif semantics.fitting_domain != "encoded-rgb":
        raise ValueError("linear-rgb fitting requires ui-compatible-v1")
    if (
        stages.version != "legacy-v1" or fitting.version != "legacy-v1"
    ) and semantics.version != "explicit-v1":
        raise ValueError(
            "modern stage and reproducible fitting policies require explicit-v1 color semantics"
        )
    if stages.version == "modern-v1" and switches["do_cc"] and fitting.version == "legacy-v1":
        raise ValueError(
            "modern-v1 CC requires reproducible-v1 fitting to honor observation exclusion"
        )
    if fitting.version != "legacy-v1" and stages.version != "modern-v1":
        raise ValueError(
            "reproducible-v1 requires modern-v1 stages to avoid fit-only legacy pixel scaling"
        )
    if semantics.version == "legacy-v1" and (
        semantics.input_encoding != "srgb"
        or semantics.output_encoding != "srgb"
        or semantics.adaptation != "CAT02"
        or semantics.evaluation_white != D65
        or semantics.output_policy != "clip"
    ):
        raise ValueError(
            "legacy-v1 retains historical encoded sRGB/CAT02 behavior; choose explicit-v1 for new color semantics"
        )
    if semantics.version == "explicit-v1":
        if semantics.output_policy != "clip":
            raise ValueError(
                "this stage engine clips outputs; explicit-v1 pipeline requires output_policy='clip'"
            )
        if image_options.get("apply_orientation", True) is False:
            raise ValueError(
                "explicit pipeline geometry requires applied orientation; use read_image_data separately to inspect raw coordinates"
            )
    elif (set(image_options) - {"apply_orientation", "icc_policy"}) or image_options.get(
        "icc_policy"
    ) == "convert-srgb":
        raise ValueError("new file decoding policies require explicit-v1 color semantics")
