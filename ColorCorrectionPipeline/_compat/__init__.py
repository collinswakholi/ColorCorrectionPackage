"""Load-only compatibility for calibrations saved by 2.x and earlier.

3.0 removed the historical fitters (``legacy-v1``, ``reproducible-v1``, the
``ui-compatible-v1`` ``staged-v1`` runtime, API-1 configurations and the 1.x
``key_functions`` module). Models those fitters saved must keep loading and
applying exactly as before. The minimum code needed for that lives here, and
nothing in this package fits a new calibration:

- :func:`load_pickle` reads trusted model pickles, mapping class paths that no
  longer exist (``ColorCorrectionPipeline.key_functions.Regressor_Model`` and
  ``.CustomNN`` from 1.x pickles) onto their current inference classes.

Importing this package must stay cheap: applying a saved bundle does not import
scikit-learn, pandas or torch.
"""

import pickle

__all__ = ["load_pickle", "REMOVED_CLASS_PATHS"]

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
