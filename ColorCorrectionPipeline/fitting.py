"""Fitting-policy identity and the shared observation validators.

``FittingPolicy`` names the numerical recipe of a calibration. New fits use
``ui-compatible-v1`` with the ``native-v1`` or ``precision-f64-v1`` runtime (the
``auto-v2`` request resolves to one of them). ``legacy-v1``, ``reproducible-v1``
and the ``staged-v1`` runtime remain representable because saved calibrations
record them; their fitters were removed in 3.0 (see docs/migration-3.md).
The dataclass defaults describe how records that omit a field are read, so they
keep their historical values.
"""

from dataclasses import dataclass, asdict

import numpy as np

_UI_SAMPLING_POLICIES = {
    "staged-v1": "ui-compatible-v1",
    "native-v1": "ui-native-v1",
    "auto-v2": "ui-native-v1",
    "precision-f64-v1": "ui-native-v1",
}


@dataclass(frozen=True)
class FittingPolicy:
    version: str = "legacy-v1"
    random_state: int = 42
    anchor_weight: float = 0.0
    condition_limit: float = 1e12
    ui_runtime: str = "staged-v1"

    def __post_init__(self):
        if self.version not in ("legacy-v1", "reproducible-v1", "ui-compatible-v1"):
            raise ValueError("unknown fitting policy version")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, (int, np.integer))
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be an integer in [0, 2**32)")
        if (
            isinstance(self.anchor_weight, (bool, np.bool_))
            or not isinstance(self.anchor_weight, (int, float, np.number))
            or not np.isfinite(self.anchor_weight)
            or self.anchor_weight < 0
        ):
            raise ValueError("anchor_weight must be finite and nonnegative")
        if (
            isinstance(self.condition_limit, (bool, np.bool_))
            or not isinstance(self.condition_limit, (int, float, np.number))
            or not np.isfinite(self.condition_limit)
            or self.condition_limit <= 1
        ):
            raise ValueError("condition_limit must be finite and greater than one")
        if self.ui_runtime not in _UI_SAMPLING_POLICIES or (
            self.ui_runtime != "staged-v1" and self.version != "ui-compatible-v1"
        ):
            raise ValueError("native UI runtime requires ui-compatible-v1")
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(self, "anchor_weight", float(self.anchor_weight))
        object.__setattr__(self, "condition_limit", float(self.condition_limit))

    def to_dict(self):
        result = asdict(self)
        if self.ui_runtime == "staged-v1":
            # Keep older policy serialization and its identity digest unchanged.
            result.pop("ui_runtime")
        return result

    @property
    def sampling_policy(self):
        """Keep runtime validation and its source-measurement contract together."""
        return (
            _UI_SAMPLING_POLICIES[self.ui_runtime]
            if self.version == "ui-compatible-v1"
            else "default"
        )

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            if value == "ui-native-v1":
                return cls(version="ui-compatible-v1", ui_runtime="native-v1")
            return cls(version=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("fitting policy must be a version, dictionary or FittingPolicy")


def _integer(value, name, minimum=1, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("%s must be an integer in [%s, %s]" % (name, minimum, maximum))
    return int(value)


def _paired_data(X, Y, groups, synthetic_mask=None):
    X = np.array(X, dtype=np.float64, copy=True)
    Y = np.array(Y, dtype=np.float64, copy=True)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape != Y.shape or not len(X):
        raise ValueError("observations and targets must be nonempty paired (N,3) arrays")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("fitting observations and targets must be finite")
    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("fitting observations require encoded RGB in [0,1]")
    group_list = list(groups)
    if len(group_list) != len(X) or any(not isinstance(x, str) or not x for x in group_list):
        raise ValueError("groups must contain one nonempty canonical identity string per row")
    mask = np.zeros(len(X), dtype=bool) if synthetic_mask is None else np.asarray(synthetic_mask)
    if mask.shape != (len(X),) or mask.dtype.kind != "b":
        raise ValueError("synthetic_mask must be a boolean vector with one entry per row")
    return X, Y, np.array(group_list), mask.copy()


_FIT_SUPPORTED_RUNTIMES = ("auto-v2", "native-v1", "precision-f64-v1")


def _removed_policy_error(policy):
    """Return the 3.0 error for a fitting policy that can only be loaded, not fitted."""
    policy = FittingPolicy.from_value(policy)
    name = policy.version if policy.version != "ui-compatible-v1" else "ui-compatible-v1/staged-v1"
    return ValueError(
        "fitting_policy %r was removed in ColorCorrectionPipeline 3.0; models saved with "
        "it still load and apply. Fit a new calibration with the default Config() or "
        'Config(model="3x3"|"rpcc2"|"rpcc3"|"linear"|"pls"|"nn", precision="auto"|"high"); '
        "see docs/migration-3.md." % name
    )


def require_fit_supported(policy):
    """Raise unless ``policy`` names a recipe that 3.0 can fit."""
    policy = FittingPolicy.from_value(policy)
    if policy.version != "ui-compatible-v1" or policy.ui_runtime not in _FIT_SUPPORTED_RUNTIMES:
        raise _removed_policy_error(policy)
    return policy


def __getattr__(name):
    if name == "DeterministicTorchRegressor":
        # reproducible-v1 pickles name this path; the class is load-only now.
        from ._compat.estimators import DeterministicTorchRegressor

        globals()[name] = DeterministicTorchRegressor
        return DeterministicTorchRegressor
    raise AttributeError(name)
