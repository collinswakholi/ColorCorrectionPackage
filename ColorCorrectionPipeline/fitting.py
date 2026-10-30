"""Versioned CPU fitting with independent-cell weighting and effective controls.

This module deliberately keeps new numerical behavior separate from legacy fits.
Reproducibility is qualified for a fixed supported CPU software environment.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
from importlib.metadata import version as package_version
import json
import math
import threading

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

_TORCH_FIT_LOCK = threading.RLock()


def _plain(value):
    """Own library parameter metadata using only JSON-supported value types."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unsupported estimator parameter metadata type: " + type(value).__name__)


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
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        if isinstance(self.anchor_weight, (bool, np.bool_)) or not isinstance(self.anchor_weight, (int, float, np.number)) or not np.isfinite(self.anchor_weight) or self.anchor_weight < 0:
            raise ValueError("anchor_weight must be finite and nonnegative")
        if isinstance(self.condition_limit, (bool, np.bool_)) or not isinstance(self.condition_limit, (int, float, np.number)) or not np.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("condition_limit must be finite and greater than one")
        if self.ui_runtime not in ("staged-v1", "native-v1") or (self.ui_runtime != "staged-v1" and self.version != "ui-compatible-v1"):
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
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("%s must be an integer in [%s, %s]" % (name, minimum, maximum))
    return int(value)


def _number(value, name, minimum=0, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        raise ValueError("%s must be finite and %s %s" % (name, ">=" if inclusive else ">", minimum))
    return float(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(name + " must be boolean")
    return bool(value)


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


def collapse_cell_observations(X, Y, groups, synthetic_mask=None):
    """Equal independent-cell weight; repeated pixels do not create new colors.

    Targets within a group must agree. Each group contributes its observed mean;
    the operation is not a claim that averaging commutes with nonlinear inference.
    """
    X, Y, groups, synthetic = _paired_data(X, Y, groups, synthetic_mask)
    identities = sorted(set(groups.tolist()))
    means, targets, kinds, counts = [], [], [], []
    for identity in identities:
        rows = groups == identity
        if np.any(synthetic[rows] != synthetic[rows][0]):
            raise ValueError("a group cannot mix measured and synthetic rows")
        if not np.allclose(Y[rows], Y[rows][0], atol=1e-12, rtol=0):
            raise ValueError("a cell group has inconsistent reference targets")
        means.append(np.mean(X[rows], axis=0))
        targets.append(Y[rows][0])
        kinds.append(synthetic[rows][0])
        counts.append(int(rows.sum()))
    return np.array(means), np.array(targets), np.array(identities), np.array(kinds), counts


def _design_report(features, policy, *, require_full_rank):
    if features.ndim != 2 or not features.size or not np.all(np.isfinite(features)):
        raise ValueError("design matrix must be nonempty and finite before SVD")
    singular = np.linalg.svd(features, compute_uv=False)
    rank = int(np.linalg.matrix_rank(features))
    full = rank == features.shape[1]
    condition = float(singular[0] / singular[-1]) if full and singular[-1] > 0 else None
    report = {"rows": len(features), "columns": features.shape[1], "rank": rank,
              "condition_number": condition, "condition_limit": policy.condition_limit}
    if require_full_rank and (not full or condition > policy.condition_limit):
        raise ValueError("polynomial fit is rank deficient or exceeds condition_limit: " + repr(report))
    return report


class DeterministicTorchRegressor:
    """Small CPU NN with grouped holdout and scoped initialization/dropout RNG."""
    def __init__(self, *, hidden_layers, learning_rate, max_epochs, batch_size,
                 patience, tol, dropout_rate, use_batch_norm, optim_type, random_state):
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.optim_type = optim_type
        self.random_state = random_state
        self.device = "cpu"

    def fit(self, X, y, groups, synthetic_mask=None):
        import torch
        import torch.nn as nn
        X, y = np.asarray(X), np.asarray(y)
        groups = np.asarray(groups)
        synthetic = np.zeros(len(X), bool) if synthetic_mask is None else np.asarray(synthetic_mask, bool)
        measured_groups = sorted(set(groups[~synthetic].tolist()))
        if len(measured_groups) < 3:
            raise ValueError("custom NN grouped validation needs at least three measured cells")
        rng = np.random.default_rng(self.random_state)
        shuffled = np.asarray(measured_groups)[rng.permutation(len(measured_groups))]
        validation_groups = set(shuffled[:max(1, int(math.ceil(len(shuffled) * .15)))])
        validation = np.array([g in validation_groups for g in groups]) & ~synthetic
        training = ~validation
        if self.use_batch_norm and (training.sum() < 2 or self.batch_size < 2):
            raise ValueError("batch normalization requires at least two training cells and batch_size>=2")
        self.split_report = {"training_groups": sorted(set(groups[training].tolist())),
                             "validation_groups": sorted(validation_groups),
                             "scope": "internal grouped CC fit selection; upstream stages fixed"}
        with _TORCH_FIT_LOCK, torch.random.fork_rng(devices=[]):
            # Seed only CPU state covered by fork_rng. torch.manual_seed also
            # touches CUDA generators, which a CPU-only fit must leave alone.
            torch.random.default_generator.manual_seed(self.random_state)
            old_deterministic = torch.are_deterministic_algorithms_enabled()
            old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                layers = []
                width = X.shape[1]
                for hidden in self.hidden_layers:
                    layers.append(nn.Linear(width, hidden, dtype=torch.float32, device="cpu"))
                    if self.use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden, dtype=torch.float32, device="cpu"))
                    layers.append(nn.ReLU())
                    if self.dropout_rate:
                        layers.append(nn.Dropout(self.dropout_rate))
                    width = hidden
                layers.append(nn.Linear(width, y.shape[1], dtype=torch.float32, device="cpu"))
                self.model = nn.Sequential(*layers)
                optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}
                optimizer = optimizers[self.optim_type](self.model.parameters(), lr=self.learning_rate)
                self.optimizer_parameters = _plain(optimizer.defaults)
                xt, yt = torch.tensor(X[training], dtype=torch.float32, device="cpu"), torch.tensor(y[training], dtype=torch.float32, device="cpu")
                xv, yv = torch.tensor(X[validation], dtype=torch.float32, device="cpu"), torch.tensor(y[validation], dtype=torch.float32, device="cpu")
                generator = torch.Generator(device="cpu").manual_seed(self.random_state)
                best, stale, state = float("inf"), 0, None
                for epoch in range(self.max_epochs):
                    self.model.train()
                    order = torch.randperm(len(xt), generator=generator, device="cpu")
                    batches = list(order.split(self.batch_size))
                    # Merge a singleton rather than discarding a measured color.
                    if self.use_batch_norm and len(batches) > 1 and len(batches[-1]) == 1:
                        batches[-2] = torch.cat((batches[-2], batches[-1]))
                        batches.pop()
                    for batch in batches:
                        optimizer.zero_grad()
                        loss = ((self.model(xt[batch]) - yt[batch]) ** 2).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("custom NN produced nonfinite training loss")
                        loss.backward()
                        optimizer.step()
                    self.model.eval()
                    with torch.no_grad():
                        score = float(((self.model(xv) - yv) ** 2).mean())
                    if not np.isfinite(score):
                        raise ValueError("custom NN produced nonfinite validation loss")
                    if score < best - self.tol:
                        best, stale = score, 0
                        state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                    else:
                        stale += 1
                    if stale >= self.patience:
                        break
                if state is None:
                    raise ValueError("custom NN did not produce a valid fitted state")
                self.model.load_state_dict(state)
                self.model.eval()
                self.epochs_run = epoch + 1
                self.best_validation_mse = best
            finally:
                torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(np.asarray(X), dtype=torch.float32, device="cpu")).numpy()


def _parameters(parameters, policy):
    p = deepcopy(parameters or {})
    aliases = {"linearcc": "linear", "mlp": "nn"}
    method = str(p.get("mtd", "linear")).lower()
    method = aliases.get(method, method)
    if method not in ("linear", "pls", "nn", "custom"):
        raise ValueError("unsupported regression method: " + method)
    if p.get("param_search", False) is not False:
        raise ValueError("param_search is not implemented; select parameters explicitly")
    common = {"mtd", "degree", "random_state", "param_search", "use_lut", "lazy_lut", "lut_grid_size", "lut_min_pixels", "device"}
    specific = {"linear": set(), "pls": {"ncomp", "max_iterations", "tol"},
                "nn": {"max_iterations", "tol", "verbose", "hidden_layers", "learning_rate", "batch_size", "patience"},
                "custom": {"max_iterations", "tol", "hidden_layers", "learning_rate", "batch_size", "patience", "dropout_rate", "optim_type", "use_batch_norm"}}
    supported = common | specific[method]
    unknown = set(p) - supported
    if unknown:
        raise ValueError("unsupported reproducible fitting settings: " + repr(sorted(unknown)))
    if p.get("device", "cpu") != "cpu":
        raise ValueError("reproducible-v1 fitting is qualified on CPU; device must be cpu")
    p["mtd"] = method
    p["degree"] = _integer(p.get("degree", 3), "degree", maximum=5)
    # Sentinels are intentionally rejected: every effective stopping cap is explicit.
    p["max_iterations"] = _integer(p.get("max_iterations", 1000), "max_iterations")
    p["random_state"] = _integer(p.get("random_state", policy.random_state), "random_state", 0, 2**32-1)
    p["tol"] = _number(p.get("tol", 1e-6), "tol")
    p["learning_rate"] = _number(p.get("learning_rate", .001), "learning_rate")
    p["batch_size"] = _integer(p.get("batch_size", 16), "batch_size")
    p["patience"] = _integer(p.get("patience", 10), "patience")
    layers = p.get("hidden_layers", [64])
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("hidden_layers must be a nonempty sequence of positive integers")
    p["hidden_layers"] = [_integer(x, "hidden_layers") for x in layers]
    p["dropout_rate"] = _number(p.get("dropout_rate", .2), "dropout_rate", inclusive=True)
    if p["dropout_rate"] >= 1:
        raise ValueError("dropout_rate must be in [0,1)")
    p["optim_type"] = p.get("optim_type", "Adam")
    if p["optim_type"] not in ("Adam", "SGD", "RMSprop"):
        raise ValueError("unsupported optimizer")
    for name, default in (("verbose", False), ("use_batch_norm", False), ("use_lut", True), ("lazy_lut", True)):
        p[name] = _boolean(p.get(name, default), name)
    p["lut_grid_size"] = _integer(p.get("lut_grid_size", 33), "lut_grid_size", 2, 129)
    p["lut_min_pixels"] = _integer(p.get("lut_min_pixels", 4096), "lut_min_pixels")
    p["_effective_keys"] = sorted(supported - {"device"})
    return p


def fit_observations(X, Y, groups, parameters=None, policy=None, synthetic_mask=None):
    """Fit equally weighted cell means; optional synthetic anchors stay explicit.

    Linear fitting supports positive anchor_weight using weighted least squares.
    Nonlinear methods reject nonzero anchor weights rather than silently ignoring
    weights. No automatic synthetic rows are introduced.
    """
    from .core.correction import Regressor_Model
    from .core.accel import build_3d_lut
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neural_network import MLPRegressor
    policy = FittingPolicy.from_value(policy)
    if policy.version != "reproducible-v1":
        raise ValueError("fit_observations requires reproducible-v1")
    original_count = len(X)
    X, Y, groups, synthetic, counts = collapse_cell_observations(X, Y, groups, synthetic_mask)
    p = _parameters(parameters or {}, policy)
    retained = ~synthetic | (policy.anchor_weight > 0)
    X, Y, groups, synthetic = X[retained], Y[retained], groups[retained], synthetic[retained]
    if np.count_nonzero(~synthetic) < 3:
        raise ValueError("at least three independent measured cells are required")
    features = PolynomialFeatures(degree=p["degree"], include_bias=True).fit_transform(X)
    design = _design_report(features, policy, require_full_rank=p["mtd"] == "linear")
    if policy.anchor_weight and p["mtd"] != "linear":
        raise ValueError("nonlinear fitting does not support synthetic anchor weighting; use anchor_weight=0")
    M = Regressor_Model()
    for key, value in p.items():
        if key not in ("device", "lut_grid_size", "_effective_keys"):
            setattr(M, key, deepcopy(value))
    M._lut_grid_size = p["lut_grid_size"]
    M.fitting_version = policy.version
    if p["mtd"] == "linear":
        M.model = LinearRegression(fit_intercept=False)
        weights = np.where(synthetic, policy.anchor_weight, 1.0)
        weighted_design = _design_report(features * np.sqrt(weights)[:, None], policy, require_full_rank=True)
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_targets = Y * np.sqrt(weights)[:, None]
        if not np.all(np.isfinite(weighted_targets)):
            raise ValueError("weighted reference targets must be finite before solving")
        M.model.fit(features, Y, sample_weight=weights)
    elif p["mtd"] == "pls":
        # Centering removes the bias column; rank after centering bounds components.
        centered_rank = int(np.linalg.matrix_rank(features - features.mean(axis=0)))
        maximum = min(len(features)-1, centered_rank)
        ncomp = _integer(p.get("ncomp", min(3, maximum)), "ncomp")
        if ncomp > maximum:
            raise ValueError("ncomp exceeds centered design rank or independent-cell count")
        M.ncomp = ncomp
        p["ncomp"] = ncomp
        M.model = PLSRegression(n_components=ncomp, max_iter=p["max_iterations"], tol=p["tol"])
        M.model.fit(features, Y)
    elif p["mtd"] == "nn":
        M.model = MLPRegressor(hidden_layer_sizes=tuple(p["hidden_layers"]), activation="relu", solver="adam",
                               learning_rate_init=p["learning_rate"], max_iter=p["max_iterations"],
                               batch_size=p["batch_size"], random_state=p["random_state"], tol=p["tol"],
                               n_iter_no_change=p["patience"], shuffle=True, early_stopping=False, verbose=p["verbose"])
        M.model.fit(features, Y)
    else:
        M.model = DeterministicTorchRegressor(hidden_layers=p["hidden_layers"], learning_rate=p["learning_rate"],
                    max_epochs=p["max_iterations"], batch_size=p["batch_size"], patience=p["patience"], tol=p["tol"],
                    dropout_rate=p["dropout_rate"], use_batch_norm=p["use_batch_norm"], optim_type=p["optim_type"], random_state=p["random_state"])
        M.model.fit(features, Y, groups, synthetic)
    prediction = M.model.predict(features)
    if prediction.shape != Y.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("fitted estimator produced malformed or nonfinite predictions")
    effective = {key: p[key] for key in p["_effective_keys"] if key in p}
    effective["device"] = "cpu"
    dependencies = {"numpy": str(np.__version__), "scikit-learn": package_version("scikit-learn")}
    if p["mtd"] == "custom":
        import torch
        dependencies["torch"] = str(torch.__version__)
        estimator_parameters = {key: getattr(M.model, key) for key in (
            "hidden_layers", "learning_rate", "max_epochs", "batch_size", "patience", "tol",
            "dropout_rate", "use_batch_norm", "optim_type", "random_state", "device")}
        estimator_parameters["optimizer_defaults"] = M.model.optimizer_parameters
    else:
        estimator_parameters = M.model.get_params(deep=False)
    M.fit_report = {"version": policy.version, "policy": policy.to_dict(), "effective_parameters": effective,
                    "input_rows": original_count, "unique_measured_cells": int(np.count_nonzero(~synthetic)),
                    "synthetic_training_groups": int(synthetic.sum()), "retained_groups": groups.tolist(),
                    "group_row_counts_before_collapse": counts, "weighting": "one observed mean per canonical cell",
                    "design": design, "feature_order": "sklearn PolynomialFeatures include_bias=True, CPU float64",
                    "estimator_precision": "CPU float32" if p["mtd"] == "custom" else "CPU float64",
                    "estimator_parameters": _plain(estimator_parameters), "dependencies": dependencies,
                    "device": "cpu", "selection_scope": "training objective" if p["mtd"] != "custom" else "grouped internal CC-only holdout"}
    # RGB fitting and perceptual evaluation have different objectives. Record
    # unreachable sRGB targets without clipping canonical reference truth.
    M.fit_report["fitting_domain"] = "encoded-rgb"
    M.fit_report["target_domain"] = "unclipped-encoded-srgb"
    outside = np.any((Y < 0) | (Y > 1), axis=1) & ~synthetic
    M.fit_report["out_of_gamut_targets"] = {
        "cell_ids": groups[outside].tolist(),
        "channel_min": float(Y.min()), "channel_max": float(Y.max())}
    if p["mtd"] == "linear":
        M.fit_report["objective"] = {
            "metric": "weighted least squares", "domain": "encoded-srgb",
            "optimizes_delta_e00": False}
    if hasattr(M.model, "split_report"):
        M.fit_report["internal_split"] = deepcopy(M.model.split_report)
    if p["mtd"] == "linear":
        M.fit_report["weighted_design"] = weighted_design
    json.dumps(M.fit_report, allow_nan=False)
    M._lut_predict_fn = M._predict_for_lut if M.use_lut else None
    if M.use_lut and not M.lazy_lut:
        M._lut = build_3d_lut(M._lut_predict_fn, grid_size=M._lut_grid_size)
        from .core.lut_cache import prime_lut_identity
        prime_lut_identity(M)
    return M


def fit_conventional_observations(X, Y, groups, parameters=None, policy=None, synthetic_mask=None):
    """Stable Finlayson least squares with explicit root-polynomial features."""
    import colour
    policy = FittingPolicy.from_value(policy)
    if policy.version != "reproducible-v1":
        raise ValueError("conventional observation fitting requires reproducible-v1")
    p = dict(parameters or {})
    unknown = set(p) - {"method", "degree", "root_polynomial_expansion"}
    if unknown:
        raise ValueError("unsupported conventional settings: " + repr(sorted(unknown)))
    if p.get("method", "Finlayson 2015") != "Finlayson 2015":
        raise ValueError("modern conventional fitting currently supports Finlayson 2015")
    p = {"method": "Finlayson 2015", "degree": _integer(p.get("degree", 3), "degree", 1, 4),
         "root_polynomial_expansion": _boolean(p.get("root_polynomial_expansion", True), "root_polynomial_expansion")}
    original_count = len(X)
    X, Y, groups, synthetic, counts = collapse_cell_observations(X, Y, groups, synthetic_mask)
    retained = ~synthetic | (policy.anchor_weight > 0)
    X, Y, groups, synthetic = X[retained], Y[retained], groups[retained], synthetic[retained]
    if np.count_nonzero(~synthetic) < 3:
        raise ValueError("at least three independent measured cells are required")
    features = colour.characterisation.polynomial_expansion_Finlayson2015(X, degree=p["degree"], root_polynomial_expansion=p["root_polynomial_expansion"])
    design = _design_report(features, policy, require_full_rank=True)
    weights = np.sqrt(np.where(synthetic, policy.anchor_weight, 1.0))[:, None]
    weighted_design = _design_report(features * weights, policy, require_full_rank=True)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_targets = Y * weights
    if not np.all(np.isfinite(weighted_targets)):
        raise ValueError("weighted reference targets must be finite before solving")
    coefficients = np.linalg.lstsq(features * weights, weighted_targets, rcond=None)[0]
    ccm = coefficients.T
    if not np.all(np.isfinite(ccm)):
        raise ValueError("conventional fitting produced nonfinite coefficients")
    return ccm, p, {"version": policy.version, "policy": policy.to_dict(), "effective_parameters": p,
                    "input_rows": original_count, "unique_measured_cells": int(np.count_nonzero(~synthetic)),
                    "synthetic_training_groups": int(synthetic.sum()), "retained_groups": groups.tolist(),
                    "group_row_counts_before_collapse": counts, "weighting": "one observed mean per canonical cell",
                    "design": design, "weighted_design": weighted_design,
                    "solver": "numpy.linalg.lstsq", "device": "cpu",
                    "estimator_precision": "CPU float64", "dependencies": {
                        "numpy": str(np.__version__), "colour-science": str(colour.__version__)}}
