"""Explicit linear-light fitting compatible with ColorCorrect Studio's maps.

Port of the MIT-licensed ccp_core maps, PLS and MLP algorithms from
ColorCorrectionPackage_UI. Copyright (c) 2026 Collins Wakholi; see LICENSE.
Legacy and reproducible-v1 estimators do not enter this module implicitly.
"""

from copy import deepcopy
from types import MappingProxyType
import math

import numpy as np

from .colorimetry import srgb_to_linear, linear_to_srgb, xyz_to_rgb, lab_to_xyz, adapt_xyz, D65
from .portable_inference import owned_array

VERSION = "ui-compatible-v1"
_KINDS = ("ccm", "rpcc2", "rpcc3", "linear", "pls", "nn")
_FEATURE_LIMIT = 96
_RIDGE = 1e-5
_NN_ALPHA = 1e-4


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("%s must be an integer in [%d,%d]" % (name, low, high))
    return value


def _real(value, name, low=0.0, high=1.0, inclusive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or (value < low if inclusive else value <= low)
        or value > high
    ):
        raise ValueError("invalid finite %s" % name)
    return float(value)


def options(parameters=None):
    """Normalize only effective controls; reject knobs the UI algorithm ignores."""
    p = deepcopy(dict(parameters or {}))
    for key in ("show", "get_deltaE"):
        if key in p and type(p.pop(key)) is not bool:
            raise ValueError(key + " must be boolean")
    cc_method = p.pop("cc_method", "ours")
    if cc_method not in ("conv", "ours"):
        raise ValueError("cc_method must be conv or ours")
    conventional = cc_method == "conv"
    if not conventional and ("method" in p or "root_polynomial_expansion" in p):
        raise ValueError("method/root settings require cc_method='conv'")
    method = p.pop("method", "Finlayson 2015")
    root = p.pop("root_polynomial_expansion", True)
    alias = p.pop("mtd", None)
    kind = p.pop("model", None)
    degree = p.pop("degree", 2)
    if kind is None:
        if conventional:
            if method != "Finlayson 2015" or type(root) is not bool:
                raise ValueError(
                    "UI conventional fitting requires explicit Finlayson/root settings"
                )
            kind = {1: "ccm", 2: "rpcc2", 3: "rpcc3"}.get(degree) if root else "linear"
        else:
            kind = (
                {"custom": "nn", "mlp": "nn", "linearcc": "linear"}.get(alias, alias)
                if alias
                else "rpcc2"
            )
    if kind not in _KINDS:
        raise ValueError("UI model must be ccm, rpcc2, rpcc3, linear, pls or nn")
    canonical_alias = {"custom": "nn", "mlp": "nn", "linearcc": "linear"}.get(alias, alias)
    expected_alias = "nn" if kind == "nn" else "pls" if kind == "pls" else "linear"
    if alias is not None and canonical_alias != expected_alias:
        raise ValueError("mtd contradicts the requested UI model")
    if conventional:
        expected_model = {1: "ccm", 2: "rpcc2", 3: "rpcc3"}.get(degree) if root else "linear"
        if method != "Finlayson 2015" or type(root) is not bool or kind != expected_model:
            raise ValueError("conventional settings contradict the requested UI model")
    degree = _integer(degree, "degree", 1, 7)
    allowed = {
        "use_lut",
        "lazy_lut",
        "lut_grid_size",
        "lut_min_pixels",
        "device",
        "random_state",
        "include_combinations",
    }
    if kind == "pls":
        allowed |= {"ncomp", "max_iterations", "tol"}
    elif kind == "nn":
        allowed |= {
            "hidden_layers",
            "max_iterations",
            "patience",
            "learning_rate",
            "tol",
            "batch_size",
            "optim_type",
            "dropout_rate",
            "use_batch_norm",
            "verbose",
        }
    if set(p) - allowed:
        raise ValueError("unsupported UI fitting settings: " + repr(sorted(set(p) - allowed)))
    if p.get("device", "cpu") != "cpu" or p.get("random_state", 42) != 42:
        raise ValueError("UI-compatible training uses CPU and fixed seed 42")
    combinations = p.get("include_combinations", True)
    if type(combinations) is not bool or (kind != "linear" and combinations is not True):
        raise ValueError(
            "include_combinations=False is supported only by UI polynomial linear fitting"
        )
    result = {
        "model": kind,
        "degree": degree,
        "include_combinations": combinations,
        "mtd": "nn" if kind == "nn" else "pls" if kind == "pls" else "linear",
        "use_lut": p.get("use_lut", True),
        "lazy_lut": p.get("lazy_lut", True),
        "lut_grid_size": p.get("lut_grid_size", 33),
        "lut_min_pixels": p.get("lut_min_pixels", 1),
    }
    for key in ("use_lut", "lazy_lut"):
        if type(result[key]) is not bool:
            raise ValueError(key + " must be boolean")
    _integer(result["lut_grid_size"], "lut_grid_size", 2, 129)
    _integer(result["lut_min_pixels"], "lut_min_pixels", 1, 2**31 - 1)
    if kind == "pls":
        if p.get("max_iterations", 500) != 500:
            raise ValueError("UI PLS uses a fixed 500-iteration inner cap")
        result.update(
            ncomp=_integer(p.get("ncomp", 0), "ncomp", 0, 10000),
            tol=_real(p.get("tol", 1e-6), "tol"),
            max_iterations=500,
        )
    if kind == "nn":
        if p.get("verbose", False) is not False:
            raise ValueError("UI-compatible NN fitting does not support verbose output")
        layers = p.get("hidden_layers", [100])
        if not isinstance(layers, (tuple, list)) or not 1 <= len(layers) <= 4:
            raise ValueError("UI NN requires one to four hidden layers")
        layers = [_integer(x, "hidden width", 4, 512) for x in layers]
        optimizer = str(p.get("optim_type", "Adam")).lower()
        if optimizer not in ("adam", "sgd", "rmsprop"):
            raise ValueError("unsupported UI NN optimizer")
        batch_norm = p.get("use_batch_norm", False)
        if type(batch_norm) is not bool:
            raise ValueError("use_batch_norm must be boolean")
        result.update(
            hidden_layers=layers,
            max_iterations=_integer(p.get("max_iterations", 10000), "max_iterations", 1, 10000000),
            patience=_integer(p.get("patience", 1500), "patience", 1, 10000000),
            learning_rate=_real(
                p.get("learning_rate", 0.001), "learning_rate", low=1e-8, high=100.0, inclusive=True
            ),
            tol=_real(p.get("tol", 1e-6), "tol"),
            optim_type=optimizer,
            dropout_rate=_real(
                p.get("dropout_rate", 0.0), "dropout_rate", high=0.8, inclusive=True
            ),
            use_batch_norm=batch_norm,
        )
        if "batch_size" in p:
            result["batch_size"] = _integer(p["batch_size"], "batch_size", 1, 200)
    return result


def features(linear_rgb, settings):
    """Native feature order, including the native 96-feature ceiling."""
    values = np.maximum(np.asarray(linear_rgb, dtype=np.float64), 0)
    r, g, b = np.moveaxis(values, -1, 0)
    kind = settings["model"]
    columns = [r, g, b]
    if kind in ("rpcc2", "rpcc3"):
        columns += [np.sqrt(r * g), np.sqrt(r * b), np.sqrt(g * b)]
        if kind == "rpcc3":
            columns += [
                np.cbrt(r * r * g),
                np.cbrt(g * g * b),
                np.cbrt(b * b * r),
                np.cbrt(r * g * g),
                np.cbrt(g * b * b),
                np.cbrt(b * r * r),
                np.cbrt(r * g * b),
            ]
    elif kind != "ccm":
        columns = []
        for total in range(0 if kind in ("pls", "nn") else 1, settings["degree"] + 1):
            for i in range(total, -1, -1):
                for j in range(total - i, -1, -1):
                    k = total - i - j
                    if (
                        kind == "linear"
                        and not settings["include_combinations"]
                        and sum(x > 0 for x in (i, j, k)) > 1
                    ):
                        continue
                    columns.append(r**i * g**j * b**k)
        columns = columns[:_FEATURE_LIMIT]
    return np.stack(columns, axis=-1)


def _solve(matrix, rhs):
    """Solve the same normal systems, retaining the UI's singular PLS fallback."""
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix + np.eye(len(matrix)) * 1e-12, rhs)


def fit_pls(x, y, ncomp, tol):
    """UI NIPALS: population scaling and convergence on the updated score vector."""
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    xstd, ystd = x.std(axis=0), y.std(axis=0)
    xstd = np.where(xstd < 1e-12, 1.0, xstd)
    ystd = np.where(ystd < 1e-12, 1.0, ystd)
    a, b = (x - xmean) / xstd, (y - ymean) / ystd
    n, p = x.shape
    k = min(p, n, ncomp if ncomp else max(1, p - 1))
    weights, loadings, targets, iterations = [], [], [], []
    for _ in range(k):
        u = b[:, 0].copy()
        for iteration in range(500):
            if u @ u < 1e-300:
                break
            w = a.T @ u
            norm = np.linalg.norm(w)
            if norm < 1e-300:
                break
            w /= norm
            t = a @ w
            if t @ t < 1e-300:
                break
            q = b.T @ t / (t @ t)
            if q @ q < 1e-300:
                break
            updated = b @ q / (q @ q)
            distance = np.linalg.norm(updated - u)
            u = updated
            if distance < tol:
                break
        if u @ u < 1e-300:
            break
        w = a.T @ u
        norm = np.linalg.norm(w)
        if norm < 1e-300:
            break
        w /= norm
        t = a @ w
        if t @ t < 1e-12:
            break
        loading = a.T @ t / (t @ t)
        target = b.T @ t / (t @ t)
        weights.append(w)
        loadings.append(loading)
        targets.append(target)
        iterations.append(iteration + 1)
        a -= np.outer(t, loading)
        b -= np.outer(t, target)
    if weights:
        w, pmat, qmat = (
            np.column_stack(weights),
            np.column_stack(loadings),
            np.column_stack(targets),
        )
        coefficients = (w @ _solve(pmat.T @ w, qmat.T)) * ystd / xstd[:, None]
    else:
        coefficients = np.zeros((p, y.shape[1]))
    return (
        coefficients,
        ymean - xmean @ coefficients,
        {"ncomp": k, "used_components": len(weights), "iterations": iterations},
    )


def _forward(x, weights, biases, batch_norm=False, dropout=0.0, rng=None, training=False):
    activations, caches = [x], []
    for index, (w, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ w + b
        if index == len(weights) - 1:
            activations.append(z)
            break
        inverse = np.ones(len(b))
        if batch_norm and len(x) > 1:
            mean = z.mean(axis=0)
            inverse = 1 / np.sqrt(z.var(axis=0) + 1e-5)
            z = (z - mean) * inverse
        normalized = z.copy()
        mask = np.ones_like(z)
        if training and dropout:
            mask = (rng.uniform(size=z.shape) < 1 - dropout) / (1 - dropout)
        activations.append(np.maximum(z, 0) * mask)
        caches.append((normalized, inverse, mask))
    return activations, caches


def _training_forward(x, weights, biases, batch_norm=False, dropout=0.0, rng=None, training=False):
    from .ui_nn import dense, sum_rows

    activations, caches = [x], []
    for index, (w, b) in enumerate(zip(weights, biases)):
        z = dense(activations[-1], w, b)
        if index == len(weights) - 1:
            activations.append(z)
            break
        inverse = np.ones(len(b))
        if batch_norm and len(x) > 1:
            mean = sum_rows(z) / len(z)
            inverse = 1 / np.sqrt(sum_rows((z - mean) * (z - mean)) / len(z) + 1e-5)
            z = (z - mean) * inverse
        normalized = z.copy()
        mask = np.ones_like(z)
        if training and dropout:
            mask = (rng.uniform(size=z.shape) < 1 - dropout) / (1 - dropout)
        activations.append(np.maximum(z, 0) * mask)
        caches.append((normalized, inverse, mask))
    return activations, caches


def _score(x, y, weights, biases, batch_norm):
    from .ui_nn import sum_rows

    output = _training_forward(x, weights, biases, batch_norm=batch_norm)[0][-1]
    residual = sum_rows((output - y) ** 2)
    total = sum_rows((y - sum_rows(y) / len(y)) ** 2)
    score = np.where(total > 0, 1 - residual / np.where(total > 0, total, 1), 0).mean()
    loss = float(residual.sum() / len(y))
    if not np.isfinite(score) or not math.isfinite(loss):
        raise ValueError("UI NN training produced nonfinite score/loss")
    return float(score), loss


def fit_nn(x, target, settings):
    """Seed-42 residual MLP, with native full-chart/tiny-set selection semantics."""
    from .ui_nn import dense, dot, sum_rows, backend

    rng = np.random.RandomState(42)
    dimensions = [x.shape[1]] + settings["hidden_layers"] + [3]
    weights, biases = [], []
    for a, b in zip(dimensions, dimensions[1:]):
        limit = math.sqrt(6.0 / (a + b))
        weights.append(rng.uniform(-limit, limit, (a, b)))
        biases.append(rng.uniform(-limit, limit, b))
    weights[-1].fill(0)
    biases[-1].fill(0)
    y = target - x[:, 1:4]
    train = np.arange(len(x))
    validation = train
    if len(x) >= 64 and not settings["use_batch_norm"]:
        shuffled = rng.permutation(len(x))
        split = min(len(x) - 1, max(1, int(math.ceil(len(x) * 0.15))))
        validation, train = shuffled[:split], shuffled[split:]
    batch = min(200, len(train))
    if settings.get("batch_size", batch) != batch:
        raise ValueError("UI NN batch_size must equal min(200, training cells) = %d" % batch)
    moments = [np.zeros_like(v) for v in weights + biases]
    variances = [np.zeros_like(v) for v in weights + biases]
    best, best_loss = _score(
        x[validation], y[validation], weights, biases, settings["use_batch_norm"]
    )
    best_w, best_b = [w.copy() for w in weights], [b.copy() for b in biases]
    stale, step = 0, 0
    rate = max(settings["learning_rate"], 1e-8)
    for epoch in range(settings["max_iterations"]):
        for start in range(0, len(train), batch):
            indices = train[start : start + batch]
            acts, caches = _training_forward(
                x[indices],
                weights,
                biases,
                settings["use_batch_norm"],
                settings["dropout_rate"],
                rng,
                True,
            )
            delta = (acts[-1] - y[indices]) / len(indices)
            gw, gb = [None] * len(weights), [None] * len(biases)
            for i in range(len(weights) - 1, -1, -1):
                gw[i] = dot(acts[i].T, delta) + _NN_ALPHA * weights[i] / len(train)
                gb[i] = sum_rows(delta)
                if i:
                    delta = dot(delta, weights[i].T)
                    normalized, inverse, mask = caches[i - 1]
                    delta *= mask
                    delta[normalized <= 0] = 0
                    if settings["use_batch_norm"] and len(indices) > 1:
                        delta = (
                            inverse
                            / len(indices)
                            * (
                                len(indices) * delta
                                - sum_rows(delta)
                                - normalized * sum_rows(delta * normalized)
                            )
                        )
            step += 1
            for index, (value, gradient) in enumerate(zip(weights + biases, gw + gb)):
                m, v = moments[index], variances[index]
                if settings["optim_type"] == "adam":
                    m *= 0.9
                    m += (1 - 0.9) * gradient
                    v *= 0.999
                    v += (1 - 0.999) * gradient * gradient
                    value -= rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                elif settings["optim_type"] == "sgd":
                    m *= 0.9
                    m += gradient
                    value -= rate * m
                else:
                    v *= 0.9
                    v += 0.1 * gradient * gradient
                    value -= rate * gradient / (np.sqrt(v) + 1e-8)
        score, loss = _score(
            x[validation], y[validation], weights, biases, settings["use_batch_norm"]
        )
        stale = 0 if score > best + settings["tol"] else stale + 1
        if score > best:
            best, best_loss = score, loss
            best_w, best_b = [w.copy() for w in weights], [b.copy() for b in biases]
        if stale > settings["patience"]:
            proposed = rate * 0.2
            if proposed <= 1e-6:
                break
            rate, stale = proposed, 0
    if settings["use_batch_norm"]:
        previous = x[train]
        for i in range(len(best_w) - 1):
            z = dense(previous, best_w[i], best_b[i])
            mean = sum_rows(z) / len(z)
            std = np.sqrt(sum_rows((z - mean) ** 2) / len(z) + 1e-5)
            best_w[i] /= std
            best_b[i] = (best_b[i] - mean) / std
            previous = np.maximum(dense(previous, best_w[i], best_b[i]), 0)
    return (
        best_w,
        best_b,
        {
            "epochs": epoch + 1,
            "best_score": best,
            "best_loss": best_loss,
            "training_rows": train.tolist(),
            "selection_rows": validation.tolist(),
            "selection_scope": (
                "training chart"
                if len(x) < 64 or settings["use_batch_norm"]
                else "internal holdout"
            ),
            "batch_size": batch,
            "seed": 42,
            "residual": True,
            "precision": "float64",
            "arithmetic": "native-ordered-f64-v1",
            "backend": backend(),
        },
    )


class UIPredictor:
    """Owned, whitelisted numerical parameters; no external runtime or pickle graph."""

    def __init__(self, settings, coefficients=None, intercept=None, weights=(), biases=()):
        if not isinstance(settings, (dict, MappingProxyType)):
            raise ValueError("UI model settings must be a dictionary")
        normalized = options(settings)
        canonical = {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()}
        if normalized != canonical:
            raise ValueError("UI model requires canonical effective settings")
        count = features(np.full((1, 3), 0.5), normalized).shape[1]
        self.settings = MappingProxyType(
            {k: tuple(v) if isinstance(v, list) else v for k, v in normalized.items()}
        )
        self.coefficients = None
        self.intercept = None
        self.weights, self.biases = (), ()

        def array(value, shape):
            a = np.asarray(value, dtype=np.float64)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("invalid UI parameter shape or nonfinite value")
            return owned_array(a)

        if normalized["model"] == "nn":
            dimensions = [count] + normalized["hidden_layers"] + [3]
            if (
                coefficients is not None
                or intercept is not None
                or len(weights) != len(dimensions) - 1
                or len(biases) != len(weights)
            ):
                raise ValueError("invalid UI NN parameter structure")
            self.weights = tuple(
                array(w, (a, b)) for w, a, b in zip(weights, dimensions, dimensions[1:])
            )
            self.biases = tuple(array(v, (b,)) for v, b in zip(biases, dimensions[1:]))
        else:
            if weights or biases:
                raise ValueError("UI affine model cannot contain network layers")
            self.coefficients = array(coefficients, (count, 3))
            self.intercept = array(intercept, (3,))

    def __reduce__(self):
        return (
            type(self),
            (dict(self.settings), self.coefficients, self.intercept, self.weights, self.biases),
        )

    def predict(self, design):
        values = np.asarray(design, dtype=np.float64)
        if self.settings["model"] == "nn":
            out = _forward(values, self.weights, self.biases)[0][-1]
            return out + values[:, 1:4]
        return values @ self.coefficients + self.intercept

    def predict_rgb(self, rgb):
        values = np.asarray(rgb, dtype=np.float64)
        if (
            values.shape[-1:] != (3,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError("UI prediction requires finite encoded RGB in [0,1]")
        shape = values.shape
        design = features(srgb_to_linear(values.reshape(-1, 3)), self.settings)
        linear = np.clip(self.predict(design), 0, 1)
        return linear_to_srgb(linear).reshape(shape)


def fit_ui(X, Y, parameters=None):
    """Fit encoded input means against unclipped linear-light RGB targets."""
    settings = options(parameters)
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if (
        X.ndim != 2
        or X.shape[1] != 3
        or X.shape != Y.shape
        or not 3 <= len(X) <= 65536
        or not np.isfinite(X).all()
        or not np.isfinite(Y).all()
        or np.any(X < 0)
        or np.any(X > 1)
    ):
        raise ValueError(
            "UI fit requires finite paired RGB triples and at least three observations"
        )
    design = features(srgb_to_linear(X), settings)
    kind = settings["model"]
    detail = {}
    if kind == "nn":
        from .ui_nn import design as training_design

        design = training_design(X, settings["degree"])
        weights, biases, detail = fit_nn(design, Y, settings)
        predictor = UIPredictor(settings, weights=weights, biases=biases)
    elif kind == "pls":
        coefficients, intercept, detail = fit_pls(design, Y, settings["ncomp"], settings["tol"])
        predictor = UIPredictor(settings, coefficients, intercept)
    else:
        matrix = np.column_stack((design, np.ones(len(design)))) if kind == "linear" else design
        coefficients = _solve(matrix.T @ matrix + np.eye(matrix.shape[1]) * _RIDGE, matrix.T @ Y)
        intercept = coefficients[-1] if kind == "linear" else np.zeros(3)
        predictor = UIPredictor(
            settings, coefficients[:-1] if kind == "linear" else coefficients, intercept
        )
    return predictor, {
        "version": VERSION,
        "effective_parameters": settings,
        "input_rows": len(X),
        "feature_count": design.shape[1],
        "feature_order": "ui/" + settings["model"] + "/linear-input/v1",
        "fitting_domain": "linear-rgb",
        "target_domain": "unclipped-linear-srgb",
        "ridge": _RIDGE if kind not in ("pls", "nn") else 0.0,
        "solver": (
            "native-nipals"
            if kind == "pls"
            else "native-residual-mlp" if kind == "nn" else "ridge-normal-equations"
        ),
        "estimator_precision": "CPU float64",
        "details": detail,
    }


def is_ui(models):
    return getattr(models, "metadata", {}).get("fitting_policy", {}).get("version") == VERSION


def round_image(rgb):
    """Represent the native float32 stage boundary in owned float64 RGB."""
    return np.asarray(rgb, dtype=np.float32).astype(np.float64)


def sample_regions(rgb, quads, samples_per_axis=9):
    """Sample the UI's inner 70% on its native odd/even per-axis grid."""
    count = _integer(samples_per_axis, "samples_per_axis", 1, 63)
    image = np.asarray(rgb, dtype=np.float32)
    q = np.asarray(quads, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or q.ndim != 3 or q.shape[1:] != (4, 2):
        raise ValueError("invalid UI sampling image or quadrilaterals")
    if (
        min(image.shape[:2]) < 1
        or not np.isfinite(q).all()
        or not np.isfinite(image).all()
        or np.any(image < 0)
        or np.any(image > 1)
    ):
        raise ValueError("UI sampling requires normalized finite RGB")
    half = count // 2
    axis = 0.5 + (np.arange(count) - half) / max(half, 1) * 0.5 * 0.70
    u, v = np.meshgrid(axis, axis)
    u, v = u.ravel()[None, :, None], v.ravel()[None, :, None]
    quad = q[:, None]
    points = (
        (1 - u) * (1 - v) * quad[:, :, 0]
        + u * (1 - v) * quad[:, :, 1]
        + u * v * quad[:, :, 2]
        + (1 - u) * v * quad[:, :, 3]
    )
    h, w = image.shape[:2]
    x, y = np.clip(points[..., 0], 0, w - 1), np.clip(points[..., 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = (x - x0)[..., None], (y - y0)[..., None]
    top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
    return top * (1 - dy) + bottom * dy


def fit_stage(owner, image, method, parameters):
    from .modern import selected_face
    from .core.correction import Regressor_Model, predict_
    from .evaluation import grouped_cc_validation
    from .colorimetry import D50

    face = selected_face(owner, image, "CC_training")
    x = np.asarray(face.measured_rgb)
    truth = owner.reference_profile.canonical_lab_d50
    y = xyz_to_rgb(
        adapt_xyz(lab_to_xyz(truth, white=D50), D50, D65), encoding="linear-srgb", policy="preserve"
    )
    params = dict(parameters or {})
    params["cc_method"] = method
    from .ui_native import enabled as native_enabled, reference as native_reference

    if native_enabled(owner):
        y = native_reference(owner.reference_profile)[0]
    predictor, report = fit_ui(x, y, params)
    if native_enabled(owner):
        report["details"]["ui_runtime"] = "native-v1"
    settings = report["effective_parameters"]
    model = Regressor_Model()
    for name, value in settings.items():
        if name not in ("model", "include_combinations", "lut_grid_size"):
            setattr(model, name, deepcopy(value))
    model._lut_grid_size = settings["lut_grid_size"]
    model.fitting_version = VERSION
    model.model = predictor
    model.fit_report = report
    model._lut_predict_fn = model._predict_for_lut if model.use_lut else None
    owner.models.model_cc = (model, deepcopy(settings), "ours")
    selection = {
        "policy": "all measured cells",
        "source": "fixed UI sampling",
        "excluded_cell_ids": [],
        "reference_assisted_values": False,
        "independent_cells": len(x),
    }
    numerical = {
        "fit": report,
        "selection": selection,
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
    return round_image(np.clip(predict_(image, model), 0, 1)), {}, False


_UI_CLASSIC_LAB = (
    (37.54, 14.37, 14.92),
    (64.66, 19.27, 17.5),
    (49.32, -3.82, -22.54),
    (43.46, -12.74, 22.72),
    (54.94, 9.61, -24.79),
    (70.48, -32.26, -0.37),
    (62.73, 35.83, 56.5),
    (39.43, 10.75, -45.17),
    (50.57, 48.64, 16.67),
    (30.1, 22.54, -20.87),
    (71.77, -24.13, 58.19),
    (71.51, 18.24, 67.37),
    (28.37, 15.42, -49.8),
    (54.38, -39.72, 32.27),
    (42.43, 51.05, 28.62),
    (81.8, 2.67, 80.41),
    (50.63, 51.28, -14.12),
    (49.57, -29.71, -28.32),
    (95.19, -1.03, 2.93),
    (81.29, -0.57, 0.44),
    (66.89, -0.75, -0.06),
    (50.76, -0.13, 0.14),
    (35.63, -0.46, -0.48),
    (20.64, 0.07, -0.46),
)


def default_reference():
    """UI Classic24 truth with its declared white; do not relabel D50 values."""
    from .contracts import ReferenceProfile

    base = ReferenceProfile.classic24()
    lab = np.array(
        _UI_CLASSIC_LAB,
        dtype=np.float64,
    )
    white = (0.34567, 0.35850)
    xyz = adapt_xyz(lab_to_xyz(lab, white=white), white, D65)
    return ReferenceProfile.from_values(
        xyz,
        color_space="xyz",
        white=D65,
        scale=1,
        profile_id="ui-classic24",
        revision=1,
        names=base.names,
        cell_ids=base.cell_ids,
        neutral_ids=base.neutral_ids,
    )


def homography(source, destination):
    """Solve the UI geometry in float64 without OpenCV's float32 point cast."""
    matrix = np.zeros((8, 8), dtype=np.float64)
    rhs = np.asarray(destination, dtype=np.float64).reshape(-1)
    for index, ((x, y), (u, v)) in enumerate(zip(source, destination)):
        matrix[2 * index] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        matrix[2 * index + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
    try:
        return np.r_[np.linalg.solve(matrix, rhs), 1.0].reshape(3, 3)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan)
