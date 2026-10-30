"""Versioned calibration quality and neutral-stage policies.

These helpers consume measured observations. They never infer clipped values from
their targets, nor add synthetic endpoints before checking measured sufficiency.
The historical algorithms remain selected by ``legacy-v1`` at pipeline level.
"""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class StagePolicy:
    version: str = "legacy-v1"
    saturation: str = "exclude"
    max_clipped_fraction: float = 0.0
    gc_min_neutrals: int = 3
    gc_min_span: float = 20.0
    gc_duplicate_tolerance: float = 1e-6
    gc_target_tolerance: float = 1.0
    wb_headroom: float = 0.95
    wb_min_signal: float = 0.01
    wb_max_gain: float = 8.0

    def __post_init__(self):
        if self.version not in ("legacy-v1", "modern-v1"):
            raise ValueError("stage policy version must be legacy-v1 or modern-v1")
        if self.saturation not in ("report", "exclude"):
            raise ValueError("saturation policy must be report or exclude")
        if isinstance(self.gc_min_neutrals, bool) or not isinstance(self.gc_min_neutrals, (int, np.integer)) or self.gc_min_neutrals < 3:
            raise ValueError("gc_min_neutrals must be an integer >= 3")
        bounds = {"max_clipped_fraction": (0, 1, True), "gc_min_span": (0, 100, False),
                  "gc_duplicate_tolerance": (0, 1, False), "gc_target_tolerance": (0, 100, True),
                  "wb_headroom": (0, 1, False), "wb_min_signal": (0, 1, False),
                  "wb_max_gain": (0, 100, False)}
        for name, (lower, upper, include_lower) in bounds.items():
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (float, int, np.number)):
                raise TypeError(name + " must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value > upper or (value < lower if include_lower else value <= lower):
                raise ValueError(name + " is outside its supported range")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gc_min_neutrals", int(self.gc_min_neutrals))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(version=value)
        if not isinstance(value, dict):
            raise TypeError("stage_policy must be a version string, StagePolicy or dictionary")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("unknown stage policy fields: %s" % sorted(unknown))
        return cls(**value)


def _cells(cell_ids, neutral_ids=None):
    cells = tuple(cell_ids)
    if not cells or any(not isinstance(c, str) or not c for c in cells) or len(set(cells)) != len(cells):
        raise ValueError("cell_ids must be unique nonempty identifiers")
    if neutral_ids is None:
        return cells, None
    neutrals = tuple(neutral_ids)
    if len(neutrals) != len(set(neutrals)) or not set(neutrals).issubset(cells):
        raise ValueError("neutral_ids must uniquely identify measured cells")
    return cells, np.array([c in neutrals for c in cells], dtype=bool)


def _clipping(fractions, count):
    if fractions is None:
        return np.zeros(count, dtype=np.float64), None
    values = np.asarray(fractions, dtype=np.float64)
    if values.shape not in ((count,), (count, 3)) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("clipped_fractions must be finite per-cell or per-cell/channel fractions in [0,1]")
    return (values.max(axis=1) if values.ndim == 2 else values), values.tolist()


def saturation_selection(cell_ids, clipped_fractions, policy=None):
    """Select measured cells using original sampled-pixel clipping evidence.

    For channel fractions a cell is rejected when any channel exceeds the limit.
    Caller-supplied per-cell fractions can instead describe any-channel clipping.
    """
    policy = StagePolicy.from_value(policy)
    cells, _ = _cells(cell_ids)
    fractions, detail = _clipping(clipped_fractions, len(cells))
    clipped = fractions > policy.max_clipped_fraction
    allowed = ~clipped if policy.saturation == "exclude" else np.ones(len(cells), dtype=bool)
    return allowed, {"policy": policy.saturation, "basis": "sampled-pixels",
                     "max_clipped_fraction": policy.max_clipped_fraction,
                     "clipped_fractions": detail if detail is not None else fractions.tolist(),
                     "flagged_cell_ids": [c for c, flag in zip(cells, clipped) if flag],
                     "excluded_cell_ids": [c for c, keep in zip(cells, allowed) if not keep],
                     "estimated_values": False, "pixel_transform": "identity"}


def _owned(values):
    a = np.asarray(values, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class MonotonicToneModel:
    """Endpoint-constrained PCHIP in Lab L*, evaluated on [0,100]."""
    measured_knots: np.ndarray
    target_knots: np.ndarray
    version: str = "neutral-pchip-v1"

    def __post_init__(self):
        x, y = np.asarray(self.measured_knots, dtype=np.float64), np.asarray(self.target_knots, dtype=np.float64)
        if self.version != "neutral-pchip-v1" or x.ndim != 1 or y.shape != x.shape or len(x) < 5:
            raise ValueError("invalid neutral PCHIP model schema")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
            raise ValueError("tone knots must be finite, distinct and monotonic")
        if not (x[0] == y[0] == 0 and x[-1] == y[-1] == 100):
            raise ValueError("tone model endpoints must be 0->0 and 100->100")
        object.__setattr__(self, "measured_knots", _owned(x))
        object.__setattr__(self, "target_knots", _owned(y))

    def predict(self, lightness):
        values = np.asarray(lightness, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("tone input lightness must be finite")
        result = PchipInterpolator(self.measured_knots, self.target_knots, extrapolate=False)(np.clip(values, 0, 100))
        # Polynomial endpoint evaluation can round just below 100. These are
        # explicit model assumptions, so honor them exactly at the boundaries.
        return np.where(values <= 0, 0.0, np.where(values >= 100, 100.0, np.clip(result, 0, 100)))

    def __reduce__(self):
        return (type(self), (self.measured_knots, self.target_knots, self.version))


def fit_monotonic_tone(measured_L, reference_L, cell_ids, neutral_ids, *, clipped_fractions=None, policy=None):
    policy = StagePolicy.from_value(policy)
    cells, neutral = _cells(cell_ids, neutral_ids)
    x, y = np.asarray(measured_L, dtype=np.float64), np.asarray(reference_L, dtype=np.float64)
    if x.shape != (len(cells),) or y.shape != x.shape:
        raise ValueError("tone inputs must contain one measured/reference L* per cell")
    if not np.isfinite(y[neutral]).all() or np.any((y[neutral] < 0) | (y[neutral] > 100)):
        raise ValueError("tone inputs must contain one measured/reference L* per cell, with finite reference L* in [0,100]")
    allowed, saturation = saturation_selection(cells, clipped_fractions, policy)
    valid = neutral & allowed & np.isfinite(x) & (x > policy.gc_duplicate_tolerance) & (x < 100 - policy.gc_duplicate_tolerance)
    used = [c for c, keep in zip(cells, valid) if keep]
    excluded = [c for c, is_n, keep in zip(cells, neutral, valid) if is_n and not keep]
    x, y = x[valid], y[valid]
    if len(x) < policy.gc_min_neutrals:
        raise ValueError("GC requires at least %d usable measured neutral cells before endpoint anchors" % policy.gc_min_neutrals)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    groups = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] - x[start] > policy.gc_duplicate_tolerance:
            target = y[start:i]
            if np.ptp(target) > policy.gc_target_tolerance:
                raise ValueError("GC duplicate measured lightness has contradictory reference targets")
            groups.append((float(np.mean(x[start:i])), float(np.mean(target))))
            start = i
    x, y = np.asarray(groups).T
    if len(x) < policy.gc_min_neutrals or x[-1] - x[0] < policy.gc_min_span:
        raise ValueError("GC measured neutrals have insufficient distinct lightness or usable span before endpoint anchors")
    if np.any(np.diff(y) < 0):
        raise ValueError("GC measured neutral ordering contradicts monotonic reference lightness")
    model = MonotonicToneModel(np.r_[0.0, x, 100.0], np.r_[0.0, y, 100.0])
    report = {"algorithm": model.version, "used_cell_ids": used, "excluded_cell_ids": excluded,
              "measured_neutral_count": len(used), "distinct_measured_lightness_count": len(x),
              "measured_lightness_span": float(x[-1] - x[0]),
              "endpoint_assumptions": [[0.0, 0.0], [100.0, 100.0]],
              "duplicate_policy": "mean only when targets agree within declared tolerance",
              "saturation": saturation}
    return model, report


def fit_neutral_wb(measured_rgb, reference_rgb, cell_ids, neutral_ids, *, clipped_fractions=None, policy=None):
    policy = StagePolicy.from_value(policy)
    cells, neutral = _cells(cell_ids, neutral_ids)
    x, y = np.asarray(measured_rgb, dtype=np.float64), np.asarray(reference_rgb, dtype=np.float64)
    if x.shape != (len(cells), 3) or y.shape != x.shape:
        raise ValueError("WB inputs must contain one measured/reference RGB triple per cell")
    if not np.isfinite(y[neutral]).all():
        raise ValueError("WB inputs must contain one finite reference RGB triple per cell")
    allowed, saturation = saturation_selection(cells, clipped_fractions, policy)
    valid = neutral & allowed & np.isfinite(x).all(axis=1) & (x > policy.wb_min_signal).all(axis=1) & (x <= 1).all(axis=1) & (y > 0).all(axis=1)
    if np.count_nonzero(valid) < 2:
        raise ValueError("WB requires at least two usable named neutral cells above min_signal")
    gains = np.median(y[valid] / x[valid], axis=0) * policy.wb_headroom
    if not np.isfinite(gains).all() or np.any(gains <= 0) or np.any(gains > policy.wb_max_gain):
        raise ValueError("WB gains are nonfinite, nonpositive or exceed wb_max_gain")
    report = {"algorithm": "neutral-channel-ratio-median-v1", "headroom": policy.wb_headroom,
              "min_signal": policy.wb_min_signal, "max_gain": policy.wb_max_gain,
              "gains": gains.tolist(), "used_cell_ids": [c for c, keep in zip(cells, valid) if keep],
              "excluded_cell_ids": [c for c, is_n, keep in zip(cells, neutral, valid) if is_n and not keep],
              "saturation": saturation}
    return np.diag(gains), report
