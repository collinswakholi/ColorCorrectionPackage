"""Versioned, explicit color interpretation independent of legacy helpers.

All XYZ values are relative (Y of the reference white is 1), unless the caller
explicitly supplies scale=100. Lab uses L*=0..100. RGB primaries are sRGB; white
coordinates retain the package's CIE D65 convention rather than silently changing
existing model interpretation to another rounding. Reference truth is never
clipped. See docs/color-semantics.md for assumptions and independent fixtures.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "D50", "D65", "ColorSemantics", "white_xy", "white_xyz",
    "srgb_to_linear", "linear_to_srgb", "rgb_to_xyz_matrix", "rgb_to_xyz", "xyz_to_rgb",
    "adaptation_matrix", "adapt_xyz", "xyz_to_lab", "lab_to_xyz",
    "reference_values_to_lab_d50", "reference_to_working_lab", "reference_to_working_rgb", "display_reference_rgb",
    "delta_e00", "delta_e00_reference",
]

D65 = (0.31271, 0.32902)
D50 = (0.3457, 0.3585)
_EPSILON = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0
_CATS = {
    "Bradford": np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]),
    "CAT02": np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]]),
}


def _numeric(values, label="values"):
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("%s must contain real numbers" % label)
    result = np.asarray(source, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("%s must be nonempty and finite" % label)
    return result


def _triples(values, label="values"):
    result = _numeric(values, label)
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ValueError("%s must have a final dimension of three" % label)
    return result


def white_xy(white):
    """Normalize a named D50/D65 or explicitly supplied, finite xy white."""
    if isinstance(white, str):
        names = {"D50": D50, "D65": D65}
        if white.upper() not in names:
            raise ValueError("white must be D50, D65, or an explicit xy pair")
        return names[white.upper()]
    xy = _numeric(white, "white")
    if xy.shape != (2,) or np.any(xy <= 0) or np.any(xy >= 1) or xy.sum() >= 1:
        raise ValueError("white must be a finite xy pair with x>0, y>0 and x+y<1")
    return tuple(float(v) for v in xy)


def white_xyz(white):
    x, y = white_xy(white)
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _scale(value, allowed, label="scale"):
    if isinstance(value, (bool, str, bytes, complex, np.complexfloating)) or not np.isscalar(value):
        raise ValueError("%s must be one of %s" % (label, allowed))
    if not np.isfinite(value) or value not in allowed:
        raise ValueError("%s must be one of %s" % (label, allowed))
    return float(value)


def _domain(values, domain):
    if domain not in ("unit", "extended"):
        raise ValueError("RGB domain must be unit or extended")
    if domain == "unit" and (np.any(values < 0) or np.any(values > 1)):
        raise ValueError("RGB values must be in [0,1]; declare extended domain explicitly")


def _output(values, policy):
    if policy not in ("clip", "preserve", "raise"):
        raise ValueError("output policy must be clip, preserve, or raise")
    if not np.isfinite(values).all():
        raise ValueError("color conversion produced nonfinite values")
    if policy == "clip":
        return np.clip(values, 0.0, 1.0)
    if policy == "raise":
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("color is outside the unit RGB gamut")
        return np.clip(values, 0.0, 1.0)
    return values.copy()


def srgb_to_linear(values, *, domain="unit"):
    """Decode sRGB; extended domain uses the signed transfer extension."""
    rgb = _numeric(values, "sRGB")
    _domain(rgb, domain)
    absolute = np.abs(rgb)
    result = rgb / 12.92
    nonlinear = absolute > 0.04045
    result = np.array(result, copy=True)
    result[nonlinear] = np.sign(rgb[nonlinear]) * ((absolute[nonlinear] + 0.055) / 1.055) ** 2.4
    return _output(result, "preserve")


def linear_to_srgb(values, *, policy="preserve"):
    """Encode linear sRGB, preserving out-of-gamut values unless requested."""
    linear = _numeric(values, "linear RGB")
    absolute = np.abs(linear)
    result = linear * 12.92
    nonlinear = absolute > 0.0031308
    result = np.array(result, copy=True)
    result[nonlinear] = np.sign(linear[nonlinear]) * (1.055 * absolute[nonlinear] ** (1.0 / 2.4) - 0.055)
    return _output(result, policy)


def rgb_to_xyz_matrix():
    """sRGB primaries scaled to the package's declared D65 xy white."""
    primaries = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
    basis = np.stack([primaries[:, 0] / primaries[:, 1], np.ones(3),
                      (1 - primaries[:, 0] - primaries[:, 1]) / primaries[:, 1]])
    return basis @ np.diag(np.linalg.solve(basis, white_xyz(D65)))


_RGB_TO_XYZ = rgb_to_xyz_matrix()
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)


def rgb_to_xyz(values, *, encoding="srgb", domain="unit"):
    """Convert declared sRGB-primary pixels to relative XYZ at D65."""
    rgb = _triples(values, "RGB")
    _domain(rgb, domain)
    if encoding == "srgb":
        rgb = srgb_to_linear(rgb, domain=domain)
    elif encoding != "linear-srgb":
        raise ValueError("encoding must be srgb or linear-srgb")
    return rgb @ _RGB_TO_XYZ.T


def xyz_to_rgb(values, *, encoding="srgb", xyz_scale=1, policy="preserve"):
    """Convert XYZ already at D65 to RGB; this function does not adapt white."""
    xyz = _triples(values, "XYZ") / _scale(xyz_scale, (1, 100), "XYZ scale")
    linear = xyz @ _XYZ_TO_RGB.T
    if encoding == "srgb":
        return linear_to_srgb(linear, policy=policy)
    if encoding == "linear-srgb":
        return _output(linear, policy)
    raise ValueError("encoding must be srgb or linear-srgb")


def adaptation_matrix(source_white, target_white, *, transform="Bradford"):
    """Build an explicit Von Kries CAT matrix, retaining full inverse precision."""
    if transform not in _CATS:
        raise ValueError("adaptation transform must be Bradford or CAT02")
    source, target = white_xy(source_white), white_xy(target_white)
    if source == target:
        return np.eye(3)
    cones = _CATS[transform]
    original, destination = cones @ white_xyz(source), cones @ white_xyz(target)
    if np.any(original <= 0) or np.any(destination <= 0):
        raise ValueError("white points must have positive cone responses for the selected CAT")
    return np.linalg.solve(cones, np.diag(destination / original) @ cones)


def adapt_xyz(values, source_white, target_white, *, transform="Bradford"):
    xyz = _triples(values, "XYZ")
    result = xyz @ adaptation_matrix(source_white, target_white, transform=transform).T
    if not np.isfinite(result).all():
        raise ValueError("chromatic adaptation produced nonfinite XYZ")
    return result


def xyz_to_lab(values, *, white="D50", scale=1):
    xyz = _triples(values, "XYZ") / _scale(scale, (1, 100), "XYZ scale")
    relative = xyz / white_xyz(white)
    f = np.where(relative > _EPSILON, np.cbrt(relative), (_KAPPA * relative + 16) / 116)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def lab_to_xyz(values, *, white="D50", scale=1):
    lab = _triples(values, "Lab")
    factor = _scale(scale, (1, 100), "XYZ scale")
    lightness = lab[..., 0] / 116
    a, b = lab[..., 1] / 500, lab[..., 2] / 200
    fy = lightness + 16 / 116
    fx, fz = fy + a, fy - b
    x = np.where(fx ** 3 > _EPSILON, fx ** 3, (lightness + a) * 116 / _KAPPA)
    y = np.where(lab[..., 0] > _KAPPA * _EPSILON, fy ** 3, lab[..., 0] / _KAPPA)
    z = np.where(fz ** 3 > _EPSILON, fz ** 3, (lightness - b) * 116 / _KAPPA)
    result = np.stack((x, y, z), axis=-1) * white_xyz(white) * factor
    if not np.isfinite(result).all():
        raise ValueError("Lab conversion produced nonfinite XYZ")
    return result


def reference_values_to_lab_d50(values, *, color_space, white, scale, adaptation="Bradford"):
    """Canonicalize explicitly described reference values without display clipping.

    Lab uses L*=0..100 and scale=1. XYZ permits scales1/100; RGB permits1/255/65535.
    RGB has sRGB primaries and the package D65 white; another white must be supplied
    as XYZ or Lab rather than relabeling a standard RGB encoding.
    """
    value = _triples(values, "reference values")
    source_white = white_xy(white)
    if color_space == "lab":
        _scale(scale, (1,), "Lab scale")
        if np.any(value[..., 0] < 0) or np.any(value[..., 0] > 100):
            raise ValueError("reference Lab lightness must be in [0,100]")
        if source_white == D50:
            # Preserve supplied canonical truth exactly, without a round trip.
            adaptation_matrix(source_white, D50, transform=adaptation)
            return value.copy()
        xyz = lab_to_xyz(value, white=source_white)
    elif color_space == "xyz":
        xyz = value / _scale(scale, (1, 100), "XYZ scale")
        if np.any(xyz < 0) or np.any(xyz[..., 1] > 1):
            raise ValueError("reference XYZ must be nonnegative with relative Y in [0,1]")
    elif color_space in ("srgb", "linear-srgb"):
        if source_white != D65:
            raise ValueError("sRGB-primary reference encoding requires the declared D65 white")
        normalized = value / _scale(scale, (1, 255, 65535), "RGB scale")
        xyz = rgb_to_xyz(normalized, encoding=color_space)
    else:
        raise ValueError("reference color_space must be lab, xyz, srgb, or linear-srgb")
    return xyz_to_lab(adapt_xyz(xyz, source_white, D50, transform=adaptation), white=D50)


def reference_to_working_lab(lab_d50, *, adaptation="Bradford"):
    """Adapt canonical truth directly to working Lab, avoiding RGB gamut effects.

    In particular, GC lightness targets must not round-trip out-of-gamut
    reference RGB through a legacy decoder with a different negative extension.
    """
    xyz = adapt_xyz(lab_to_xyz(lab_d50, white=D50), D50, D65, transform=adaptation)
    return xyz_to_lab(xyz, white=D65)


def reference_to_working_rgb(lab_d50, *, adaptation="Bradford", clip=False):
    """Encode canonical reference truth in working RGB; preserve gamut by default."""
    if not isinstance(clip, bool):
        raise TypeError("clip must be boolean")
    xyz_d50 = lab_to_xyz(lab_d50, white=D50)
    xyz_d65 = adapt_xyz(xyz_d50, D50, D65, transform=adaptation)
    return xyz_to_rgb(xyz_d65, policy="clip" if clip else "preserve")


def display_reference_rgb(lab_d50, *, adaptation="Bradford", clip=True):
    """Display swatches are a derived view; they must never replace Lab truth."""
    return reference_to_working_rgb(lab_d50, adaptation=adaptation, clip=clip)


def delta_e00(lab_a, lab_b):
    """CIEDE2000, kL=kC=kH=1, on same-white Lab in L*=0..100 units."""
    first, second = _triples(lab_a, "first Lab"), _triples(lab_b, "second Lab")
    if first.shape != second.shape:
        raise ValueError("Delta-E requires same-shape, same-cell Lab arrays")
    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)

    def chroma_ratio(c):
        with np.errstate(divide="ignore", over="ignore"):
            return 1 / (1 + (25 / c) ** 7)

    g = 0.5 * (1 - np.sqrt(chroma_ratio((c1 + c2) / 2)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    h1, h2 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360), np.mod(np.degrees(np.arctan2(b2, ap2)), 360)
    no_chroma = (cp1 == 0) | (cp2 == 0)
    dh = h2 - h1
    dh = np.where(no_chroma, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dl, dc = l2 - l1, cp2 - cp1
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))
    average_l, average_c = (l1 + l2) / 2, (cp1 + cp2) / 2
    sum_h = h1 + h2
    average_h = np.where(no_chroma, sum_h, np.where(np.abs(h1 - h2) <= 180, sum_h / 2,
                          np.where(sum_h < 360, (sum_h + 360) / 2, (sum_h - 360) / 2)))
    radians = np.radians(average_h)
    t = 1 - 0.17 * np.cos(radians - np.radians(30)) + 0.24 * np.cos(2 * radians) + 0.32 * np.cos(3 * radians + np.radians(6)) - 0.20 * np.cos(4 * radians - np.radians(63))
    sl = 1 + 0.015 * (average_l - 50) ** 2 / np.sqrt(20 + (average_l - 50) ** 2)
    sc, sh = 1 + 0.045 * average_c, 1 + 0.015 * average_c * t
    rotation = -2 * np.sqrt(chroma_ratio(average_c)) * np.sin(np.radians(60 * np.exp(-((average_h - 275) / 25) ** 2)))
    normalized_l, normalized_c, normalized_h = dl / sl, dc / sc, d_h / sh
    squared = normalized_l ** 2 + normalized_c ** 2 + normalized_h ** 2 + rotation * normalized_c * normalized_h
    result = np.sqrt(np.maximum(squared, 0))
    if not np.isfinite(result).all():
        raise ValueError("Delta-E calculation produced nonfinite results")
    return result


def delta_e00_reference(rgb, canonical_lab_d50, *, input_encoding="srgb", evaluation_white="D65", adaptation="Bradford", rgb_domain="unit"):
    """Compare measured RGB and canonical truth at one explicitly shared white."""
    evaluation_white = white_xy(evaluation_white)
    measured_xyz = rgb_to_xyz(rgb, encoding=input_encoding, domain=rgb_domain)
    measured = xyz_to_lab(adapt_xyz(measured_xyz, D65, evaluation_white, transform=adaptation), white=evaluation_white)
    reference_xyz = lab_to_xyz(canonical_lab_d50, white=D50)
    reference = xyz_to_lab(adapt_xyz(reference_xyz, D50, evaluation_white, transform=adaptation), white=evaluation_white)
    return delta_e00(measured, reference)


@dataclass(frozen=True)
class ColorSemantics:
    """An explicit, JSON-serializable interpretation; no estimator changes implied.

    The current correction stages fit encoded sRGB. Linear-light input/output
    denotes conversion at the boundary, not a silent new regression algorithm.
    Pipeline code must reject output policies its stages cannot implement.
    """
    version: str = "legacy-v1"
    input_encoding: str = "srgb"
    output_encoding: str = "srgb"
    primaries: str = "srgb"
    working_white: Tuple[float, float] = D65
    reference_white: Tuple[float, float] = D50
    evaluation_white: Tuple[float, float] = D65
    adaptation: Optional[str] = None
    fitting_domain: str = "encoded-rgb"
    output_policy: str = "clip"

    def __post_init__(self):
        if self.version not in ("legacy-v1", "explicit-v1"):
            raise ValueError("color semantics version must be legacy-v1 or explicit-v1")
        if self.primaries != "srgb" or self.fitting_domain != "encoded-rgb":
            raise ValueError("only sRGB primaries and encoded-rgb fitting are implemented")
        for field in ("input_encoding", "output_encoding"):
            if getattr(self, field) not in ("srgb", "linear-srgb"):
                raise ValueError("%s must be srgb or linear-srgb" % field)
        for field in ("working_white", "reference_white", "evaluation_white"):
            object.__setattr__(self, field, white_xy(getattr(self, field)))
        if self.working_white != D65 or self.reference_white != D50:
            raise ValueError("current explicit color contract requires working D65 and canonical reference D50")
        if self.evaluation_white not in (D50, D65):
            raise ValueError("evaluation white must be D50 or D65")
        transform = self.adaptation if self.adaptation is not None else ("CAT02" if self.version == "legacy-v1" else "Bradford")
        if transform not in _CATS:
            raise ValueError("adaptation must be Bradford or CAT02")
        object.__setattr__(self, "adaptation", transform)
        if self.output_policy not in ("clip", "preserve", "raise"):
            raise ValueError("output policy must be clip, preserve, or raise")
        if self.version == "legacy-v1" and (self.input_encoding != "srgb" or self.output_encoding != "srgb" or self.adaptation != "CAT02" or self.output_policy != "clip"):
            raise ValueError("legacy-v1 retains encoded sRGB, CAT02, and clipping; use explicit-v1 for new semantics")

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("color semantics must be a ColorSemantics instance or dictionary")

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "input_encoding": self.input_encoding,
                "output_encoding": self.output_encoding, "primaries": self.primaries,
                "working_white": list(self.working_white), "reference_white": list(self.reference_white),
                "evaluation_white": list(self.evaluation_white), "adaptation": self.adaptation,
                "fitting_domain": self.fitting_domain, "output_policy": self.output_policy}

    def reference_to_working_rgb(self, canonical_lab_d50, *, clip=False):
        return reference_to_working_rgb(canonical_lab_d50, adaptation=self.adaptation, clip=clip)

    def reference_to_working_lab(self, canonical_lab_d50):
        return reference_to_working_lab(canonical_lab_d50, adaptation=self.adaptation)

    def delta_e00(self, rgb, canonical_lab_d50, *, input_encoding=None):
        return delta_e00_reference(rgb, canonical_lab_d50, input_encoding=input_encoding or self.input_encoding,
                                  evaluation_white=self.evaluation_white, adaptation=self.adaptation)
