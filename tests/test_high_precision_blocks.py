"""Rewritten colour helpers must be bit-identical to their historical forms."""

import numpy as np

from ColorCorrectionPipeline import colorimetry, ui_native


def _probe_values():
    rng = np.random.default_rng(11)
    edges = []
    for threshold in (0.04045, 0.0031308):
        for value in (threshold, -threshold):
            edges += [value, np.nextafter(value, 1.0), np.nextafter(value, -1.0)]
    edges += [0.0, -0.0, 1.0, np.nextafter(1.0, 0.0), 5e-324, 1e-300]
    return np.concatenate(
        (np.array(edges), rng.random(200_003), rng.random(50_000) * 0.05, rng.normal(size=20_000))
    )


def _masked_decode(values):
    # The historical masked-assignment form the np.where rewrite replaced.
    result = values / 12.92
    mask = values > 0.04045
    result[mask] = ((values[mask] + 0.055) / 1.055) ** 2.4
    return result


def _masked_encode(values):
    values = np.clip(values, 0, 1)
    result = values * 12.92
    mask = np.abs(values) > 0.0031308
    result[mask] = np.sign(values[mask]) * (1.055 * np.abs(values[mask]) ** (1.0 / 2.4) - 0.055)
    return result


def _masked_srgb_to_linear(rgb):
    absolute = np.abs(rgb)
    result = rgb / 12.92
    nonlinear = absolute > 0.04045
    result[nonlinear] = np.sign(rgb[nonlinear]) * ((absolute[nonlinear] + 0.055) / 1.055) ** 2.4
    return result


def _masked_linear_to_srgb(linear):
    absolute = np.abs(linear)
    result = linear * 12.92
    nonlinear = absolute > 0.0031308
    result[nonlinear] = np.sign(linear[nonlinear]) * (
        1.055 * absolute[nonlinear] ** (1.0 / 2.4) - 0.055
    )
    return result


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def test_transfer_rewrites_are_bit_identical_to_masked_forms():
    values = _probe_values()
    unit = np.clip(values, 0, 1)
    _same_bits(ui_native._decode(unit), _masked_decode(unit))
    _same_bits(ui_native._encode(values), _masked_encode(values))
    _same_bits(
        colorimetry.srgb_to_linear(values, domain="extended"), _masked_srgb_to_linear(values)
    )
    _same_bits(colorimetry.linear_to_srgb(values), _masked_linear_to_srgb(values))
    # The returned arrays are owned and do not alias their inputs.
    assert not np.shares_memory(colorimetry.srgb_to_linear(unit), unit)


def test_matrix_and_lab_helpers_match_stacked_forms():
    values = np.random.default_rng(5).random((257, 3))
    stacked = np.stack(
        [
            values[..., 0] * row[0] + values[..., 1] * row[1] + values[..., 2] * row[2]
            for row in ui_native.RGB_TO_XYZ
        ],
        axis=-1,
    )
    _same_bits(ui_native._multiply(ui_native.RGB_TO_XYZ, values), stacked)
    scaled = values / ui_native._white(ui_native.D65)
    f = np.where(
        scaled > ui_native.EPSILON,
        np.cbrt(scaled),
        (ui_native.KAPPA * scaled + 16.0) / 116.0,
    )
    lab = np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )
    _same_bits(ui_native._xyz_to_lab(values, ui_native.D65), lab)
    fy = (lab[..., 0] + 16.0) / 116.0
    g = np.stack([lab[..., 1] / 500.0 + fy, fy, fy - lab[..., 2] / 200.0], axis=-1)
    expected = np.where(g > 6.0 / 29.0, g * g * g, (116.0 * g - 16.0) / ui_native.KAPPA)
    _same_bits(
        ui_native._lab_to_xyz(lab, ui_native.D65), expected * ui_native._white(ui_native.D65)
    )
