"""Blocked/threaded high-precision prediction must equal the full-frame equations."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, _cpu_workers, colorimetry, ui_native
from ColorCorrectionPipeline.prediction import chunked_cc

IMAGES = Path(__file__).resolve().parents[1] / "Images"
CHART, WHITE, SCENE = (str(IMAGES / n) for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))


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


def _full_frame_reference(source, models, selected, chunk_pixels=None):
    """The former single-threaded full-frame high-precision stage loop."""
    order = ("FFC", "GC", "WB", "CC")
    stop = max(order.index(k) for k in selected)
    gain, curve = models.model_ffc, models.model_gc
    domain = ui_native.ffc_domain(models)
    current, result = source, {}
    for name in order[: stop + 1]:
        if name == "FFC" and gain is not None:
            if "FFC" in selected or curve is None or stop == 0:
                current = ui_native.apply_color(source, gain, rounded=False, domain=domain)
        elif name == "GC" and curve is not None:
            current = ui_native.apply_color(source, gain, curve, rounded=False, domain=domain)
        elif name == "WB" and models.model_wb is not None:
            current = np.clip(current @ models.model_wb, 0, 1)
        elif name == "CC" and models.model_cc is not None:
            current = np.clip(chunked_cc(current, models.model_cc, chunk_pixels or 65536), 0, 1)
        if name in selected:
            result[name] = current.copy()
    return result


@pytest.fixture(scope="module")
def high_fit():
    config = Config(
        chart={"device": "cpu"}, input={"frame_index": 0}, numerics={"precision": "high"}
    )
    pipe = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, failed = pipe.run(CHART, WHITE, name_="chart", config=config)
    assert not failed, pipe.last_run["errors"]
    assert ui_native.high_precision(pipe.models)
    data, source = pipe._prepare_input(SCENE, pipe.color_semantics, {"frame_index": 0})
    return pipe, np.asarray(source, dtype=np.float64)


@pytest.mark.slow
@pytest.mark.parametrize(
    "selected, chunk_pixels",
    [
        (("CC",), None),
        (("FFC", "GC", "WB", "CC"), None),
        (("FFC", "WB"), None),
        (("GC",), None),
        (("FFC", "CC"), 10_000),
        (("CC",), 100_003),
    ],
)
@pytest.mark.parametrize("workers", [1, 8])
def test_blocked_high_precision_matches_full_frame(
    high_fit, selected, chunk_pixels, workers, monkeypatch
):
    pipe, source = high_fit
    monkeypatch.setattr(_cpu_workers, "worker_count", lambda size, minimum: workers)
    expected = _full_frame_reference(source, pipe.models, selected, chunk_pixels)
    actual = ui_native.predict(source, pipe.models, selected, chunk_pixels)
    assert list(actual) == list(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
        assert actual[name].dtype == np.float64 and actual[name].shape == source.shape


@pytest.mark.slow
def test_blocked_high_precision_reports_the_first_invalid_stage(high_fit):
    pipe, source = high_fit
    models = copy.deepcopy(pipe.models)
    models.model_wb = np.diag([np.nan, 1.0, 1.0])
    with pytest.raises(FloatingPointError, match="native UI WB produced invalid"):
        ui_native.predict(source, models, ("FFC", "GC", "WB", "CC"))
