"""Classic 24, SpyderCHECKR 24 and Digital SG: references, custom values and calibration."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import synthetic_charts as sc

from ColorCorrectionPipeline import Config, ReferenceProfile, calibrate
from ColorCorrectionPipeline.chart_references import (
    CHARTS,
    builtin_reference,
    is_builtin_reference,
)
from ColorCorrectionPipeline.ui_native import default_reference

STUDIO_TEMPLATES = (
    Path(__file__).resolve().parents[2]
    / "ColorCorrectionPackage_UI"
    / "app"
    / "ColorCorrect.Engine"
    / "ReferenceTemplates.json"
)


# -- built-in references ------------------------------------------------------


def test_classic_builtin_is_the_unchanged_default_reference():
    assert builtin_reference("classic").digest == default_reference().digest
    assert builtin_reference("passport").digest == default_reference().digest


@pytest.mark.parametrize(
    "chart, layout, neutrals",
    [
        ("spyder", (4, 6), (18, 19, 20, 21, 22, 23)),
        ("digital-sg", (10, 14), (60, 61, 62, 63, 64, 65)),
    ],
)
def test_builtin_references_use_studios_named_grey_patches(chart, layout, neutrals):
    ref = builtin_reference(chart)
    assert ref.layout == layout and ref.neutral_indices == neutrals
    assert is_builtin_reference(ref)
    restored = ReferenceProfile.from_dict(ref.to_dict())
    assert restored.digest == ref.digest
    # Grey patches really are grey; a coloured patch would bend GC and WB.
    chroma = np.hypot(*ref.canonical_lab_d50[list(neutrals), 1:].T)
    assert chroma.max() < 4


@pytest.mark.skipif(not STUDIO_TEMPLATES.exists(), reason="ColorCorrect Studio checkout not found")
def test_reference_values_match_studios_catalog():
    studio = {t["Chart"]: t for t in json.loads(STUDIO_TEMPLATES.read_text(encoding="utf-8-sig"))}
    for key, template in (("spyder", "spyder_4x6"), ("digital-sg", "digital_sg_10x14")):
        spec = CHARTS[template]
        assert list(spec.names) == studio[key]["Names"]
        np.testing.assert_array_equal(np.array(spec.lab_d50), np.array(studio[key]["Lab"]))


# -- user-supplied references ---------------------------------------------------


def test_custom_values_in_every_layout_and_order_give_the_same_reference():
    lab = builtin_reference("spyder").canonical_lab_d50
    row_major = ReferenceProfile.for_chart("spyder", lab.tolist())
    column_major = ReferenceProfile.for_chart(
        "spyder",
        [lab[r * 6 + c].tolist() for c in range(6) for r in range(4)],
        order="column-major",
    )
    cells = "\n".join(
        ["cell,L,a,b"]
        + [
            "%s%d,%r,%r,%r" % (chr(65 + c), r + 1, *lab[r * 6 + c])
            for r in range(4)
            for c in range(6)
        ][::-1]
    )
    coordinates = "\n".join(
        "%d\t%d\t%r\t%r\t%r" % (r + 1, c + 1, *lab[r * 6 + c]) for r in range(4) for c in range(6)
    )
    for ref in (
        row_major,
        column_major,
        ReferenceProfile.from_text(cells, chart="spyder"),
        ReferenceProfile.from_text(coordinates, chart="spyder"),
    ):
        np.testing.assert_allclose(ref.canonical_lab_d50, lab, atol=1e-9)
        assert (
            ref.template_id == "spyder_4x6"
            and ref.neutral_ids == builtin_reference("spyder").neutral_ids
        )
        assert not is_builtin_reference(ref)


def test_custom_reference_options_and_errors(tmp_path):
    grey = np.full((24, 3), 128).tolist()
    ref = ReferenceProfile.for_chart(
        "classic",
        grey,
        color_space="srgb",
        white="D65",
        scale=255,
        neutral_ids=["white", "black", "neutral 5"],
    )
    assert ref.neutral_ids == ("r4c1", "r4c6", "r4c4")
    path = tmp_path / "my_chart.csv"
    path.write_text("\n".join("%r,%r,%r" % tuple(v) for v in default_reference().canonical_lab_d50))
    from_file = ReferenceProfile.from_csv(path)
    assert from_file.name == "my_chart"
    np.testing.assert_allclose(
        from_file.canonical_lab_d50, default_reference().canonical_lab_d50, atol=1e-9
    )
    lab = builtin_reference("spyder").canonical_lab_d50.tolist()
    with pytest.raises(ValueError, match="needs 24 values"):
        ReferenceProfile.for_chart("spyder", lab[:23])
    with pytest.raises(ValueError, match="white must be"):
        ReferenceProfile.for_chart("spyder", lab, white="D99")
    with pytest.raises(ValueError, match="ambiguous"):
        ReferenceProfile.for_chart("spyder", lab, neutral_ids=["neutral"])
    with pytest.raises(ValueError, match="chart type"):
        ReferenceProfile.for_chart("macbeth", lab)


def test_chart_type_cannot_contradict_a_custom_reference():
    custom = ReferenceProfile.for_chart(
        "spyder", builtin_reference("spyder").canonical_lab_d50.tolist()
    )
    with pytest.raises(ValueError, match="contradicts the custom reference"):
        Config(reference=custom, chart={"type": "classic"}).validate()
    with pytest.raises(ValueError, match="chart type"):
        Config(chart={"type": "macbeth"}).validate()


# -- calibration on synthetic photographs ----------------------------------------


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    folder = tmp_path_factory.mktemp("charts")
    out = {"white": sc.write(folder / "white.png", sc.white_field())}
    for chart in ("classic", "spyder", "digital-sg"):
        _, card = sc.chart_card(chart)
        out[chart] = sc.write(folder / ("%s.png" % chart), sc.camera(sc.scene([card])))
    _, card = sc.chart_card("spyder")
    out["spyder-upside-down"] = sc.write(folder / "spyder180.png", sc.camera(sc.scene([card], 180)))
    out["classic+spyder"] = sc.write(
        folder / "both.png",
        sc.camera(sc.scene([sc.chart_card("classic")[1], sc.chart_card("spyder")[1]])),
    )
    return out


def _fit(photo, white, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrate(photo, white=white, device="cpu", **options)


def _after(cal, stage):
    return next(
        v["delta_e00_after_mean"] for k, v in cal.metrics.items() if k.endswith("_" + stage)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "chart, template",
    [("spyder", "spyder_4x6"), ("digital-sg", "digital_sg_10x14"), ("classic", "classic_4x6")],
)
def test_auto_finds_each_chart_and_every_stage_reduces_error(photos, chart, template):
    cal = _fit(photos[chart], photos["white"])
    assert cal.pipeline.last_run["chart"]["template"] == template
    assert cal.pipeline.reference_profile.template_id == template
    errors = [_after(cal, stage) for stage in ("FFC", "GC", "WB", "CC")]
    assert errors == sorted(errors, reverse=True), errors  # each stage helps
    assert errors[-1] < 1.5, errors
    # GC and WB carry real weight: without them CC alone ends clearly worse.
    cc_only = _fit(
        photos[chart], photos["white"], steps={"gc": {"enabled": False}, "wb": {"enabled": False}}
    )
    assert _after(cc_only, "CC") > 2 * errors[-1]


@pytest.mark.slow
def test_chart_orientation_does_not_change_the_result(photos):
    upright = _after(_fit(photos["spyder"], photos["white"]), "CC")
    flipped = _after(_fit(photos["spyder-upside-down"], photos["white"]), "CC")
    assert abs(upright - flipped) < 0.05


@pytest.mark.slow
def test_auto_prefers_classic_and_chart_type_selects_another(photos):
    both = photos["classic+spyder"]
    assert _fit(both, None).pipeline.last_run["chart"]["template"] == "classic_4x6"
    spyder = _fit(both, None, chart_type="spyder")
    assert spyder.pipeline.last_run["chart"]["template"] == "spyder_4x6"
    assert "classic_4x6" in spyder.pipeline.last_run["detection"][-1]["ignored_layouts"]


@pytest.mark.slow
def test_custom_reference_pins_its_chart_and_matches_the_builtin(photos):
    custom = ReferenceProfile.for_chart(
        "digital-sg", builtin_reference("digital-sg").canonical_lab_d50.tolist()
    )
    pinned = _fit(photos["digital-sg"], photos["white"], reference=custom)
    assert pinned.pipeline.last_run["chart"] == {
        "requested": "auto",
        "template": "digital_sg_10x14",
        "reference": "custom-digital-sg-10x14",
        "builtin_reference": False,
        "validation": "new in 3.1: verified on synthetic charts; not yet on real photographs",
    }
    builtin = _fit(photos["digital-sg"], photos["white"])
    assert abs(_after(pinned, "CC") - _after(builtin, "CC")) < 1e-6
    with pytest.raises(Exception, match="digital_sg_10x14|reference is for"):
        _fit(photos["spyder"], photos["white"], reference=custom)


@pytest.mark.slow
def test_saved_bundle_of_a_spyder_calibration_reloads(photos, tmp_path):
    cal = _fit(photos["spyder"], photos["white"], save_to=tmp_path / "spyder.ccpb")
    from ColorCorrectionPipeline import correct

    np.testing.assert_array_equal(
        correct(photos["spyder"], tmp_path / "spyder.ccpb"), correct(photos["spyder"], cal)
    )
