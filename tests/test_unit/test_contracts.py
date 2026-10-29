"""Interpretation, ownership, assignment, and serialization contract tests."""

import copy
import json
import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from ColorCorrectionPipeline.contracts import ImageData, ReferenceProfile, ChartObservation, ObservationSet


def profile(**changes):
    options = dict(profile_id="fixture", revision=1, names=tuple("Color %d" % n for n in range(6)),
                   canonical_lab_d50=np.array([[20 + 10 * n, n - 2, 2 - n] for n in range(6)], dtype=float),
                   neutral_ids=("r2c1", "r2c2", "r2c3"), layout=(2, 3), template_id="custom_2x3")
    options.update(changes)
    return ReferenceProfile(**options)


def observation(**changes):
    reference = profile()
    quads = np.array([[[10 + c * 30, 10 + r * 30], [30 + c * 30, 10 + r * 30],
                       [30 + c * 30, 30 + r * 30], [10 + c * 30, 30 + r * 30]]
                      for r in range(2) for c in range(3)], dtype=float)
    options = dict(chart_id="chart-1", face_id="face-1", reference_digest=reference.digest,
                   cell_ids=reference.cell_ids, quadrilaterals=quads, measured_rgb=np.full((6, 3), 0.5),
                   clipped_fractions=np.zeros(6), confidence=0.9, image_shape=(100, 150),
                   layout=reference.layout, template_id=reference.template_id)
    options.update(changes)
    return ChartObservation(**options)


def assert_readonly(array):
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)
    with pytest.raises(ValueError):
        array.flat[0] = 0


def test_uint8_bgr_input_has_owned_normalized_rgb_and_source_metadata():
    pixels = np.array([[[0, 127, 255], [10, 20, 30]]], dtype=np.uint8)
    image = ImageData.from_array(pixels, channel_order="BGR", provenance={"file": "fixture"})
    np.testing.assert_allclose(image.pixels[0, 0], [1, 127 / 255, 0])
    assert image.channel_order == "RGB"
    assert image.source_channel_order == "BGR"
    assert image.source_dtype == "uint8"
    assert image.source_range == (0, 255)
    pixels[:] = 0
    assert image.pixels[0, 0, 0] == 1
    assert_readonly(image.pixels)


def test_uint16_preserves_sub_uint8_precision():
    pixels = np.array([[[32768, 32769, 32770]]], dtype=np.uint16)
    image = ImageData.from_array(pixels)
    np.testing.assert_array_equal(image.pixels, pixels.astype(float) / 65535)
    assert image.pixels[0, 0, 0] != image.pixels[0, 0, 1]
    assert image.source_dtype == "uint16"
    assert image.to_bgr8().dtype == np.uint8
    assert image.to_bgr8().shape == pixels.shape


def test_big_endian_uint16_is_normalized_by_precision():
    pixels = np.array([[[1, 256, 65535]]], dtype=">u2")
    np.testing.assert_array_equal(ImageData.from_array(pixels).pixels, pixels.astype(float) / 65535)


def test_float_range_is_explicit_and_never_inferred_from_maximum():
    pixels = np.array([[[0., 127.5, 255.]]])
    with pytest.raises(ValueError, match="declared"):
        ImageData.from_array(pixels)
    image = ImageData.from_array(pixels, value_range=(0, 255))
    np.testing.assert_array_equal(image.pixels, [[[0, 0.5, 1]]])
    assert image.source_range == (0, 255)


@pytest.mark.parametrize("orientation,expected", [
    (1, [[0, 1, 2], [3, 4, 5]]), (2, [[2, 1, 0], [5, 4, 3]]),
    (3, [[5, 4, 3], [2, 1, 0]]), (4, [[3, 4, 5], [0, 1, 2]]),
    (5, [[0, 3], [1, 4], [2, 5]]), (6, [[3, 0], [4, 1], [5, 2]]),
    (7, [[5, 2], [4, 1], [3, 0]]), (8, [[2, 5], [1, 4], [0, 3]]),
])
def test_all_exif_orientations_have_explicit_provenance(orientation, expected):
    pixels = np.repeat(np.arange(6, dtype=np.uint8).reshape(2, 3, 1), 3, axis=2)
    image = ImageData.from_array(pixels, orientation=orientation)
    np.testing.assert_array_equal(image.pixels[:, :, 0] * 255, expected)
    assert image.source_orientation == orientation
    assert image.source_shape == (2, 3, 3)
    assert image.orientation_applied is True
    assert ImageData.from_dict(image.to_dict()).digest == image.digest


def test_unapplied_orientation_is_recorded_without_rotating_pixels():
    pixels = np.zeros((2, 3, 3))
    image = ImageData.from_array(pixels, orientation=6, apply_orientation=False)
    assert image.shape == (2, 3, 3)
    assert image.orientation_applied is False
    assert image.source_orientation == 6
    normalized = image.with_orientation_applied()
    assert normalized.shape == (3, 2, 3)
    assert normalized.source_shape == (2, 3, 3)
    assert normalized.source_orientation == 6
    assert normalized.orientation_applied is True
    assert normalized.with_orientation_applied() is normalized
    assert image.orientation_applied is False
    assert_readonly(normalized.pixels)


def test_immutable_metadata_and_independent_legacy_adapter():
    provenance = {"reader": {"tags": ["original"]}}
    image = ImageData.from_array(np.full((2, 3, 3), 0.5), provenance=provenance)
    provenance["reader"]["tags"].append("changed")
    assert image.provenance["reader"]["tags"] == ("original",)
    with pytest.raises(TypeError):
        image.provenance["reader"]["other"] = True
    adapter = image.to_srgb()
    adapter[:] = 0
    assert image.pixels.mean() == 0.5
    metadata = image.to_metadata()
    assert "pixels" not in metadata
    metadata["provenance"]["reader"]["tags"].append("changed")
    assert image.provenance["reader"]["tags"] == ("original",)
    assert copy.deepcopy(image) is image
    with pytest.raises(FrozenInstanceError):
        image.color_space = "linear-srgb"


def test_image_identity_distinguishes_encoding_and_provenance():
    integer = ImageData.from_array(np.zeros((2, 3, 3), dtype=np.uint8))
    floating = ImageData.from_array(np.zeros((2, 3, 3)))
    linear = ImageData.from_array(np.zeros((2, 3, 3)), color_space="linear-srgb")
    assert integer.pixel_digest == floating.pixel_digest
    assert integer.digest != floating.digest
    assert floating.pixel_digest != linear.pixel_digest


def test_linear_input_adapter_encodes_without_relabeling():
    image = ImageData.from_array(np.full((1, 1, 3), 0.5), color_space="linear-srgb")
    expected = 1.055 * 0.5 ** (1 / 2.4) - 0.055
    np.testing.assert_allclose(image.to_srgb(), expected, atol=1e-15)
    assert image.pixels.mean() == 0.5


@pytest.mark.parametrize("pixels,kwargs", [
    (np.zeros((2, 3)), {}), (np.zeros((2, 3, 4)), {}),
    (np.zeros((0, 3, 3)), {}), (np.full((2, 3, 3), np.nan), {}),
    (np.full((2, 3, 3), np.inf), {}), (np.full((2, 3, 3), -0.1), {}),
    (np.zeros((2, 3, 3), dtype=bool), {}), (np.zeros((2, 3, 3), dtype=np.int16), {}),
    (np.zeros((2, 3, 3)), {"channel_order": "RGBA"}),
    (np.zeros((2, 3, 3)), {"color_space": "unknown"}),
    (np.zeros((2, 3, 3)), {"orientation": True}),
    (np.zeros((2, 3, 3)), {"orientation": 9}),
    (np.zeros((2, 3, 3)), {"value_range": (1, 0)}),
    (np.zeros((2, 3, 3)), {"provenance": []}),
    (np.zeros((2, 3, 3)), {"provenance": {1: "not-string"}}),
])
def test_image_rejects_ambiguous_or_invalid_interpretations(pixels, kwargs):
    with pytest.raises((TypeError, ValueError)):
        ImageData.from_array(pixels, **kwargs)


@pytest.mark.parametrize("version", [2, True, 1.0])
def test_unknown_or_noninteger_contract_version_is_rejected(version):
    with pytest.raises(ValueError, match="version"):
        ImageData(np.zeros((2, 3, 3)), version=version)


def test_image_serialization_restores_immutable_owned_arrays_and_checks_identity():
    image = ImageData.from_array(np.arange(18, dtype=np.uint16).reshape(2, 3, 3))
    restored = ImageData.from_dict(json.loads(json.dumps(image.to_dict())))
    assert restored.digest == image.digest
    assert_readonly(restored.pixels)
    assert_readonly(pickle.loads(pickle.dumps(image)).pixels)
    changed = image.to_dict()
    changed["pixels"][0][0][0] += 0.1
    with pytest.raises(ValueError, match="digest"):
        ImageData.from_dict(changed)
    changed = image.to_dict()
    changed["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        ImageData.from_dict(changed)
    with pytest.raises(ValueError, match="missing"):
        ImageData.from_dict(image.to_metadata())


def test_reference_owns_unclipped_truth_and_stable_ids_separate_from_labels():
    values = np.array([[50., 150., -150.]] * 6)
    reference = profile(canonical_lab_d50=values, names=("same human label",) * 6)
    values[:] = 0
    np.testing.assert_array_equal(reference.canonical_lab_d50[0], [50, 150, -150])
    assert reference.cell_ids == ("r1c1", "r1c2", "r1c3", "r2c1", "r2c2", "r2c3")
    assert reference.neutral_indices == (3, 4, 5)
    assert_readonly(reference.canonical_lab_d50)
    display = reference.display_rgb()
    assert display.min() >= 0 and display.max() <= 1
    np.testing.assert_array_equal(reference.canonical_lab_d50[0], [50, 150, -150])


def test_default_classic_profile_is_canonical_and_named():
    first, second = ReferenceProfile.classic24(), ReferenceProfile.classic24()
    assert first.layout == (4, 6)
    assert first.template_id == "classic_4x6"
    assert len(first.names) == 24
    assert first.neutral_indices == (18, 19, 20, 21, 22, 23)
    assert first.digest == second.digest
    assert first.names[0].lower() == "dark skin"
    assert np.isfinite(first.canonical_lab_d50).all()


def test_default_catalog_reference_ignores_colour_global_domain_range():
    from colour.utilities import domain_range_scale
    expected = ReferenceProfile.classic24()
    with domain_range_scale("100"):
        scaled_context = ReferenceProfile.classic24()
    assert scaled_context.digest == expected.digest
    np.testing.assert_array_equal(scaled_context.canonical_lab_d50, expected.canonical_lab_d50)


def test_declared_reference_spaces_preserve_source_semantics():
    canonical = profile().canonical_lab_d50
    result = ReferenceProfile.from_values(canonical, color_space="lab", white="D50", scale=1,
                                         profile_id="lab", revision=1, names=profile().names,
                                         neutral_ids=profile().neutral_ids, layout=(2, 3), template_id="custom_2x3")
    np.testing.assert_array_equal(result.canonical_lab_d50, canonical)
    assert result.source_color_space == "lab" and result.source_white == "D50"
    rgb = np.full((6, 3), 127.5)
    result = ReferenceProfile.from_values(rgb, color_space="srgb", white="D65", scale=255,
                                         profile_id="rgb", revision=1, names=profile().names,
                                         neutral_ids=profile().neutral_ids, layout=(2, 3), template_id="custom_2x3")
    assert result.source_scale == 255
    assert result.source_white == "D65"
    assert result.source_color_space == "srgb"
    assert np.isfinite(result.canonical_lab_d50).all()


@pytest.mark.parametrize("changes", [
    {"cell_ids": ("duplicate",) * 6}, {"neutral_ids": ("missing",)},
    {"revision": 0}, {"revision": True}, {"profile_id": ""},
    {"canonical_lab_d50": np.full((6, 3), np.nan)},
    {"canonical_lab_d50": np.zeros((5, 3))}, {"template_id": "classic_4x6"},
    {"source_scale": 100}, {"source_color_space": "srgb", "source_white": "D50"},
    {"adaptation": "guess"}, {"provenance": {"bad": float("nan")}},
])
def test_reference_rejects_invalid_identity_or_semantics(changes):
    with pytest.raises((TypeError, ValueError)):
        profile(**changes)


def test_reference_json_pickle_and_digest_verification():
    reference = profile(provenance={"edition": [1]})
    restored = ReferenceProfile.from_dict(json.loads(json.dumps(reference.to_dict())))
    assert restored.digest == reference.digest
    assert_readonly(restored.canonical_lab_d50)
    assert_readonly(pickle.loads(pickle.dumps(reference)).canonical_lab_d50)
    changed = reference.to_dict()
    changed["revision"] = 2
    with pytest.raises(ValueError, match="digest"):
        ReferenceProfile.from_dict(changed)
    changed = reference.to_dict()
    changed["canonical_white"] = "D65"
    with pytest.raises(ValueError, match="canonical"):
        ReferenceProfile.from_dict(changed)


def test_observation_partial_clipping_cannot_hide_behind_patch_mean():
    reference = profile()
    base = observation()
    samples = np.full((6, 2, 3), 0.5)
    samples[0, 0, 0] = 1
    result = ChartObservation.from_samples(samples=samples, chart_id="chart-1", face_id="face-1",
                                           reference_digest=reference.digest, cell_ids=reference.cell_ids,
                                           quadrilaterals=base.quadrilaterals, confidence=0.9,
                                           image_shape=base.image_shape, layout=base.layout, template_id=base.template_id)
    assert result.measured_rgb[0].max() < 0.99
    assert result.clipped_fractions[0] == 0.5
    assert result.cell_statuses[0] == "clipped"
    assert result.sample_count == 2
    assert result.validate_reference(reference) is result
    for array in (result.quadrilaterals, result.measured_rgb, result.clipped_fractions):
        assert_readonly(array)


@pytest.mark.parametrize("kind", ["bowtie", "flat", "outside", "duplicate", "nonfinite"])
def test_observation_geometry_must_be_valid_ordered_distinct_regions(kind):
    quads = observation().quadrilaterals.copy()
    if kind == "bowtie":
        quads[0] = quads[0, [0, 2, 1, 3]]
    elif kind == "flat":
        quads[0, :, 1] = 10
    elif kind == "outside":
        quads[0, 0] = [-1, -1]
    elif kind == "duplicate":
        quads[1] = quads[0]
    else:
        quads[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        observation(quadrilaterals=quads)


@pytest.mark.parametrize("changes", [
    {"reference_digest": "not-a-digest"}, {"cell_ids": ("same",) * 6},
    {"clipped_fractions": np.full(6, 1.1)}, {"confidence": float("nan")},
    {"confidence": 1.01}, {"measured_rgb": np.full((6, 3), -0.1)},
    {"cell_statuses": ("unknown",) * 6}, {"image_shape": (100, 150, 4)},
    {"clipping_basis": "patch-means"}, {"sample_count": 0},
])
def test_observation_rejects_false_or_invalid_measurement_metadata(changes):
    with pytest.raises((TypeError, ValueError)):
        observation(**changes)


def test_assignment_checks_reference_digest_template_and_exact_cell_order():
    result = observation()
    with pytest.raises(ValueError, match="assignment"):
        result.validate_reference(profile(revision=2))
    with pytest.raises(ValueError, match="assignment"):
        observation(cell_ids=result.cell_ids[::-1]).validate_reference(profile())
    different_template = profile(template_id="another_2x3")
    with pytest.raises(ValueError, match="assignment"):
        result.validate_reference(different_template)


@pytest.mark.parametrize("template_id,layout", [("classic_4x6", (4, 6)), ("spyder_4x6", (4, 6)), ("digital_sg_10x14", (10, 14))])
def test_known_templates_reject_reordered_reference_and_observation_ids(template_id, layout):
    count = layout[0] * layout[1]
    cells = tuple("r%dc%d" % (r + 1, c + 1) for r in range(layout[0]) for c in range(layout[1]))
    reference = ReferenceProfile(profile_id="known", revision=1, names=tuple("Cell %d" % n for n in range(count)),
                                 canonical_lab_d50=np.tile([50., 0, 0], (count, 1)), neutral_ids=(),
                                 layout=layout, template_id=template_id)
    assert reference.cell_ids == cells
    with pytest.raises(ValueError, match="canonical rNcM row-major"):
        replace(reference, cell_ids=cells[::-1])
    quads = np.array([[[2 + c * 5, 2 + r * 5], [5 + c * 5, 2 + r * 5],
                       [5 + c * 5, 5 + r * 5], [2 + c * 5, 5 + r * 5]]
                      for r in range(layout[0]) for c in range(layout[1])], dtype=float)
    valid = ChartObservation(chart_id="known", face_id="face", reference_digest=reference.digest,
                             cell_ids=cells, quadrilaterals=quads, measured_rgb=np.full((count, 3), 0.5),
                             clipped_fractions=np.zeros(count), confidence=0.9,
                             image_shape=(100, 100), layout=layout, template_id=template_id)
    assert valid.validate_reference(reference) is valid
    with pytest.raises(ValueError, match="canonical rNcM row-major"):
        replace(valid, cell_ids=cells[::-1])


def test_generic_custom_templates_can_use_explicit_non_grid_cell_ids():
    custom = profile(cell_ids=("a", "b", "c", "d", "e", "f"), neutral_ids=("d", "e", "f"))
    custom_observation = observation(reference_digest=custom.digest, cell_ids=custom.cell_ids)
    assert custom_observation.validate_reference(custom) is custom_observation


def test_observation_roundtrip_keeps_arrays_immutable_and_detects_tampering():
    result = observation()
    restored = ChartObservation.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.digest == result.digest
    assert_readonly(restored.quadrilaterals)
    assert_readonly(pickle.loads(pickle.dumps(result)).measured_rgb)
    changed = result.to_dict()
    changed["measured_rgb"][0][0] += 0.1
    with pytest.raises(ValueError, match="digest"):
        ChartObservation.from_dict(changed)


def test_multiple_faces_have_explicit_selection_without_cell_relabeling():
    first = observation(chart_id="chart-1", face_id="face")
    second = observation(chart_id="chart-2", face_id="face", confidence=0.8,
                         quadrilaterals=first.quadrilaterals * 1.2)
    result = ObservationSet((first, second), "a" * 64)
    assert result.select_face() is first
    assert result.select_face("largest") is second
    assert result.select_face(chart_id="chart-2", face_id="face") is second
    with pytest.raises(ValueError, match="ambiguous"):
        result.select_face(face_id="face")
    assert result.validate_reference(profile()) is result
    restored = ObservationSet.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.digest == result.digest
    assert_readonly(restored.observations[0].measured_rgb)
    assert_readonly(pickle.loads(pickle.dumps(result)).observations[0].quadrilaterals)


def test_empty_observations_require_an_honest_unavailable_reason():
    result = ObservationSet((), "a" * 64, unavailable_reason="legacy MCC does not expose audited quads")
    assert result.observations == ()
    assert ObservationSet.from_dict(result.to_dict()).digest == result.digest
    with pytest.raises(ValueError):
        ObservationSet((), "a" * 64)
    with pytest.raises(ValueError):
        result.select_face()


def test_observation_set_rejects_mixed_or_duplicate_stage_identities():
    first = observation()
    with pytest.raises(ValueError, match="unique"):
        ObservationSet((first, first), "a" * 64)
    with pytest.raises(ValueError, match="one stage"):
        ObservationSet((first, replace(first, chart_id="other", stage="GC")), "a" * 64)
    with pytest.raises(ValueError):
        ObservationSet((first,), "not-a-digest")
    changed = ObservationSet((first,), "a" * 64).to_dict()
    changed["source_digest"] = "b" * 64
    with pytest.raises(ValueError, match="digest"):
        ObservationSet.from_dict(changed)
