"""Loading contracts must not bypass the current pipeline's capability gates."""

import copy
import pickle

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, ReferenceProfile
from ColorCorrectionPipeline.models import MyModels


def completed_model():
    pipeline = ColorCorrection()
    config = Config(do_ffc=False, do_gc=False, do_wb=False, do_cc=False, check_saturation=False,
                    color_semantics={"version": "explicit-v1"}, chart_detection={"device": "cpu"})
    assert pipeline.run(np.full((4, 4, 3), 0.2), config=config)[2] is False
    return pipeline.models


@pytest.mark.parametrize("change", [
    "preserve-output", "legacy-evaluation-white", "unsupported-template", "missing-stage", "extra-stage",
    "nonboolean-stage", "active-model-disabled", "enabled-stage-without-model", "invalid-image-options",
    "reference-mismatch", "source-mismatch",
])
def test_invalid_saved_capabilities_preserve_active_models(tmp_path, change):
    invalid = copy.deepcopy(completed_model())
    if change == "preserve-output":
        invalid.metadata["color_semantics"]["output_policy"] = "preserve"
    elif change == "legacy-evaluation-white":
        invalid.metadata["color_semantics"].update(version="legacy-v1", adaptation="CAT02", evaluation_white=[0.3457, 0.3585])
    elif change == "unsupported-template":
        original = ReferenceProfile.from_dict(invalid.metadata["reference_profile"])
        unsupported = ReferenceProfile(original.profile_id, original.revision, original.names,
                                       original.canonical_lab_d50, original.neutral_ids, template_id="spyder_4x6")
        invalid.metadata["reference_profile"] = unsupported.to_dict()
    elif change == "missing-stage":
        invalid.metadata["enabled_stages"].pop("do_gc")
    elif change == "extra-stage":
        invalid.metadata["enabled_stages"]["future_stage"] = False
    elif change == "nonboolean-stage":
        invalid.metadata["enabled_stages"]["do_gc"] = 0
    elif change == "active-model-disabled":
        invalid.model_wb = np.eye(3)
    elif change == "enabled-stage-without-model":
        invalid.metadata["enabled_stages"]["do_wb"] = True
    elif change == "invalid-image-options":
        invalid.metadata["image_options"] = {"icc_policy": "silently-convert"}
    elif change == "reference-mismatch":
        invalid.metadata["reference_profile"]["revision"] += 1
    else:
        invalid.metadata["input_image"]["digest"] = "f" * 64
    destination = tmp_path / "inconsistent.pkl"
    destination.write_bytes(pickle.dumps(invalid))
    active = MyModels()
    active.model_wb = np.eye(3) * 0.5
    old = active.model_wb
    with pytest.raises(OSError):
        active.load(str(tmp_path), "inconsistent")
    assert active.model_wb is old
    assert active.metadata == {}


def test_valid_generated_contract_metadata_roundtrips(tmp_path):
    generated = completed_model()
    generated.save(str(tmp_path), "valid")
    restored = MyModels()
    restored.load(str(tmp_path), "valid")
    assert restored.metadata == generated.metadata


def test_legacy_container_without_contract_metadata_remains_compatible(tmp_path):
    original = MyModels()
    original.model_wb = np.eye(3) * 0.95
    del original.metadata
    (tmp_path / "legacy.pkl").write_bytes(pickle.dumps(original))
    restored = MyModels()
    restored.load(str(tmp_path), "legacy")
    np.testing.assert_array_equal(restored.model_wb, original.model_wb)
    assert restored.metadata == {}


@pytest.mark.parametrize("white", [[0.3457, 0.3585], [float("nan"), .32902], [.3, 0], [.2, .3, .4], "D65"])
def test_invalid_saved_explicit_reference_illuminant_preserves_active_models(tmp_path, white):
    invalid = completed_model()
    invalid.metadata["legacy_reference_illuminant"] = white
    (tmp_path / "white.pkl").write_bytes(pickle.dumps(invalid))
    active = MyModels()
    active.model_wb = np.eye(3)
    previous = active.model_wb
    with pytest.raises(OSError):
        active.load(str(tmp_path), "white")
    assert active.model_wb is previous


def test_valid_legacy_custom_reference_illuminant_remains_loadable(tmp_path):
    model = completed_model()
    model.metadata["color_semantics"].update(version="legacy-v1", adaptation="CAT02")
    model.metadata["legacy_reference_illuminant"] = [0.3457, 0.3585]
    model.save(str(tmp_path), "legacy-white")
    loaded = MyModels()
    loaded.load(str(tmp_path), "legacy-white")
    assert loaded.metadata["legacy_reference_illuminant"] == [0.3457, 0.3585]
