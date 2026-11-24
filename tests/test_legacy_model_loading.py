"""Models saved by 2.x and earlier must keep loading and applying unchanged.

The fixtures in ``tests/data/legacy_models`` were fitted and saved by 2.3.0 with every
fitting recipe that 3.0 removes (see ``generate_fixtures.py`` there), together with
their predictions on a fixed scene. This module only uses the public load APIs, so it
passes unchanged on 2.3.0 and on later releases that keep load/apply compatibility.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, MyModels, __version_info__, correct

DATA = Path(__file__).resolve().parent / "data" / "legacy_models"
SCENE = str(DATA / "scene.png")
MANIFEST = json.loads((DATA / "expected.json").read_text())["fixtures"]
# Arrays are stored as float32 subsamples; the float64 statistics cover every pixel.
ARRAY_ATOL = 1e-6
STATS_RTOL, STATS_ATOL = 1e-7, 1e-9

PIPELINE = sorted(key for key in MANIFEST if not key.startswith("key_functions_"))
KEY_FUNCTIONS = sorted(key for key in MANIFEST if key.startswith("key_functions_"))
BUNDLES = sorted(key for key in PIPELINE if MANIFEST[key]["format"] == "bundle")


def _load(record):
    path = DATA / record["file"]
    pipeline = ColorCorrection()
    if record["format"] == "pickle":
        pipeline.models.load(str(DATA), path.stem)
    else:
        pipeline.load_bundle(path)
    return pipeline


def _predict(pipeline):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.predict_image(SCENE)


def _assert_matches(record, images, stages=None):
    with np.load(DATA / record["expected"]) as stored:
        for stage in stages or record["stages"]:
            expected = record["outputs"][stage]
            value = np.asarray(images[stage])
            assert str(value.dtype) == expected["dtype"], stage
            assert list(value.shape) == expected["shape"], stage
            pixels = value.reshape(-1, 3)
            np.testing.assert_allclose(
                pixels.mean(0), expected["mean"], rtol=STATS_RTOL, atol=STATS_ATOL
            )
            np.testing.assert_allclose(
                pixels.std(0), expected["std"], rtol=STATS_RTOL, atol=STATS_ATOL
            )
            sample = stored[stage]
            step = value.shape[0] // sample.shape[0]
            np.testing.assert_allclose(
                value[::step, ::step].astype(np.float32), sample, rtol=0, atol=ARRAY_ATOL
            )


@pytest.mark.parametrize("key", PIPELINE)
def test_saved_model_predictions_are_unchanged(key):
    record = MANIFEST[key]
    images = _predict(_load(record))
    assert sorted(images) == record["stages"]
    _assert_matches(record, images)


@pytest.mark.parametrize("key", BUNDLES)
def test_saved_bundles_apply_through_the_two_call_api(key):
    record = MANIFEST[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = correct(SCENE, str(DATA / record["file"]))
    _assert_matches(record, {"CC": final}, stages=["CC"])


@pytest.mark.parametrize("key", BUNDLES)
def test_saved_bundles_load_into_a_model_container(key):
    record = MANIFEST[key]
    models = MyModels().load_bundle(DATA / record["file"])
    assert models.has_models()
    assert models.bundle_interpretation == (
        "recorded" if models.metadata else "legacy-v1; capture/reference provenance unknown"
    )


def test_metadata_free_models_are_interpreted_as_legacy_v1():
    pipeline = ColorCorrection().load_bundle(DATA / "metadata_free.ccpb")
    assert pipeline.models.metadata == {}
    assert pipeline.fitting_policy.version == "legacy-v1"
    assert pipeline.stage_policy.version == "legacy-v1"
    assert pipeline.color_semantics.version == "legacy-v1"


@pytest.mark.parametrize("key", KEY_FUNCTIONS)
def test_key_functions_pickles_load_with_their_estimators(key):
    """<=1.2 pickles name ``ColorCorrectionPipeline.key_functions`` classes."""
    from sklearn.preprocessing import PolynomialFeatures

    record = MANIFEST[key]
    assert b"ColorCorrectionPipeline.key_functions" in (DATA / record["file"]).read_bytes()
    models = MyModels()
    models.load(str(DATA), Path(record["file"]).stem)
    model, _, method = models.model_cc
    assert method == "ours"
    rgb = _predict(ColorCorrection())["CC"]  # identity: the decoded scene
    features = PolynomialFeatures(degree=model.degree).fit_transform(rgb.reshape(-1, 3))
    output = np.clip(model.model.predict(features), 0, 1).reshape(rgb.shape)
    _assert_matches(record, {"CC": output})


@pytest.mark.xfail(
    __version_info__ < (3,),
    strict=True,
    reason="2.3.0 unpickled <=1.2 models but predict_image failed on missing LUT attributes",
)
@pytest.mark.parametrize("key", KEY_FUNCTIONS)
def test_key_functions_pickles_apply_through_the_pipeline(key):
    record = MANIFEST[key]
    _assert_matches(record, _predict(_load(record)))
