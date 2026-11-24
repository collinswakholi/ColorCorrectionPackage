"""Generate the saved-model fixtures used by ``tests/test_legacy_model_loading.py``.

The fixtures were written by ColorCorrectionPipeline 2.3.0 (review/2026-11-deep-refine
@ b8baaad), before 3.0 removed the legacy fitters. This script therefore only runs
against a 2.3.0 checkout; it is kept as the record of how each fixture was made:

    PYTHONPATH=<2.3.0 checkout> python tests/data/legacy_models/generate_fixtures.py

Inputs are the repository sample photographs downscaled 8x (INTER_AREA). Each
recipe is fitted once, saved as a trusted pickle and as a portable bundle where the
2.3.0 exporter supports it, reloaded through the public load APIs, and its
predictions on ``scene.png`` are recorded in ``expected.json`` / ``<name>.npz``.
"""

import copy
import json
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCALE = 8
CHART = {"device": "cpu"}
STAGES = ("FFC", "GC", "WB", "CC")


def _images():
    names = {"chart": "Image_1.JPG", "white": "white.JPG", "scene": "Sample_1.JPG"}
    for key, name in names.items():
        image = cv2.imread(str(ROOT / "Images" / name))
        small = cv2.resize(
            image,
            (image.shape[1] // SCALE, image.shape[0] // SCALE),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(HERE / f"{key}.png"), small)
    return [str(HERE / f"{key}.png") for key in ("chart", "white", "scene")]


def _legacy(cc_kwargs, *, stages=False):
    from ColorCorrectionPipeline import Config

    switches = {} if stages else {"do_ffc": False, "do_gc": False, "do_wb": False}
    switches["check_saturation"] = stages
    return Config(
        fitting_policy="legacy-v1", chart_detection=CHART, CC_kwargs=cc_kwargs, **switches
    )


def _recipes():
    from ColorCorrectionPipeline import Config

    explicit = {"version": "explicit-v1"}
    modern = {"version": "modern-v1"}
    no_lut = {"use_lut": False}
    return {
        # fitting_policy="legacy-v1": conventional Finlayson CCM.
        "legacy_v1_conv": lambda: _legacy(
            {"cc_method": "conv", "method": "Finlayson 2015", "degree": 2}
        ),
        # fitting_policy="legacy-v1": the "ours" custom CC with each estimator.
        "legacy_v1_ours_linear": lambda: _legacy(
            {"cc_method": "ours", "mtd": "linear", "degree": 2, **no_lut}
        ),
        "legacy_v1_ours_pls": lambda: _legacy(
            {"cc_method": "ours", "mtd": "pls", "degree": 2, "ncomp": 6, **no_lut}
        ),
        "legacy_v1_ours_nn": lambda: _legacy(
            {
                "cc_method": "ours",
                "mtd": "nn",
                "degree": 2,
                "hidden_layers": [16],
                "max_iterations": 200,
                **no_lut,
            }
        ),
        "legacy_v1_ours_custom": lambda: _legacy(
            {
                "cc_method": "ours",
                "mtd": "custom",
                "degree": 2,
                "hidden_layers": [16],
                "max_iterations": 60,
                "random_state": 3,
                **no_lut,
            }
        ),
        # fitting_policy="legacy-v1" with FFC + saturation + GC + WB and a stored LUT.
        "legacy_v1_all_stages": lambda: _legacy(
            {
                "cc_method": "ours",
                "mtd": "linear",
                "degree": 2,
                "use_lut": True,
                "lazy_lut": False,
                "lut_grid_size": 9,
                "lut_min_pixels": 64,
            },
            stages=True,
        ),
        # fitting_policy="reproducible-v1" (modern stages, explicit color semantics).
        "reproducible_v1_custom": lambda: Config(
            fitting_policy="reproducible-v1",
            color_semantics=explicit,
            stage_policy=modern,
            do_ffc=False,
            chart_detection=CHART,
            CC_kwargs={
                "cc_method": "ours",
                "mtd": "custom",
                "degree": 2,
                "hidden_layers": [8],
                "max_iterations": 40,
                **no_lut,
            },
        ),
        "reproducible_v1_conv": lambda: Config(
            fitting_policy="reproducible-v1",
            color_semantics=explicit,
            stage_policy=modern,
            do_ffc=False,
            chart_detection=CHART,
            CC_kwargs={"cc_method": "conv", "degree": 2},
        ),
        # fitting_policy="ui-compatible-v1" with the staged-v1 runtime.
        "ui_compatible_v1_staged": lambda: Config(
            fitting_policy="ui-compatible-v1",
            check_saturation=False,
            chart_detection=CHART,
            CC_kwargs={"lut_grid_size": 9},
        ),
        # API 1: an unversioned configuration dictionary (inferred legacy-v1).
        "api_v1_unversioned_dict": lambda: Config.from_dict(
            {
                "do_ffc": False,
                "do_gc": True,
                "do_wb": True,
                "chart_detection": CHART,
                "CC_kwargs": {"mtd": "linear", "degree": 3, "use_lut": False},
            }
        ),
        # Control: the 2.x default fit (API 2, native-v1 runtime).
        "api_v2_default": lambda: Config(chart=CHART, CC_kwargs={"lut_grid_size": 9}),
    }


def _keyfunctions_models(source):
    """Rebuild a 1.x-style container whose CC classes live at key_functions."""
    from ColorCorrectionPipeline import key_functions as kf
    from ColorCorrectionPipeline.models import MyModels

    model, params, method = source.model_cc
    legacy = kf.Regressor_Model()
    for name in vars(legacy):
        if name != "model":
            setattr(legacy, name, copy.deepcopy(getattr(model, name)))
    estimator = model.model
    if type(estimator).__name__ == "CustomNN":
        network = kf.CustomNN(hidden_layers=list(estimator.hidden_layers))
        network.input_dim, network.output_dim = estimator.input_dim, estimator.output_dim
        network.model = estimator.model
        network.device = estimator.device
        estimator = network
    legacy.model = estimator
    models = MyModels()
    models.model_cc = (legacy, copy.deepcopy(params), method)
    del models.metadata  # 1.x containers predate calibration metadata
    return models


def _summary(images):
    record, arrays = {}, {}
    final = [key for key in STAGES if key in images][-1]
    for key in STAGES:
        if key not in images:
            continue
        value = np.asarray(images[key])
        record[key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "mean": value.reshape(-1, 3).mean(0).tolist(),
            "std": value.reshape(-1, 3).std(0).tolist(),
        }
        step = 4 if key == final else 8
        arrays[key] = value[::step, ::step].astype(np.float32)
    return record, arrays


def main():
    from ColorCorrectionPipeline import ColorCorrection, __version__
    from ColorCorrectionPipeline.models import MyModels

    assert __version__ == "2.3.0", "fixtures must be generated by 2.3.0"
    chart, white, scene = _images()
    manifest = {"generator": f"ColorCorrectionPipeline {__version__}", "fixtures": {}}

    def record(name, fmt, filename, predictor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images = predictor.predict_image(scene)
        stats, arrays = _summary(images)
        key = f"{name}__{fmt}"
        expected = f"{key}.npz"
        previous = HERE / f"{name}__pickle.npz"
        if fmt == "bundle" and previous.exists():
            with np.load(previous) as stored:
                if set(stored) == set(arrays) and all(
                    np.array_equal(stored[k], arrays[k]) for k in arrays
                ):
                    expected = previous.name  # identical outputs share one file
        manifest["fixtures"][key] = {
            "file": filename,
            "format": fmt,
            "expected": expected,
            "stages": sorted(images),
            "outputs": stats,
        }
        if expected == f"{key}.npz":
            np.savez_compressed(HERE / expected, **arrays)
        print("recorded", key, sorted(images), expected)

    fitted = {}
    for name, recipe in _recipes().items():
        pipe = ColorCorrection()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, failed = pipe.run(chart, white, name, config=recipe())
        if failed:
            raise RuntimeError(f"{name}: {pipe.last_run['errors']}")
        fitted[name] = pipe.models
        pipe.models.save(str(HERE), name)
        loaded = ColorCorrection()
        loaded.models.load(str(HERE), name)
        record(name, "pickle", f"{name}.pkl", loaded)
        try:
            pipe.models.save_bundle(HERE / f"{name}.ccpb")
        except Exception as exc:  # recorded, not silently skipped
            print("no bundle for", name, exc)
            continue
        bundle = ColorCorrection().load_bundle(HERE / f"{name}.ccpb")
        record(name, "bundle", f"{name}.ccpb", bundle)

    # Metadata-free 2.x container: interpreted as legacy-v1 on load.
    # FFC is left out to keep the fixtures small; legacy_v1_all_stages covers it.
    bare = copy.copy(fitted["legacy_v1_all_stages"])
    bare.metadata = {}
    bare.model_ffc = None
    bare.save(str(HERE), "metadata_free")
    loaded = ColorCorrection()
    loaded.models.load(str(HERE), "metadata_free")
    record("metadata_free", "pickle", "metadata_free.pkl", loaded)
    MyModels.save_bundle(bare, HERE / "metadata_free.ccpb")
    bundle = ColorCorrection().load_bundle(HERE / "metadata_free.ccpb")
    record("metadata_free", "bundle", "metadata_free.ccpb", bundle)

    # <=1.2 pickles reference ColorCorrectionPipeline.key_functions classes.
    for name, source in (
        ("key_functions_linear", "legacy_v1_ours_linear"),
        ("key_functions_customnn", "legacy_v1_ours_custom"),
    ):
        models = _keyfunctions_models(fitted[source])
        with open(HERE / f"{name}.pkl", "wb") as handle:
            pickle.dump(models, handle, protocol=4)
        assert b"ColorCorrectionPipeline.key_functions" in (HERE / f"{name}.pkl").read_bytes()
        # 2.3.0 unpickles these containers but cannot apply them through
        # predict_image (the 1.x Regressor_Model lacks LUT attributes), so the
        # expected output is the 1.x CPU prediction: key_functions.predict_.
        from ColorCorrectionPipeline import key_functions as kf
        from ColorCorrectionPipeline.core.utils import to_float64

        reloaded = MyModels()
        reloaded.load(str(HERE), name)
        rgb = to_float64(cv2.imread(scene)[:, :, ::-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = np.clip(kf.predict_(rgb, reloaded.model_cc[0]), 0, 1)
        stats, arrays = _summary({"CC": output})
        key = f"{name}__pickle"
        manifest["fixtures"][key] = {
            "file": f"{name}.pkl",
            "format": "pickle",
            "expected": f"{key}.npz",
            "stages": ["CC"],
            "outputs": stats,
            "reference": "key_functions.predict_ (2.3.0 predict_image could not apply it)",
        }
        np.savez_compressed(HERE / f"{key}.npz", **arrays)
        print("recorded", key)

    (HERE / "expected.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())
    print(f"total fixture bytes: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
