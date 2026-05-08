"""Run FFC configuration comparisons on the bundled sample images."""

import json
from pathlib import Path

import cv2
import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import write_image


BASE = Path("results") / "ffc_config_compare"
SAMPLE_IMAGE = "Images/Sample_1.JPG"
WHITE_IMAGE = "Images/white.JPG"
PREDICT_IMAGE = "Images/Image_1.JPG"
STAGES = ["FFC", "GC", "WB", "CC"]


def run_config(name: str, ffc_kwargs: dict) -> dict:
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)

    pipeline = ColorCorrection()
    config = Config(
        do_ffc=True,
        do_gc=True,
        do_wb=True,
        do_cc=True,
        check_saturation=False,
        FFC_kwargs=ffc_kwargs,
        GC_kwargs={"max_degree": 3, "get_deltaE": True, "show": False},
        WB_kwargs={"get_deltaE": True, "show": False},
        CC_kwargs={
            "cc_method": "ours",
            "mtd": "linear",
            "degree": 1,
            "n_samples": 1,
            "get_deltaE": True,
            "show": False,
        },
    )

    metrics, images, error = pipeline.run(SAMPLE_IMAGE, WHITE_IMAGE, name, config)

    original_bgr = cv2.imread(SAMPLE_IMAGE)
    original_rgb = original_bgr[:, :, ::-1].astype(np.float64) / 255.0
    original_path = out / f"{name}_00_original.png"
    write_image(original_path, original_rgb)

    stage_images = {"original": str(original_path.resolve())}
    for index, stage in enumerate(STAGES, 1):
        key = f"{name}_{stage}"
        if key in images:
            path = out / f"{name}_{index:02d}_{stage.lower()}.png"
            write_image(path, images[key])
            stage_images[stage] = str(path.resolve())

    prediction = pipeline.predict_image(PREDICT_IMAGE)
    prediction_images = {}
    for index, stage in enumerate(STAGES, 1):
        if stage in prediction:
            path = out / f"image_1_{index:02d}_{stage.lower()}.png"
            write_image(path, prediction[stage])
            prediction_images[stage] = str(path.resolve())

    return {
        "run_error": error,
        "output_dir": str(out.resolve()),
        "stage_images": stage_images,
        "prediction_images": prediction_images,
        "metrics": metrics,
    }


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)

    configs = {
        "linear": {
            "fit_method": "linear",
            "bins": 50,
            "smooth_window": 7,
            "degree": 5,
            "interactions": True,
            "max_iter": 1000,
            "get_deltaE": True,
            "show": False,
        },
        "nn": {
            "fit_method": "nn",
            "bins": 50,
            "smooth_window": 7,
            "degree": 5,
            "interactions": True,
            "max_iter": 1000,
            "random_seed": 0,
            "get_deltaE": True,
            "show": False,
        },
    }

    summary = {name: run_config(name, kwargs) for name, kwargs in configs.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
