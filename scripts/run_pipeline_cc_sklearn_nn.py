"""Run the bundled image pipeline with the sklearn NN CC config."""

import json
from pathlib import Path

import cv2
import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import write_image


OUT_DIR = Path("results") / "pipeline_cc_sklearn_nn"
SAMPLE_IMAGE = "Images/Sample_1.JPG"
WHITE_IMAGE = "Images/white.JPG"
PREDICT_IMAGE = "Images/Image_1.JPG"
STAGES = ["FFC", "GC", "WB", "CC"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = ColorCorrection()
    cc_kwargs = {
        "cc_method": "ours",
        "mtd": "nn",
        "degree": 2,
        "n_samples": 50,
        "hidden_layers": [100],
        "max_iterations": 10000,
        "use_batch_norm": True,
        "get_deltaE": True,
        "show": False,
    }
    config = Config(
        do_ffc=True,
        do_gc=True,
        do_wb=True,
        do_cc=True,
        check_saturation=False,
        FFC_kwargs={
            "fit_method": "linear",
            "bins": 50,
            "smooth_window": 7,
            "degree": 5,
            "interactions": True,
            "max_iter": 1000,
            "get_deltaE": True,
            "show": False,
        },
        GC_kwargs={"max_degree": 3, "get_deltaE": True, "show": False},
        WB_kwargs={"get_deltaE": True, "show": False},
        CC_kwargs=cc_kwargs,
    )

    metrics, images, error = pipeline.run(SAMPLE_IMAGE, WHITE_IMAGE, "sklearn_nn", config)

    original_bgr = cv2.imread(SAMPLE_IMAGE)
    original_rgb = original_bgr[:, :, ::-1].astype(np.float64) / 255.0
    original_path = OUT_DIR / "sklearn_nn_00_original.png"
    write_image(original_path, original_rgb)

    stage_images = {"original": str(original_path.resolve())}
    for index, stage in enumerate(STAGES, 1):
        key = f"sklearn_nn_{stage}"
        if key in images:
            path = OUT_DIR / f"sklearn_nn_{index:02d}_{stage.lower()}_sample.png"
            write_image(path, images[key])
            stage_images[stage] = str(path.resolve())

    prediction = pipeline.predict_image(PREDICT_IMAGE, show=False)
    prediction_images = {}
    for index, stage in enumerate(STAGES, 1):
        if stage in prediction:
            path = OUT_DIR / f"image_1_{index:02d}_{stage.lower()}.png"
            write_image(path, prediction[stage])
            prediction_images[stage] = str(path.resolve())

    summary = {
        "run_error": error,
        "output_dir": str(OUT_DIR.resolve()),
        "cc_kwargs": cc_kwargs,
        "stage_images": stage_images,
        "prediction_images": prediction_images,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
