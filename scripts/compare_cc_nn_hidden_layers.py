"""Compare sklearn NN color-correction hidden-layer widths."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import write_image


OUT_ROOT = Path("results") / "cc_nn_hidden_compare"
SAMPLE_IMAGE = "Images/Sample_1.JPG"
WHITE_IMAGE = "Images/white.JPG"
PREDICT_IMAGE = "Images/Image_1.JPG"
STAGES = ["FFC", "GC", "WB", "CC"]


def run_config(width: int) -> dict:
    run_name = f"nn_{width}"
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ColorCorrection()
    cc_kwargs = {
        "cc_method": "ours",
        "mtd": "nn",
        "degree": 2,
        "n_samples": 50,
        "hidden_layers": [width],
        "max_iterations": 10000,
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

    started = time.perf_counter()
    metrics, images, error = pipeline.run(SAMPLE_IMAGE, WHITE_IMAGE, run_name, config)
    elapsed_seconds = time.perf_counter() - started

    original_bgr = cv2.imread(SAMPLE_IMAGE)
    original_rgb = original_bgr[:, :, ::-1].astype(np.float64) / 255.0
    original_path = out_dir / f"{run_name}_00_original.png"
    write_image(original_path, original_rgb)

    stage_images = {"original": str(original_path.resolve())}
    for index, stage in enumerate(STAGES, 1):
        key = f"{run_name}_{stage}"
        if key in images:
            path = out_dir / f"{run_name}_{index:02d}_{stage.lower()}_sample.png"
            write_image(path, images[key])
            stage_images[stage] = str(path.resolve())

    prediction = pipeline.predict_image(PREDICT_IMAGE, show=False)
    prediction_images = {}
    for index, stage in enumerate(STAGES, 1):
        if stage in prediction:
            path = out_dir / f"image_1_{index:02d}_{stage.lower()}.png"
            write_image(path, prediction[stage])
            prediction_images[stage] = str(path.resolve())

    return {
        "run_name": run_name,
        "run_error": error,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(out_dir.resolve()),
        "cc_kwargs": cc_kwargs,
        "stage_images": stage_images,
        "prediction_images": prediction_images,
        "metrics": metrics,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_config(width) for width in (32, 64)]
    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps({"summary_path": str(summary_path.resolve()), "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
