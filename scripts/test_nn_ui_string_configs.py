"""Exercise sklearn NN color correction with UI-style string configs."""

import json
import time

from ColorCorrectionPipeline import ColorCorrection, Config


CONFIGS = [
    (
        "ui_scalar_64",
        {
            "cc_method": "ours",
            "mtd": "nn",
            "degree": "2",
            "n_samples": "10",
            "hidden_layers": "64",
            "max_iterations": "100",
            "random_state": "0",
            "tol": "1e-5",
            "use_lut": "true",
            "lazy_lut": "true",
            "get_deltaE": True,
            "show": False,
        },
    ),
    (
        "json_64",
        {
            "cc_method": "ours",
            "mtd": "nn",
            "degree": "2",
            "n_samples": "10",
            "hidden_layers": "[64]",
            "max_iterations": "100",
            "random_state": "0",
            "tol": "1e-5",
            "use_lut": "false",
            "get_deltaE": True,
            "show": False,
        },
    ),
    (
        "comma_32_64",
        {
            "cc_method": "ours",
            "mtd": "nn",
            "degree": "1",
            "n_samples": "10",
            "hidden_layers": "32,64",
            "max_iterations": "100",
            "random_state": "0",
            "tol": "1e-5",
            "use_lut": "true",
            "lazy_lut": "true",
            "get_deltaE": True,
            "show": False,
        },
    ),
]


def main():
    results = []
    for name, cc_kwargs in CONFIGS:
        start = time.perf_counter()
        pipeline = ColorCorrection()
        config = Config(
            do_ffc=False,
            do_gc=False,
            do_wb=False,
            do_cc=True,
            check_saturation=False,
            CC_kwargs=cc_kwargs,
        )
        metrics, images, error = pipeline.run(
            "Images/Sample_1.JPG",
            None,
            name,
            config,
        )
        model = pipeline.models.model_cc[0]
        results.append(
            {
                "name": name,
                "error": bool(error),
                "layers": model.hidden_layers,
                "degree": model.degree,
                "elapsed_s": round(time.perf_counter() - start, 3),
                "image_keys": sorted(images.keys()),
                "metric_keys": sorted(metrics.keys()),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
