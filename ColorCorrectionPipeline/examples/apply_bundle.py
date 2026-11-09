"""Apply a saved calibration in a fresh process.

python -m ColorCorrectionPipeline.examples.apply_bundle --bundle run/calibration.ccpb \
    --image scene.png --output replay
"""
import argparse
from pathlib import Path
import time

from ColorCorrectionPipeline import ColorCorrection
from ColorCorrectionPipeline.io import write_image_data
from ._common import new_output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("uint8", "uint16", "float32", "float64"), default="uint8")
    args = parser.parse_args()
    output = new_output(args.output)
    pipeline = ColorCorrection().load_bundle(args.bundle)
    encoding = (pipeline.models.metadata.get("color_semantics") or {}).get("output_encoding", "srgb")
    records = []
    for index, path in enumerate(args.image):
        start = time.perf_counter()
        corrected = pipeline.predict_image(path, stages="final")["CC"]
        apply_seconds = time.perf_counter() - start
        suffix = ".png" if args.dtype in ("uint8", "uint16") else ".tiff"
        record = write_image_data(output / (str(index + 1) + "-" + path.stem + suffix),
                                  corrected, dtype=args.dtype, allow_precision_loss=True,
                                  input_encoding=encoding,
                                  png_profile="fast-lossless" if suffix == ".png" else "compatible")
        records.append({"input": str(path), "output": str(record.path),
                        "apply_seconds": apply_seconds,
                        "maximum_codec_error": record.maximum_codec_error})
    report(output, {"bundle": str(args.bundle),
                    "fitting_policy": pipeline.models.metadata.get("fitting_policy"),
                    "images": records})


if __name__ == "__main__":
    main()
