"""Use high precision for calibration and write verified TIFF plus a display PNG.

python -m ColorCorrectionPipeline.examples.high_precision --output precision-demo

The built-in JPEG demo expands stored byte codes into uint16 containers. It
demonstrates the API and creates no additional capture precision. Pass genuine
uint16/float TIFF files with --chart, --white and --scene for scientific inputs.
"""
import numpy as np
from ColorCorrectionPipeline import Config, ImageData, Steps, FFCOptions, CCOptions
from ColorCorrectionPipeline.io import read_image_data, write_image_data
from ._common import (sample_paths, input_arguments, inputs, new_output, fit,
                      export_and_replay, report, scores)


def high_image(path):
    source = read_image_data(path, color_space="srgb", frame_index=0)
    if source.source_dtype == "uint8":
        return ImageData.from_array(np.rint(source.to_srgb() * 65535).astype(np.uint16),
                                    color_space="srgb",
                                    provenance={"example_conversion": "uint8 codes expanded to uint16"})
    return source


def main():
    args = input_arguments(__doc__).parse_args()
    with sample_paths() as samples:
        chart, white, scene = inputs(args, samples)
        output = new_output(args.output)
        config = Config(
            steps=Steps(ffc=FFCOptions(enabled=white is not None),
                        cc=CCOptions(fit={"model": "rpcc2"})),
            numerics={"precision": "high"}, chart={"device": args.device},
        )
        pipeline, metrics, elapsed = fit(high_image(chart),
                                         high_image(white) if white is not None else None, config)
        corrected, result = export_and_replay(pipeline, high_image(scene), output, dtype="float64")
        integer = write_image_data(output / "corrected-uint16.tiff", corrected,
                                   dtype="uint16", allow_precision_loss=True)
        preview = write_image_data(output / "preview.png", corrected, dtype="uint8",
                                   allow_precision_loss=True)
        result.update(scores(metrics))
        result.update(calibration_seconds=elapsed, fitting_policy=pipeline.models.metadata["fitting_policy"],
                      uint16=str(integer.path), display_preview=str(preview.path),
                      sample_note="Byte samples expanded to uint16 when source captures are uint8.")
        report(output, result)


if __name__ == "__main__":
    main()
