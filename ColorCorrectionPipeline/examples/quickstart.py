"""Calibrate RPCC2, correct a scene and verify a saved bundle.

Run with pip install ColorCorrectionPipeline, then:
python -m ColorCorrectionPipeline.examples.quickstart --output results-rpcc2

Add --chart chart.tiff --white white.tiff --scene scene.tiff for your own captures.
Omit --white with a caller-supplied chart to disable flat-field correction.
"""
from ColorCorrectionPipeline import Config, Steps, FFCOptions, CCOptions
from ._common import (sample_paths, input_arguments, inputs, new_output, fit,
                      export_and_replay, report, scores)


def main():
    args = input_arguments(__doc__).parse_args()
    with sample_paths() as samples:
        chart, white, scene = inputs(args, samples)
        output = new_output(args.output)
        config = Config(
            steps=Steps(ffc=FFCOptions(enabled=white is not None),
                        cc=CCOptions(fit={"model": "rpcc2"})),
            input={"frame_index": 0}, chart={"device": args.device},
        )
        pipeline, metrics, elapsed = fit(chart, white, config)
        _, result = export_and_replay(pipeline, scene, output)
        result.update(scores(metrics))
        result.update(calibration_seconds=elapsed, config=config.to_dict(),
                      grouped_config=config.to_grouped_dict(),
                      fitting_policy=pipeline.models.metadata["fitting_policy"],
                      metric_scope="Training-chart patch means. Scene has no measured reference.")
        report(output, result)


if __name__ == "__main__":
    main()
