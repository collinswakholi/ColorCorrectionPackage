"""Refit an explicit historical estimator and replay its safe calibration.

python -m ColorCorrectionPipeline.examples.legacy_training --output historical --method custom

This CC-only tutorial uses the historical chart sampling, feature expansion and
selected sklearn/Torch trainer. Its iteration cap is deliberately short.
"""
from ColorCorrectionPipeline import Config
from ._common import (sample_paths, input_arguments, inputs, new_output, fit,
                      export_and_replay, report)


def main():
    parser = input_arguments(__doc__)
    parser.add_argument("--method", choices=("linear", "pls", "nn", "custom"), default="custom")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    with sample_paths() as samples:
        chart, _, scene = inputs(args, samples)
        output = new_output(args.output)
        config = Config(
            fitting_policy="legacy-v1", do_ffc=False, do_gc=False, do_wb=False,
            check_saturation=False, chart_detection={"device": args.device},
            CC_kwargs={"mtd": args.method, "degree": 2, "hidden_layers": [64],
                       "max_iterations": args.iterations, "random_state": 42,
                       "use_lut": False},
        )
        pipeline, metrics, elapsed = fit(chart, None, config)
        _, result = export_and_replay(pipeline, scene, output)
        fitted = pipeline.models.model_cc[0]
        result.update(method=args.method, estimator=type(fitted.model).__name__,
                      calibration_seconds=elapsed, config=config.to_dict(),
                      metrics=metrics, iteration_cap=args.iterations)
        report(output, result)


if __name__ == "__main__":
    main()

