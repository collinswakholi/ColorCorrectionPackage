"""Compare solvers and bases on the same chart and capture settings.

python -m ColorCorrectionPipeline.examples.compare_bases --output comparison \
    --models linear,pls,nn --bases polynomial,root,hybrid --degrees 2 --pls-components 3

NN uses its full 10000-epoch cap unless --nn-epochs is supplied.
All reported held-out colors share the already fitted upstream stages.
"""
from ColorCorrectionPipeline import Config, Steps, FFCOptions, CCOptions
from ._common import (sample_paths, input_arguments, inputs, new_output, fit,
                      export_and_replay, report, scores)


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = input_arguments(__doc__)
    parser.add_argument("--models", default="linear,pls,nn", type=comma_list)
    parser.add_argument("--bases", default="polynomial,root,hybrid", type=comma_list)
    parser.add_argument("--degrees", default="2", type=comma_list)
    parser.add_argument("--pls-components", default=3, type=int)
    parser.add_argument("--nn-epochs", default=10000, type=int)
    args = parser.parse_args()
    if not set(args.models) <= {"ccm", "rpcc2", "rpcc3", "linear", "pls", "nn"}:
        parser.error("Unknown model.")
    if not set(args.bases) <= {"polynomial", "root", "hybrid"}:
        parser.error("Unknown feature basis.")
    degrees = [int(value) for value in args.degrees]
    if not degrees or not set(degrees) <= {1, 2, 3, 4}:
        parser.error("Degrees must be in 1,2,3,4.")
    with sample_paths() as samples:
        chart, white, scene = inputs(args, samples)
        output = new_output(args.output)
        results = []
        for model in args.models:
            cases = [(None, None)] if model in ("ccm", "rpcc2", "rpcc3") else [
                (basis, degree) for basis in args.bases for degree in degrees]
            for basis, degree in cases:
                selection = {"model": model}
                if model == "pls":
                    selection["pls"] = {"components": args.pls_components}
                if model == "nn":
                    selection["nn"] = {"hidden_layers": [100], "max_iterations": args.nn_epochs}
                if basis is not None:
                    selection.update(basis=basis, degree=degree)
                config = Config(
                    steps=Steps(ffc=FFCOptions(enabled=white is not None),
                                cc=CCOptions(fit=selection,
                                    validation={"enabled": True, "n_splits": 4, "random_state": 42})),
                    input={"frame_index": 0}, chart={"device": args.device},
                )
                name = model if basis is None else model + "-" + basis + "-d" + str(degree)
                folder = new_output(output / name)
                pipeline, metrics, elapsed = fit(chart, white, config)
                _, result = export_and_replay(pipeline, scene, folder)
                result.update(scores(metrics))
                numerical = pipeline.numerical_report["CC"]
                result.update(name=name, calibration_seconds=elapsed,
                              fit=numerical["fit"], validation=numerical["validation"])
                report(folder, result)
                results.append(result)
        report(output, {"comparisons": results,
                        "scope": "Held-out CC colors conditional on fitted FFC/GC/WB. "
                                 "Independent scene pixels have no reference truth.",
                        "nn_epoch_cap": args.nn_epochs})


if __name__ == "__main__":
    main()
