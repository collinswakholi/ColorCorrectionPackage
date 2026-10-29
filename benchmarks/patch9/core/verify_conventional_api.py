"""Qualify actual old/current Colour matrix APIs without extra test dependencies."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / "matplotlib-cache"))
import colour
import numpy as np
import ColorCorrectionPipeline as package
from ColorCorrectionPipeline.conventional import apply_conventional, conventional_basis

assert Path(sys.prefix).resolve() in Path(package.__file__).resolve().parents
cases = [{"method": "Cheung 2004", "terms": n} for n in (3, 5, 7)] + [
    {"method": "Finlayson 2015", "degree": 1},
    {"method": "Finlayson 2015", "degree": 2},
    {"method": "Finlayson 2015", "degree": 2, "root_polynomial_expansion": False},
    {"method": "Finlayson 2015", "degree": 3, "root_polynomial_expansion": True},
    {"method": "Finlayson 2015", "degree": 3, "root_polynomial_expansion": None},
] + [{"method": "Vandermonde", "degree": n} for n in (1, 2, 3)]
records = []
for options in cases:
    print("Checking " + json.dumps(options), flush=True)
    rng = np.random.default_rng(124)
    image = rng.uniform(.05, .95, (2, 5, 3))
    basis = conventional_basis(image.reshape(-1, 3), **options)
    matrix = rng.normal(0, .2, (3, basis.shape[1]))
    native = getattr(colour.characterisation, "apply_matrix_colour_correction", None)
    if callable(native):
        expected = native(image, matrix, **options)
        expected_single = native(image[0, 0], matrix, **options)
    else:
        observed = rng.uniform(.05, .95, (48, 3))
        targets = conventional_basis(observed, **options) @ matrix.T
        matrix = colour.characterisation.matrix_colour_correction(observed, targets, **options)
        expected = colour.characterisation.colour_correction(image, observed, targets, **options)
        # Colour 0.4.1's Finlayson expansion has an upstream singleton-squeeze
        # bug with NumPy 1.24. A duplicated row is the same pointwise oracle.
        expected_single = colour.characterisation.colour_correction(np.stack([image[0, 0], image[0, 0]]), observed, targets, **options)[0]
    fitted_helpers = {name: getattr(colour.characterisation, name) for name in (
        "matrix_colour_correction", "matrix_colour_correction_Cheung2004",
        "matrix_colour_correction_Finlayson2015", "matrix_colour_correction_Vandermonde")}
    def forbidden(*a, **kw):
        raise AssertionError("applying a saved matrix attempted to refit")
    for name in fitted_helpers:
        setattr(colour.characterisation, name, forbidden)
    try:
        actual = apply_conventional(image, matrix, **options)
        np.testing.assert_array_equal(actual, expected)
        for value in (image[0, 0], image[0, :1], image[:1, :1]):
            actual_single = apply_conventional(value, matrix, **options).reshape(3)
            if callable(native):
                np.testing.assert_array_equal(actual_single, native(value, matrix, **options).reshape(3))
            else:
                np.testing.assert_allclose(actual_single, expected_single, rtol=0, atol=2e-14)
    finally:
        for name, function in fitted_helpers.items():
            setattr(colour.characterisation, name, function)
    records.append({"options": options, "max_difference": float(np.max(np.abs(actual - expected))), "no_refit": True})
report = {"python": sys.version, "colour_science": colour.__version__, "numpy": np.__version__,
    "package_file": str(package.__file__), "cases": records, "passed": len(records)}
args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
