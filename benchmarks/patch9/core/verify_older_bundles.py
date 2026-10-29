"""Local audit: apply actual patch-7/8 bundles across isolated dependency stacks."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import site
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-baseline", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / "matplotlib-cache"))
    import numpy as np
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline import ColorCorrection
    from ColorCorrectionPipeline.core.accel import HAS_NUMBA

    prefix = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix
    assert "include-system-site-packages = false" in (prefix / "pyvenv.cfg").read_text().lower()
    assert site.ENABLE_USER_SITE is False and prefix in Path(package.__file__).resolve().parents
    source = args.fixtures / "input.npy"
    image = np.load(source, allow_pickle=False)
    assert np.isfinite(image).all()
    results = {}
    for family in ("linear", "pls", "nn", "custom", "conv"):
        bundle = args.fixtures / (family + ".ccpb")
        expected_path = args.fixtures / (family + "-expected.npy")
        expected = np.load(expected_path, allow_pickle=False)
        model = ColorCorrection().load_bundle(bundle)
        actual = model.predict_image(image, stages="final")["CC"]
        assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
        tolerance = 2e-6 if family == "custom" else 2e-12
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
        results[family] = {"max_difference": float(np.max(np.abs(actual - expected))),
            "atol": tolerance, "bundle_sha256": digest(bundle), "expected_sha256": digest(expected_path),
            "output_shape": list(actual.shape), "output_dtype": str(actual.dtype)}
    versions = {}
    for name in ("ColorCorrectionPipeline", "numpy", "scikit-learn", "scipy", "torch", "colour-science", "onnxruntime", "numba"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    reference = None
    if args.legacy_baseline:
        baseline = json.loads(args.legacy_baseline.read_text(encoding="utf8"))
        legacy = ColorCorrection()
        actual = legacy.get_reference_values()
        assert actual.index.tolist() == baseline["reference_names"]
        np.testing.assert_allclose(actual.values, baseline["reference_rgb"], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(legacy.REF_ILLUMINANT, baseline["reference_illuminant"])
        reference = {"max_difference": float(np.max(np.abs(actual.values - np.asarray(baseline["reference_rgb"])))),
            "atol": 1e-10, "names_match": True, "illuminant_matches": True}
    report = {"python": platform.python_version(), "executable": sys.executable,
        "package_file": str(package.__file__), "system_site_packages": False,
        "user_site_enabled": site.ENABLE_USER_SITE, "numba_present": HAS_NUMBA,
        "input_sha256": digest(source), "source_validation_sha256": digest(args.fixtures / "validation.json"),
        "versions": versions, "families": results, "legacy_reference": reference}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
