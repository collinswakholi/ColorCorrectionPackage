# Distribution qualification

The package version remains **1.4.6** during these local patches. A successful
local build is not a published release, and the workflows never create or push
tags. The source API and existing prediction defaults remain compatible.

## Installation profiles

```shell
pip install ColorCorrectionPipeline
pip install "ColorCorrectionPipeline[performance]"
```

The base installation includes the scientific stack, ONNX detector runtimes,
Torch and plotting dependencies needed by current imports and public APIs. Numba
is optional through `performance`; a base installation uses NumPy CPU fallbacks.
An explicit unavailable Numba request reports the fallback, or fails when
`allow_fallback=False`. Installing the extra does not opt existing calls into the
new prepared-execution API or change their numerical policy.
The requested prepared backend controls CC/LUT evaluation. Legacy gamma
application retains its existing runtime behavior, including Torch CUDA when
available; selecting a prepared CPU backend does not override that legacy stage.

Torch and plotting are still mandatory. A smaller detector-only or fitting-only
dependency profile would require a separate import-boundary refactor and further
qualification. This patch does not claim a minimal package footprint.

Python **3.8 and later** remains the metadata contract. Local Windows qualification
uses isolated Python **3.8.20** and **3.12.10** environments with no inherited
system or user site packages. Compatible dependency versions are resolved for
each interpreter; they are recorded in the JSON evidence rather than assumed
identical. The Python 3.8 source build selects Hatchling 1.27.0, while newer
interpreters can select a newer compatible backend. The CI matrix covers 3.8–3.12
and Windows, Linux and macOS; that matrix must actually pass before a release.
Local Windows results do not establish execution on the other platforms or on
unavailable CUDA, Radeon or Apple hardware.
Interpreters newer than 3.12 are allowed by the dependency metadata but are not
locally tested or covered by this CI matrix.

## Reproducible archive and installed-wheel checks

Build the sdist and then its wheel, inspect both, and check package metadata:

```shell
python -m build
python -m twine check --strict dist/*
python scripts/verify_distribution.py --output evidence/artifacts.json
```

The archive inspector checks package/source versions, minimum Python metadata,
the optional Numba requirement, the exact sizes and SHA-256 hashes of both ONNX
inference models, and model notices/license files. It rejects cached bytecode,
Numba caches, pickle artifacts and the unused `.pt` training checkpoint. The
checkpoint remains available in the working repository for compatibility, but
is excluded from wheel and sdist. It contributed about 20.7 MB compressed to
the previous wheel and is not loaded by package inference.

Install the wheel into a fresh virtual environment created without
`--system-site-packages`. Run from a neutral directory using the environment's
Python and an absolute script path:

```shell
python -I /absolute/source/scripts/verify_installed.py --expect-numba no --forbid-source /absolute/source --output /absolute/evidence/base.json --photo /absolute/source/Images/Image_1.JPG --legacy-baseline /absolute/source/tests/fixtures/legacy_numeric_baseline.json
```

Run a dependency consistency check. Then install the same wheel with its
`performance` extra and repeat with `--expect-numba yes` and a separate output
path. The script verifies its own interpreter, import paths and isolation. It
fits linear, PLS, sklearn NN and custom Torch models from grouped observations;
saves/loads safe bundles; compares direct and image predictions; checks mixed
batch failure isolation, file-mode batch results, guarded flat-field fitting,
uint16 PNG and float TIFF round trips. It also checks prepared NumPy/auto/Numba
execution, honest unavailable-backend behavior, and cube export. The optional
photo path exercises the packaged detector and a full image fit/save/load/apply
workflow. Small training caps exercise serialization and inference contracts;
they do not establish convergence or calibration quality.
The legacy baseline option compares the original frozen gamma fixture within
its established `2e-6` tolerance and requires exact same-runtime legacy GC/WB
bundle reload predictions, with and without the performance extra.

Evidence records the installed dependency versions, package location, import
paths, backend selections and numerical differences. Performance claims require
the separate measured benchmark report; these smoke checks are not benchmarks.

## Release gates

`qualification.yml` runs the full test suite, builds once, validates archives,
and installs the same wheel in isolated base and performance environments. The
normal tests workflow calls it. Publication only runs for an explicitly
published GitHub release or manual dispatch with `publish=true`, after all
qualification jobs succeed. It downloads the qualified archives without
rebuilding and verifies a release tag matches the package version.
Linux and Windows CI runners preinstall CPU Torch from the
[official PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/), avoiding
unneeded CUDA runtime downloads for these CPU correctness checks. This does not
change the dependency metadata or a user's selected Torch installation.
The 24 tests requiring an external photo corpus are skipped by the default CI
command unless `CCP_PHOTO_DIR` is supplied. The installed-wheel CI smoke uses
the repository's bundled `Images/Image_1.JPG` for one real-photo workflow; it
does not substitute for execution of the external corpus. Local validation can
set `CCP_PHOTO_DIR` to the separately available UI corpus and report those
results independently. The matrix is defined here and has not been executed
remotely during this local patch work.

Before enabling publication, configure the repository's **pypi** environment
with required reviewers and configure a matching PyPI trusted publisher. Merely
naming an environment in YAML does not create a review policy. Repository/PyPI
settings cannot be qualified by the local test suite. No workflow has been
executed remotely and no publication, remote tag or push is part of local patch
qualification.

## Model distribution limitations

Source code is MIT licensed. Bundled model metadata declares AGPL-3.0, which is
disclosed in package metadata and retained in wheel/sdist license files. Read the
packaged [upstream notice](../ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md)
and [asset provenance](../ColorCorrectionPipeline/detection/models/PROVENANCE.md).
Artifact qualification preserves the known declarations and checks exact bytes;
it does not resolve missing training-data permissions, base-weight ancestry or
model-specific rights records.
