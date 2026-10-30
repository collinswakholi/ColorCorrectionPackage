# Distribution qualification

Build a wheel and source archive, validate metadata and inspect packaged assets:

```shell
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
python scripts/verify_distribution.py --output /absolute/evidence/artifacts.json
```

The inspector checks package/source versions, the Python 3.8 minimum, optional
Numba metadata, exact ONNX asset sizes and SHA-256 hashes, and model attribution.
It rejects bytecode, Numba caches, pickle artifacts and training checkpoints.
Only the two required ONNX inference assets are distributed.

## Installed-wheel checks

Create a fresh virtual environment without `--system-site-packages`. Install the
wheel, run `python -m pip check`, change to a neutral directory, and invoke the
environment's Python with isolation enabled:

```shell
python -I /absolute/source/scripts/verify_installed.py --expect-numba no --forbid-source /absolute/source --output /absolute/evidence/base.json --photo /absolute/source/Images/Image_1.JPG
```

Install that same wheel with its `performance` extra and repeat using
`--expect-numba yes` and a separate output path. The script checks interpreter
and import isolation, linear/PLS/sklearn-NN/custom-Torch fitting, safe-bundle
reload, prediction parity, guarded FFC, mixed-success and file-mode batches,
uint16 PNG and floating TIFF round trips, prepared CPU backends, honest fallback
and cube export. The photo option fits, saves, reloads and applies an actual
chart image under both staged and native UI policies. Native checks cover
safe reload, byte-input validation, prepared-mode rejection and actual optional
image-kernel execution. A short residual NN fit is compared with its ordered
NumPy reference, with the actual arithmetic backend recorded. Small training
caps verify execution and serialization, not convergence or physical color
accuracy. An external frozen JSON gamma fixture
can optionally be supplied with `--legacy-baseline`.

The evidence records dependency versions, package paths, actual backend choices
and numerical differences. It is generated outside the package and is not part
of a distribution. Timing measurements are described separately in
[performance](performance.md).

## CI and publication

The reusable qualification workflow builds and inspects distributions once,
then installs that wheel in isolated base and performance environments on
Windows, Linux and macOS for Python 3.8 and 3.12, plus Ubuntu for 3.9–3.11.
Linux and Windows runners install CPU Torch from the official PyTorch index;
this does not change package requirements or a user's selected Torch runtime.
The matrix must pass before release. Local Windows execution cannot qualify
unavailable operating systems, Radeon/Apple hardware or newer interpreters.

Numba is optional. Base installations use NumPy fallbacks; strict unavailable
backend requests fail. Prepared CPU selection controls CC/LUT execution, while
legacy gamma retains its historical Torch CUDA behavior. GPU chart detection
has a separate provider policy. Torch and plotting remain mandatory because
existing public imports require them.

The publication workflow runs only after a published GitHub release or explicit
manual `publish=true` dispatch, and requires successful qualification. It uses
the same qualified archives and checks a release tag against the package version.
It does not create tags. Configure required reviewers for the `pypi` environment
and a matching PyPI trusted publisher before enabling publication. A branch push
or draft PR does not publish a package.

## Model terms

Source is MIT licensed. Bundled model metadata declares AGPL-3.0; attribution is
included in wheel and sdist metadata. Read the [upstream notice](../ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md)
and [provenance](../ColorCorrectionPipeline/detection/models/PROVENANCE.md).
Byte/hash qualification preserves known declarations; it does not resolve missing
training-data permissions, model-specific rights records or base-weight ancestry.
