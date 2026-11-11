# Python compatibility

The package metadata accepts Python 3.8 and later. Installation also requires
binary wheels for the numerical and inference dependencies. A newer interpreter
can therefore accept the package metadata while its dependencies still prevent
a complete installation.

Use standard CPython 3.13 or 3.14 for a new environment. Existing Python 3.8–3.12
installations remain regression targets. CPython's own security support follows
its [published version lifecycle](https://devguide.python.org/versions/).

## Qualification coverage

The CI matrix targets Python 3.8–3.14. Windows, Linux and macOS jobs exercise
built distributions with the base and optional performance dependencies. Local
Windows tests cover Python 3.8, 3.12, 3.13 and 3.14. Cross-platform results come
from CI. Qualification uses standard CPython builds with the GIL.

Checks include isolated wheel installation, dependency consistency, actual
sample-image detection, 36 model/basis/degree requests, historical trainers,
safe bundle replay, strict exports and precision-sensitive samples. The
performance extra is tested separately so NumPy reference execution remains
available when Numba is absent. The source archive is independently built and
installed, and the examples run outside the repository.

## Python 3.15

On November 6, 2026, a clean binary-only installation under CPython 3.15.2
failed on required dependency availability. ONNX Runtime 1.31.1 and
scikit-image 0.26.0 lacked CPython 3.15 wheels. All maintained package Python
modules parsed under 3.15, but that syntax check cannot qualify a working
installation. Python 3.15 is therefore outside the qualified version set.

The inference runtime supplies chart detection and verified GPU/CPU recovery.
Replacing it to bypass a wheel constraint would require numerical, detector
and device qualification. The package keeps its established inference contract
while upstream wheels become available. The open Python metadata permits
future qualification without changing an arbitrary upper-version bound.
[ONNX Runtime files](https://pypi.org/project/onnxruntime/1.31.1/#files),
[scikit-image files](https://pypi.org/project/scikit-image/0.26.0/#files).

## Qualifying another interpreter

Build the wheel with a qualified interpreter, create an isolated environment
using the target interpreter, and install the wheel with binary dependencies:

```shell
python -m pip install --only-binary=:all: dist/colorcorrectionpipeline-2.1.0-py3-none-any.whl
python -m pip check
python -I scripts/verify_installed.py --expect-numba no --photo Images/Image_1.JPG --output base.json
python -I scripts/verify_v2.py --photos --output base-v2.json
```

Use absolute paths to the verifier scripts and photo when running outside the
checkout. Then install the wheel's performance extra, repeat the checks with
`--expect-numba yes`, and run the packaged quickstart. The CI workflow contains
the full installed-wheel/source-archive procedure.

Promote a Python version to the supported classifiers and blocking CI matrix
after its actual installations and numerical checks pass. Recheck the
dependency versions and device execution evidence when comparing performance.
Saved numerical bundles avoid training-library object deserialization, but
their predictions still need qualification when the numerical stack changes.
