Image/export installed-wheel qualification
==========================================

`verify_installed_image_export.py` supports Python 3.8 and later. Run it with an isolated interpreter that has the built wheel installed. It imports the candidate from that interpreter's site-packages, requires a virtual environment, and loads three frozen P9 source fixtures only as independent numerical/metadata/byte oracles. It does not add the repository to `sys.path`, install dependencies, or benchmark performance.

From a neutral working directory, using absolute paths:

```powershell
$env:TEMP = '<writable-audit-directory>'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = '<writable-audit-directory>\mpl-cache'
& '<isolated-venv>\Scripts\python.exe' -I '<repository>\benchmarks\image_export\verify_installed_image_export.py' --fixtures '<repository>\tests\fixtures' --output '<writable-audit-directory>\installed-image-export.json'
```

The fixtures are `image_contract_p9.py`, `image_data_p9_reference.py`, and `writers_p9_reference.py`. The report records their SHA256 values and the actual interpreter/library versions. Candidate and frozen writer use the same dependencies to isolate the implementation change. An independent frozen contract additionally verifies exact canonical pixel bytes and both historical SHA256 identities.

Each tested runtime passed 208 small contract identity cases, seven large chunked/endian/float16/strided cases, 20 exact encoded-byte and complete ExportRecord comparisons, two transfer conversion exports, four static PNG single-decode cases, three late PNG encoding metadata cases, predecode allocation caps, late ICC rejection, two transactional metadata failures, and short-read/growth checks.

The complete distribution run also applied all five actual P78 bundles in each of four isolated profiles. Maximum differences across profiles were linear 2.665e-15, PLS 2.609e-15, sklearn NN 4.441e-16, custom Torch NN 5.961e-8, and conventional 2.498e-15. Those comparisons used absolute tolerance 2e-12 except custom Torch NN 2e-6, with relative tolerance zero. Same-runtime fitted and photographic bundle reloads were exact.

The four local Windows profiles are Python 3.8.20 and 3.12.10, each with and without optional Numba. They retain their independently installed dependency stacks; installing the candidate used `--no-index --no-deps --force-reinstall`. Actual GPU proof in the clean 3.12 base profile observed 210 WebGPU/D3D12 nodes, with no CPU fallback. Radeon/Apple hardware and remote CI were not exercised here.

Machine-readable evidence is under `docs/validation/image_export/distribution/`. These functional probes make no timing claims; performance results and separate process-level benchmarks are recorded in the image/export timing report.
