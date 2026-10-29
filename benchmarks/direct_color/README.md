# Reproduce direct color qualification

The primary public benchmark workers are `bench_direct.py` (explicit --site/--models/--inputs/--output) and `bench_ui.py` (--ui/--native/--prior-benchmark/--inputs/--output, --fast-mode true or false). Use the frozen model/input hashes in the result JSON; do not refit. Set writable TEMP/TMP before imports, and run timing workers sequentially.

`bench_paired.py`, `run_benchmarks.py`, `stress_direct_color.py`, `run_tests.py` and `run_distribution.py` are historical local drivers; their ROOT/AUDIT paths refer to the original audit staging layout. Adjust them or invoke the original audit copies. The paired benchmark loads the frozen acceleration module with the unchanged reference color/CC code and alternates both public file workflows in one process.

`verify_installed_direct.py --expect-numba yes|no --output <path>` is portable Python3.8+ installed-wheel validation. Run it with isolated `python -I` from a neutral directory after installing the wheel. It checks the actual source import, tone backend, float64 error and exact bundle/trusted-local-pickle reloads. Existing full installation/image-export probes remain in scripts/ and benchmarks/image_export/.

No driver publishes artifacts or pushes Git state. Raw results and the exact commands are in docs/validation/direct_color.
