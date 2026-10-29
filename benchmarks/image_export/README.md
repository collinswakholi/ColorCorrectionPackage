# Image/export benchmark reproduction

These workers time public APIs without disabling validation, color math, hashing or persistence. Run one worker at a time in the qualified Python environment. The recorded run used the audit Python 3.12 environment and frozen patch-9 fixtures/models; see `docs/validation/image_export/final-summary.json` and the recorded manifests for exact fingerprints and commands. Model fitting is not part of these timings.

From the repository root (replace the explicit fixture/site placeholders):

```powershell
python benchmarks/image_export/bench_image_io.py --site . --inputs FROZEN_INPUT_DIR --output NEW_OUTPUT_DIR --profile --memory
python benchmarks/image_export/bench_image_io.py --site FROZEN_P9_WHEEL_SITE --inputs FROZEN_INPUT_DIR --output NEW_BASELINE_DIR --profile --memory
python benchmarks/image_export/bench_pipeline_io.py --site . --models FROZEN_PACKAGE_MODELS --inputs FROZEN_INPUT_DIR --output NEW_PIPELINE_DIR --sizes 4k --modes relaxed --repetitions 3 --file-only
python benchmarks/image_export/bench_ui_apply.py --ui ../CCP_New_UI --native FROZEN_NATIVE_ROOT --prior-benchmark FROZEN_UI_BENCHMARK_JSON --inputs FROZEN_INPUT_DIR --output NEW_UI_DIR --repetitions 7
```

All outputs must be fresh directories. `bench_image_io.py` accepts `--sizes`, `--operations` and `--repetitions`. `bench_pipeline_io.py` accepts `--modes direct` or `--modes relaxed`; the latter fixes the original measured LUT129 qualification budgets. Startup, fitting, and execution preparation are excluded from warm operation timings and recorded separately where applicable.

The `run_*`, `check_public_copy.py`, `ablate_quantization.py`, `summarize_final.py` and `write_measured_report.py` files preserve historical audit replay paths. Their recorded audit hierarchy is explicit in source and manifests; generic CLI workers above are preferred for new environments. `ablate_contracts.py` is a generic CLI diagnostic; it monkeypatches only its own process and does not edit application source.

Small raw JSON, text profiles and logs are under `docs/validation/image_export`. Binary profiles, all PNG/native/model fixtures and large generated image outputs remain local in the original audit. The repository evidence contains source, input, model, payload and worker SHA256 fingerprints so changes cannot be silently attributed to an old timing run. Final candidate n=3, contemporary relaxed control n=3, earlier direct control n=6 and intermediate ABBA n=6/source are distinguished in the report. The integer quantizer diagnostic freezes its original A implementation inline so it remains reproducible after B has shipped; this replay clarification does not alter the recorded algorithms or application source.
