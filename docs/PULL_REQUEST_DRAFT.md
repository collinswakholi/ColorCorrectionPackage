# Draft PR: qualify model-based color correction and exact I/O execution

Calibration failures could leave partial state, source/reference meaning was implicit, and persistence lacked complete validation. This change makes calibration transactional, reports stage and detector failures, defaults chart detection to a bundled ONNX model with observable GPU/CPU execution, and adds explicit contracts, reproducible opt-in algorithms and non-executable bundles.

Existing configurations, legacy prediction APIs and saved numerical models retain their compatibility routes. New bounded batch, strict image export, compact reporting and qualified prepared execution provide explicit precision and ownership policies. The final I/O work fuses exact float64 normalization/quantization, isolates public pixel headers and adds an opt-in filter-None/uncompressed PNG profile. Default pixels, digests, encoded PNG bytes and export facts match the frozen implementation on the tested inputs; every profile retains fsync/read-back/decode/sample/metadata/atomic publication checks.

Final paired 4K file→verified PNG8 medians: 1880.50ms frozen, 1674.84ms compatible default, 1505.32ms explicit fast-lossless. The full-file UI API gap remains open; different fitting/arithmetic/interpolation/persistence prevents treating UI timing as a float64 accuracy oracle.

Validation: 1261 tests pass, including the external-photo corpus; four isolated Python 3.8/3.12 base/performance profiles pass; old bundles and serialization preserve predictions; dense and singleton quantization/error facts match exactly; default bytes/all ExportRecord facts and fast-profile decoded samples match. Actual chart and UI GPU execution is verified. Archive/license/Twine/source-build checks and all raw evidence are linked in `docs/IO_FINAL_OPTIMIZATION.md`.

This is a local review commit, not a release request. Remote OS CI and unavailable hardware gates remain unrun. Model provenance/license limitations remain disclosed. The candidate version must advance from 1.4.6 before future publication.
