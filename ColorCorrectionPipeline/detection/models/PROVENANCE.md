# Packaged inference assets

The Python source is MIT licensed. The bundled ONNX models contain Ultralytics
metadata declaring AGPL-3.0; keep the accompanying license and upstream notice.
The source license does not replace those model declarations.

| Asset | Bytes | SHA-256 | Available provenance |
| --- | ---: | --- | --- |
| `detection/models/chart_detector_1280.onnx` | 39,354,948 | `1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240` | Copied unchanged from `ColorCorrectionPackage_UI/models/opt2/best_assemble_fp32_1280.onnx`, inspected at revision `16c6f42176cbd392c167ecabea4bb2519984fda8`; declares Ultralytics 8.4.92. |
| `flat_field/models/plane_det_model_YOLO_512_n.onnx` | 44,658,747 | `1ca8a8745f3cc9ad679a191f264a1b7ea4376f3aacf1c0702a27089a86952f54` | Existing package plane detector; declares Ultralytics 8.4.12. No training record or independently verified base-weight ancestry is supplied. |

The paths above are relative to `ColorCorrectionPipeline`. The package loads
these ONNX files directly. Training checkpoints are not needed for inference and are not distributed. `scripts/verify_distribution.py` checks exact
asset sizes, hashes, notices and license texts before an artifact is qualified.

The chart model's sanitized metadata was inherited from the UI implementation.
The original unsanitized chart artifact and original hashes are not included
here, so this package does not independently reproduce that earlier metadata
sanitization audit. Training-data permissions and model-specific rights records
remain unverified, as disclosed in [UPSTREAM-NOTICE.md](UPSTREAM-NOTICE.md).
