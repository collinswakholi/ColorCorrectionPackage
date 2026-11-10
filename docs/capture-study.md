# Accuracy on independent chart captures

A fit residual measures how well the model reproduces its training chart. To test a new exposure, freeze the reference, calibration image, white field, methods and detector settings first. Apply those calibrations to images from a separate session. Keep their scores out of tuning and model selection. The package's grouped CC validation answers a different question about excluded patch colors within one capture.

From a source checkout with its dependencies installed, run:

```shell
python scripts/evaluate_captures.py study.json study-results
```

Use a new output directory. The manifest uses paths relative to its location. Each `capture_id` identifies an original exposure across conversions or crops. The evaluator rejects repeated IDs, shared fit/evaluation sessions and duplicate decoded pixels. Supply provenance for lossy conversions and crops since their pixel hashes change. Set `chart_index` explicitly if an image has several Classic24 faces. Inspect the saved quadrilaterals and cell order on the raw photographs.

```json
{
  "scope": "controlled",
  "precision": "native-byte",
  "reference": "target-reference.json",
  "measurement_record": "measurements/instrument-export.csv",
  "physical_chart_id": "chart-serial-from-acquisition-record",
  "controlled_setup": "See camera, lens, light, exposure and white-field records in setup.md",
  "fit": {"capture_id": "A-001", "session": "A", "image": "A/chart.png", "white": "A/white.png"},
  "held_out": [
    {"capture_id": "B-002", "session": "B", "image": "B/chart.png", "chart_index": 0},
    {"capture_id": "C-003", "session": "C", "image": "C/darker-chart.png", "difficult_exposure": "one stop lower"}
  ]
}
```

Use a Classic24 `ReferenceProfile.to_dict()` JSON with canonical patch order and declared neutral cells. For a controlled study, create that profile from the instrument record and document its units, observer and illuminant. Use `"scope": "exploratory"` for a nominal reference or unverified conditions. Keep separate manifests for a lighting or camera transfer experiment. Only include images with known chart correspondence. Use distinct shutter releases for repeatability and keep each exposure's acquisition record.

Omitting the fit white field disables flat-field correction. The four presets share gamma correction, white balance, reference, CPU detector and sampling geometry. NN100 uses degree-two polynomial inputs, 100 hidden units, seed 42 and a 10,000-epoch cap. `native-byte` uses the UI-compatible 8-bit input policy and a 33-point LUT. Select `high` for direct float64 correction of higher-precision inputs. Document sensor depth and the conversion when declaring an `original_high_depth` capture.

Each method produces a portable bundle. In `results.json`, `fit` holds the package's own CC residual and separately scored training pixels. `held_out` contains the raw capture and each method's pre-CC (through white balance) and final predictions. The scorer measures unexported sRGB predictions in float64, uses the same 9 × 9 bilinear grid over the central 70% of each fixed patch, averages encoded RGB, and computes CIEDE2000 in Lab at D65 against the adapted reference. Quantizing to PNG before scoring is a different measurement.

Each score includes errors by patch ID, mean, median, 95th percentile, worst cell, mean neutral error, and fraction of sampled RGB triples with any channel at an endpoint. `cc_minus_pre_cc` is paired within a capture. A negative value means CC improved on the upstream stages. The summary weights each capture equally, reports the range of capture means, and gives expected and scored counts. Inspect those counts and the number of distinct sessions before interpreting a mean. A failed fit, detection or prediction suppresses the affected aggregate, records the failure and returns a nonzero exit status.
