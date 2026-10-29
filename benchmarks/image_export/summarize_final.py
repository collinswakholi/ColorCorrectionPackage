"""Summarize measured runs and assert exact baseline/candidate parity."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(encoding="utf8"))
def no_path(record):
    return {key: value for key, value in record.items() if key != "path"}

baseline = read(ROOT/"baseline"/"results.json")
final_root = ROOT/"final-revision"
candidate = read(final_root/"components"/"results.json")
assert baseline["source_unchanged"] and candidate["source_unchanged"]
rows, comparable = [], 0
for c in candidate["runs"]:
    b = next(row for row in baseline["runs"] if (row["size"], row["operation"]) == (c["size"], c["operation"]))
    if b["record"] is not None:
        assert no_path(b["record"]) == no_path(c["record"]), (c["size"], c["operation"], b["record"], c["record"])
        comparable += 1
    before, after = b["timing"]["median_ms"], c["timing"]["median_ms"]
    rows.append({"size": c["size"], "operation": c["operation"], "baseline_median_ms": before,
        "candidate_median_ms": after, "speedup": before/after,
        "reduction_percent": 100*(1-after/before), "exact_record_parity": True if b["record"] is not None else None})
summary = {"component_rows": rows, "exact_component_records": comparable,
           "memory_baseline": baseline["memory"], "memory_candidate": candidate["memory"],
           "candidate_source_sha256": candidate["source_sha256"], "baseline_source_sha256": baseline["source_sha256"]}
pipeline = ROOT/"pipeline-abba"
if (pipeline/"manifest.json").exists() and "completed_utc" in read(pipeline/"manifest.json"):
    records, output_rows = [], []
    for mode in ("direct", "relaxed"):
        files = {label: read(pipeline/(mode+"-"+label)/"results.json") for label in ("A1", "B1", "B2", "A2")}
        assert all(value["source_unchanged"] for value in files.values())
        assert files["A1"]["source_sha256"] == files["A2"]["source_sha256"]
        assert files["B1"]["source_sha256"] == files["B2"]["source_sha256"] == read(ROOT/"final-components"/"results.json")["source_sha256"]
        runs = {label: value["runs"][0] for label, value in files.items()}
        b = runs["A1"]
        for label, c in runs.items():
            assert (b["input_sha256"], b["checks"]["file_pixels_sha256"], no_path(b["export_record"])) == (c["input_sha256"], c["checks"]["file_pixels_sha256"], no_path(c["export_record"])), (mode, label)
            assert files["A1"]["model_sha256"] == files[label]["model_sha256"]
        for metric in ("file_predict_ms", "strict_export_ms", "file_total_ms"):
            raw = {site: sum([runs[label]["timings"][metric]["raw_ms"] for label in labels], []) for site, labels in (("baseline", ("A1", "A2")), ("candidate", ("B1", "B2")))}
            before, after = statistics.median(raw["baseline"]), statistics.median(raw["candidate"])
            output_rows.append({"mode": mode, "metric": metric, "baseline_median_ms": before,
                "candidate_median_ms": after, "speedup": before/after,
                "reduction_percent": 100*(1-after/before), "raw_ms": raw})
        records.append({"mode": mode, "pixel_sha256": b["checks"]["file_pixels_sha256"],
            "export_record": no_path(b["export_record"]), "model_sha256": files["A1"]["model_sha256"]})
    summary["intermediate_pipeline_rows"], summary["intermediate_pipeline_parity"] = output_rows, records
    output_rows, records = [], []
    for mode in ("direct", "relaxed"):
        final = read(final_root/("candidate-"+mode)/"results.json")
        assert final["source_unchanged"] and final["source_sha256"] == candidate["source_sha256"]
        c = final["runs"][0]
        controls = [read(pipeline/(mode+"-"+label)/"results.json") for label in ("A1", "A2")] if mode == "direct" else [read(final_root/"control-relaxed"/"results.json")]
        for control in controls:
            assert control["source_unchanged"] and control["source_sha256"] == baseline["source_sha256"]
            b = control["runs"][0]
            assert (b["input_sha256"], b["checks"]["file_pixels_sha256"], no_path(b["export_record"]), control["model_sha256"]) == (c["input_sha256"], c["checks"]["file_pixels_sha256"], no_path(c["export_record"]), final["model_sha256"]), mode
        intermediate = next(row for row in summary["intermediate_pipeline_parity"] if row["mode"] == mode)
        assert c["checks"]["file_pixels_sha256"] == intermediate["pixel_sha256"] and no_path(c["export_record"]) == intermediate["export_record"]
        for metric in ("file_predict_ms", "strict_export_ms", "file_total_ms"):
            raw = {"baseline": sum([value["runs"][0]["timings"][metric]["raw_ms"] for value in controls], []), "candidate": c["timings"][metric]["raw_ms"]}
            before, after = statistics.median(raw["baseline"]), statistics.median(raw["candidate"])
            output_rows.append({"mode": mode, "metric": metric, "baseline_median_ms": before,
                "candidate_median_ms": after, "speedup": before/after,
                "reduction_percent": 100*(1-after/before), "raw_ms": raw,
                "baseline_n": len(raw["baseline"]), "candidate_n": len(raw["candidate"])})
        records.append({"mode": mode, "pixel_sha256": c["checks"]["file_pixels_sha256"], "export_record": no_path(c["export_record"]), "model_sha256": final["model_sha256"]})
    summary["pipeline_rows"], summary["pipeline_parity"] = output_rows, records
if (ROOT/"ui-final"/"results.json").exists():
    summary["ui"] = read(ROOT/"ui-final"/"results.json")
if (ROOT/"copy-control"/"results.json").exists():
    summary["copy_control"] = read(ROOT/"copy-control"/"results.json")
summary["contracts_ablation"] = read(ROOT/"ablations"/"results.json")
summary["quantization_ablation"] = read(ROOT/"quantization-ablation"/"results.json")
(ROOT/"final-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
print(json.dumps({key: value for key, value in summary.items() if key in ("component_rows", "exact_component_records", "pipeline_rows")}, indent=2))
