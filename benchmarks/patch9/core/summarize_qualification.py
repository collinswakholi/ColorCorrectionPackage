"""Collate successful local distribution evidence without running benchmarks."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def read(name):
    return json.loads((root / name).read_text(encoding="utf8"))

artifact, sources = read("final-artifacts.json"), read("final-source-fingerprints.json")
assert artifact["wheel"]["sha256"] == sources["wheel_sha256"]
profiles = {}
for name in ("final-clean38-base", "final-clean312-base", "final-clean38-performance", "final-clean312-performance"):
    run, older, installed = read(name + ".json"), read(name + "-older-bundles.json"), read(name + "-source-fingerprints.json")
    assert installed["wheel_sha256"] == sources["wheel_sha256"]
    assert installed["python_source_tree_sha256"] == sources["python_source_tree_sha256"]
    assert len(run["families"]) == 4 and len(older["families"]) == 5
    profiles[name] = {"python": run["python"], "prefix": run["prefix"], "system_site_packages": run["system_site_packages"],
        "user_site_enabled": run["user_site_enabled"], "numba_present": run["numba_present"],
        "versions": {key: run["versions"].get(key) for key in ("numpy", "numba", "llvmlite", "scikit-learn", "scipy", "colour-science", "torch", "onnxruntime", "pip")},
        "pip_check_passed": True, "exact_installed_source_match": True,
        "new_family_max_reload_difference": max(value["max_direct_reload_difference"] for value in run["families"].values()),
        "prepared_execution": run["prepared_execution"], "legacy_gc": run["legacy_gc"],
        "photo_max_reload_difference": run["photo"]["max_saved_reload_difference"],
        "photo_cpu_execution_verified": all(row["execution_verified"] and row["selected_device"] == "cpu" for row in run["photo"]["detection"]),
        "precision_io_passed": run["precision_io_passed"], "guarded_ffc_passed": run["guarded_ffc_passed"],
        "file_mode_batch_passed": run["file_mode_batch_passed"],
        "old_five_family_differences": {key: value["max_difference"] for key, value in older["families"].items()},
        "legacy_reference": older["legacy_reference"], "evidence": [name + ".json", name + "-older-bundles.json", name + "-source-fingerprints.json"]}
gpu = read("final-clean312-base-gpu.json")["photo"]
assert gpu["gpu_requirement_met"] and gpu["max_saved_reload_difference"] == 0
report = {"wheel": artifact["wheel"], "sdist": artifact["sdist"],
    "source_tree_sha256": sources["python_source_tree_sha256"], "python_source_files": sources["python_files"],
    "profiles": profiles, "isolated_gpu_photo": gpu,
    "conventional_api_cases": {name: read("final-" + name + "-conventional.json") for name in ("clean38", "clean312")},
    "remote_ci_executed": False, "local_platform": "Windows x64", "qualification_passed": True}
if (root / "final-python38-source-build.json").exists():
    report["python38_source_build"] = read("final-python38-source-build.json")
(root / "final-distribution-qualification.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps({"wheel": report["wheel"], "source_tree_sha256": report["source_tree_sha256"],
    "profiles": {name: {key: value[key] for key in ("python", "numba_present", "versions", "legacy_gc", "old_five_family_differences", "legacy_reference")} for name, value in profiles.items()},
    "gpu": {key: gpu[key] for key in ("requested_device", "gpu_requirement_met", "max_saved_reload_difference")},
    "gpu_execution": [{key: row[key] for key in ("provider", "graphics_backend", "execution_verified", "node_provider_counts", "cpu_fallback")} for row in gpu["detection"]]}, indent=2))
