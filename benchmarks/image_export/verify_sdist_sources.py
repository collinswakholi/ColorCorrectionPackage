"""Prove candidate sdist carries exact application sources and regression oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--sdist", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
required = ["pyproject.toml", "README.md", "LICENSE", "scripts/verify_installed.py", "scripts/verify_distribution.py",
    "tests/test_unit/test_image_contract_optimization.py", "tests/test_unit/test_image_decode_optimization.py",
    "tests/test_unit/test_strict_export_optimization.py", "tests/fixtures/image_contract_p9.py",
    "tests/fixtures/image_data_p9_reference.py", "tests/fixtures/writers_p9_reference.py",
    "ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md",
    "ColorCorrectionPipeline/detection/models/PROVENANCE.md",
    "ColorCorrectionPipeline/detection/models/LICENSE-AGPL-3.0.txt"]
required += sorted(path.relative_to(args.source).as_posix()
                   for path in (args.source / "ColorCorrectionPipeline").rglob("*.py"))
with tarfile.open(args.sdist, "r:gz") as archive:
    members = {"/".join(Path(member.name).parts[1:]): member
               for member in archive.getmembers() if member.isfile()}
    records = {}
    for name in required:
        assert name in members, "missing candidate source/oracle: " + name
        value = archive.extractfile(members[name]).read()
        assert value == (args.source / name).read_bytes(), "sdist/clone source mismatch: " + name
        records[name] = hashlib.sha256(value).hexdigest()
    assert not any("__pycache__" in name or name.startswith("docs/validation/") for name in members)
report = {"sdist_sha256": hashlib.sha256(args.sdist.read_bytes()).hexdigest(),
          "all_required_sources_exact": True, "exact_python_sources": 43,
          "required_source_and_oracle_count": len(records), "sha256": records}
args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps({key: value for key, value in report.items() if key != "sha256"}, indent=2))
