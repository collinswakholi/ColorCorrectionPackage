"""Record exact wheel Python source identity against the qualified local clone."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument("--wheel", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with zipfile.ZipFile(args.wheel) as archive:
    sources = sorted(name for name in archive.namelist() if name.startswith("ColorCorrectionPipeline/") and name.endswith(".py"))
    records = {}
    for name in sources:
        value = archive.read(name)
        assert value == (args.source / name).read_bytes(), "wheel/source mismatch: " + name
        records[name] = hashlib.sha256(value).hexdigest()
    expected = {path.relative_to(args.source).as_posix() for path in (args.source / "ColorCorrectionPipeline").rglob("*.py")}
    assert set(sources) == expected, (set(sources) ^ expected)
tree = hashlib.sha256("".join(name + "\0" + value + "\n" for name, value in records.items()).encode()).hexdigest()
report = {"wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
    "python_source_tree_sha256": tree, "python_files": len(records), "exact_clone_match": True,
    "files": records}
args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
