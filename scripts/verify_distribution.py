"""Inspect built archives without importing the package or detector runtimes."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


PACKAGE = "ColorCorrectionPipeline/"
MODELS = PACKAGE + "detection/models/"
ASSETS = {
    MODELS + "chart_detector_1280.onnx": (39354948, "1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240"),
    PACKAGE + "flat_field/models/plane_det_model_YOLO_512_n.onnx": (44658747, "1ca8a8745f3cc9ad679a191f264a1b7ea4376f3aacf1c0702a27089a86952f54"),
}


def verify_files(files, read):
    for name in files:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, "unsafe archive path: " + name
        assert not name.endswith((".pt", ".pkl", ".pickle", ".pyc", ".nbc", ".nbi")), "unneeded executable/cache asset: " + name
        assert "__pycache__" not in path.parts, "cache directory: " + name
    records = {}
    for name, (size, digest) in ASSETS.items():
        assert name in files, "missing inference asset: " + name
        data = read(name)
        actual = hashlib.sha256(data).hexdigest()
        assert len(data) == size and actual == digest, "inference asset differs: " + name
        records[name] = {"bytes": len(data), "sha256": actual}
    for name in ("LICENSE-AGPL-3.0.txt", "UPSTREAM-NOTICE.md", "PROVENANCE.md", "manifest.json"):
        assert MODELS + name in files, "missing detector attribution: " + name
    notice = read(MODELS + "UPSTREAM-NOTICE.md").decode("utf8")
    assert "AGPL-3.0" in notice and "8.4.92" in notice and "8.4.12" in notice
    manifest = json.loads(read(MODELS + "manifest.json"))
    assert manifest["sha256"] == ASSETS[MODELS + "chart_detector_1280.onnx"][1]
    assert manifest["bytes"] == ASSETS[MODELS + "chart_detector_1280.onnx"][0]
    return records


def verify_wheel(path, expected_version=None):
    with zipfile.ZipFile(path) as archive:
        files = set(archive.namelist())
        records = verify_files(files, archive.read)
        names = [name for name in files if name.endswith(".dist-info/METADATA")]
        assert len(names) == 1, "expected one METADATA file"
        metadata = BytesParser().parsebytes(archive.read(names[0]))
        assert metadata["Name"].lower() == "colorcorrectionpipeline"
        version = metadata["Version"]
        if expected_version:
            assert version == (expected_version[1:] if expected_version.startswith("v") else expected_version), (version, expected_version)
        source = archive.read(PACKAGE + "__version__.py").decode("utf8")
        assert re.search(r'__version__\s*=\s*[\"\']' + re.escape(version) + r'[\"\']', source)
        assert metadata["Requires-Python"] == ">=3.8", metadata["Requires-Python"]
        requirements = metadata.get_all("Requires-Dist", [])
        numba = [req for req in requirements if req.lower().startswith("numba")]
        assert numba and all("extra == 'performance'" in req or 'extra == "performance"' in req for req in numba), numba
        assert "performance" in metadata.get_all("Provides-Extra", [])
        assert "bundled detector models declare AGPL-3.0" in metadata["License"]
        for suffix in ("/licenses/LICENSE", "/licenses/ColorCorrectionPipeline/detection/models/LICENSE-AGPL-3.0.txt", "/licenses/ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md"):
            assert any(name.endswith(suffix) for name in files), "missing metadata license: " + suffix
        return {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": version, "requires_python": metadata["Requires-Python"], "numba_optional": True, "assets": records}


def verify_sdist(path, expected_version=None):
    with tarfile.open(path, "r:gz") as archive:
        all_members = archive.getmembers()
        for member in all_members:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and ".." not in name.parts, "unsafe archive path: " + member.name
            assert not member.issym() and not member.islnk(), "linked source archive member: " + member.name
        members = [member for member in all_members if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        assert len(roots) == 1, "sdist must have one root"
        mapped = {"/".join(PurePosixPath(member.name).parts[1:]): member for member in members}
        def read(name):
            return archive.extractfile(mapped[name]).read()
        records = verify_files(set(mapped), read)
        for required in ("pyproject.toml", "README.md", "LICENSE", "scripts/verify_installed.py", "scripts/verify_distribution.py"):
            assert required in mapped, "missing sdist build/qualification source: " + required
        metadata = BytesParser().parsebytes(read("PKG-INFO"))
        if expected_version:
            assert metadata["Version"] == expected_version.lstrip("v")
        return {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "assets": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    wheels = list(args.dist_dir.glob("*.whl"))
    sdists = list(args.dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, "expected exactly one wheel and one sdist"
    report = {"wheel": verify_wheel(wheels[0], args.expected_version), "sdist": verify_sdist(sdists[0], args.expected_version)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
