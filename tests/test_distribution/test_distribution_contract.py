"""Adversarial archive checks; no package import or heavyweight inference."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_distribution", ROOT / "scripts/verify_distribution.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def contents(monkeypatch):
    data = {name: ("test asset " + name).encode() for name in verifier.ASSETS}
    hashes = {name: (len(value), hashlib.sha256(value).hexdigest()) for name, value in data.items()}
    monkeypatch.setattr(verifier, "ASSETS", hashes)
    chart = hashes[verifier.MODELS + "chart_detector_1280.onnx"]
    data.update({
        verifier.MODELS + "LICENSE-AGPL-3.0.txt": b"AGPL-3.0 test license",
        verifier.MODELS + "UPSTREAM-NOTICE.md": b"AGPL-3.0 Ultralytics 8.4.92 / 8.4.12",
        verifier.MODELS + "PROVENANCE.md": b"Known and missing model provenance",
        verifier.MODELS + "manifest.json": json.dumps({"bytes": chart[0], "sha256": chart[1]}).encode(),
        verifier.PACKAGE + "__version__.py": b'__version__ = "1.4.6"',
        "colorcorrectionpipeline-1.4.6.dist-info/METADATA": (
            "Metadata-Version: 2.4\nName: ColorCorrectionPipeline\nVersion: 1.4.6\nRequires-Python: >=3.8\n"
            "License: MIT for source code; bundled detector models declare AGPL-3.0.\n"
            "Provides-Extra: performance\nRequires-Dist: numba>=0.55.0; extra == 'performance'\n\n"
        ).encode(),
        "colorcorrectionpipeline-1.4.6.dist-info/licenses/LICENSE": b"MIT test license",
        "colorcorrectionpipeline-1.4.6.dist-info/licenses/ColorCorrectionPipeline/detection/models/LICENSE-AGPL-3.0.txt": b"AGPL-3.0 test license",
        "colorcorrectionpipeline-1.4.6.dist-info/licenses/ColorCorrectionPipeline/detection/models/UPSTREAM-NOTICE.md": b"AGPL-3.0 notice",
    })
    return data


def write_wheel(tmp_path, contents):
    path = tmp_path / "colorcorrectionpipeline-1.4.6-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    return path


def test_valid_wheel_checks_exact_assets_metadata_and_license(tmp_path, contents):
    report = verifier.verify_wheel(write_wheel(tmp_path, contents), "v1.4.6")
    assert report["version"] == "1.4.6" and report["numba_optional"]
    assert len(report["assets"]) == 2


@pytest.mark.parametrize("asset", list(verifier.ASSETS))
def test_missing_inference_asset_is_rejected(tmp_path, contents, asset):
    del contents[asset]
    with pytest.raises(AssertionError, match="missing inference asset"):
        verifier.verify_wheel(write_wheel(tmp_path, contents))


@pytest.mark.parametrize("asset", list(verifier.ASSETS))
def test_changed_inference_bytes_are_rejected(tmp_path, contents, asset):
    original = contents[asset]
    contents[asset] = bytes([original[0] ^ 1]) + original[1:]
    with pytest.raises(AssertionError, match="inference asset differs"):
        verifier.verify_wheel(write_wheel(tmp_path, contents))


@pytest.mark.parametrize("suffix", ["LICENSE-AGPL-3.0.txt", "UPSTREAM-NOTICE.md", "PROVENANCE.md", "manifest.json"])
def test_missing_attribution_is_rejected(tmp_path, contents, suffix):
    del contents[verifier.MODELS + suffix]
    with pytest.raises(AssertionError, match="missing detector attribution"):
        verifier.verify_wheel(write_wheel(tmp_path, contents))


@pytest.mark.parametrize("name", ["train.pt", "model.pkl", "model.pickle", "cached.pyc", "cached.nbc", "cached.nbi", "__pycache__/file", "../outside"])
def test_training_pickles_caches_and_path_traversal_are_rejected(tmp_path, contents, name):
    contents[name] = b"not part of an inference distribution"
    with pytest.raises(AssertionError):
        verifier.verify_wheel(write_wheel(tmp_path, contents))


def test_numba_cannot_silently_become_a_base_dependency(tmp_path, contents):
    metadata = "colorcorrectionpipeline-1.4.6.dist-info/METADATA"
    contents[metadata] = contents[metadata].replace(b"; extra == 'performance'", b"")
    with pytest.raises(AssertionError):
        verifier.verify_wheel(write_wheel(tmp_path, contents))


def test_wrong_release_tag_cannot_publish_this_version(tmp_path, contents):
    with pytest.raises(AssertionError):
        verifier.verify_wheel(write_wheel(tmp_path, contents), "v1.4.7")


def test_sdist_retains_build_qualification_sources(tmp_path, contents):
    metadata = contents["colorcorrectionpipeline-1.4.6.dist-info/METADATA"]
    contents = {name: value for name, value in contents.items() if ".dist-info/" not in name}
    contents.update({name: b"source" for name in ("pyproject.toml", "README.md", "LICENSE", "scripts/verify_installed.py", "scripts/verify_distribution.py")})
    contents["PKG-INFO"] = metadata
    path = tmp_path / "colorcorrectionpipeline-1.4.6.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, value in contents.items():
            member = tarfile.TarInfo("colorcorrectionpipeline-1.4.6/" + name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    assert len(verifier.verify_sdist(path, "v1.4.6")["assets"]) == 2


@pytest.mark.parametrize("name,linked", [("/outside", False), ("root/../outside", False), ("root/link", True)])
def test_sdist_rejects_unsafe_paths_and_links(tmp_path, name, linked):
    path = tmp_path / "invalid.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if linked:
            member.type, member.linkname = tarfile.SYMTYPE, "/outside"
        archive.addfile(member)
    with pytest.raises(AssertionError, match="archive"):
        verifier.verify_sdist(path)


def test_publish_is_explicit_gated_and_never_pushes_tags():
    workflow = (ROOT / ".github/workflows/publish-to-pypi.yml").read_text()
    assert "needs: qualification" in workflow
    assert "name: pypi" in workflow and "default: false" in workflow
    assert "git push" not in workflow and "git tag" not in workflow
    assert "\n  push:" not in workflow and "\n  pull_request:" not in workflow
    qualification = (ROOT / ".github/workflows/qualification.yml").read_text()
    assert "needs: [source-tests, build, installed-wheel]" in qualification
    assert "system_site_packages=False" in qualification
