"""Heavy dependencies load on first use, not on import, without changing behaviour."""

import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("colour", "matplotlib", "pandas", "sklearn")


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_pipeline_does_not_load_heavy_dependencies():
    loaded = _run("""
        import sys
        from ColorCorrectionPipeline import ColorCorrection, Config, calibrate, correct
        import ColorCorrectionPipeline.core, ColorCorrectionPipeline.io
        print(sorted(m for m in %r if m in sys.modules))
        """ % (HEAVY,))
    assert loaded == "[]"


def test_applying_a_saved_bundle_does_not_load_fitting_dependencies(tmp_path):
    bundle = tmp_path / "cal.ccpb"
    _run("""
        import warnings; warnings.simplefilter("ignore")
        from ColorCorrectionPipeline import calibrate
        calibrate("Images/Image_1.JPG", white="Images/white.JPG", device="cpu", save_to=%r)
        print("ok")
        """ % str(bundle))
    loaded = _run("""
        import sys, warnings; warnings.simplefilter("ignore")
        from ColorCorrectionPipeline import correct
        out = correct("Images/Sample_1.JPG", %r)
        assert out.shape == (768, 1152, 3)
        print(sorted(m for m in %r if m in sys.modules))
        """ % (str(bundle), HEAVY))
    assert loaded == "[]"


def test_customnn_pickled_before_the_module_split_still_loads():
    with open(ROOT / "tests" / "data" / "customnn_2_3_0.pkl", "rb") as handle:
        data = pickle.load(handle)
    model = data["model"]
    from ColorCorrectionPipeline.core import CustomNN as from_core  # noqa: N813
    from ColorCorrectionPipeline.core.correction import CustomNN

    assert type(model) is CustomNN is from_core
    assert type(model).__module__ == "ColorCorrectionPipeline.core.correction"
    # float32 network: BLAS kernels differ by an ulp between CPUs, so compare
    # to float32 precision rather than bit-for-bit across machines.
    np.testing.assert_allclose(model.predict(data["X"]), data["pred"], rtol=1e-5, atol=1e-7)
    assert type(pickle.loads(pickle.dumps(model))) is CustomNN


def test_lazy_module_proxy_forwards_and_loads_once():
    from ColorCorrectionPipeline._lazy import LazyModule

    proxy = LazyModule("json")
    assert "not loaded" in repr(proxy)
    assert proxy.dumps({"a": 1}) == '{"a": 1}'
    assert "loaded" in repr(proxy) and "not loaded" not in repr(proxy)
    assert "dumps" in dir(proxy)
    with pytest.raises(ModuleNotFoundError):
        LazyModule("no_such_module_xyz").anything
