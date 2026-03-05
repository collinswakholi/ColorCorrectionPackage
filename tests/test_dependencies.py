"""
Critical dependency verification tests
========================================

These tests run automatically with ``pytest`` and verify that all critical
third-party modules required by ColorCorrectionPipeline are correctly
installed and functional.  They are designed to catch the most common
installation issues early — especially the opencv-python vs
opencv-contrib-python conflict that silently removes cv2.mcc.

Run standalone:
    python -m pytest tests/test_dependencies.py -v
"""

import importlib
import sys

import pytest


# ── OpenCV / MCC ──────────────────────────────────────────────────────────

class TestOpenCV:
    """Verify that opencv-contrib-python (with MCC) is installed."""

    def test_cv2_imports(self):
        import cv2
        assert cv2.__version__, "cv2 imported but __version__ is empty"

    def test_cv2_version_minimum(self):
        import cv2
        major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
        assert (major, minor) >= (4, 5), (
            f"opencv-contrib-python >= 4.5.0 required, got {cv2.__version__}"
        )

    def test_cv2_mcc_module_exists(self):
        """cv2.mcc must exist — it is only present in opencv-contrib-python."""
        import cv2
        assert hasattr(cv2, "mcc"), (
            "cv2.mcc module missing — install opencv-contrib-python "
            "(and uninstall opencv-python if present)"
        )

    def test_mcc_checker_detector_creatable(self):
        """CCheckerDetector.create() must succeed."""
        import cv2
        if not hasattr(cv2, "mcc"):
            pytest.skip("cv2.mcc not available")
        detector = None
        for factory in [
            lambda: cv2.mcc.CCheckerDetector.create(),
            lambda: cv2.mcc.CCheckerDetector_create(),
        ]:
            try:
                detector = factory()
                break
            except Exception:
                continue
        assert detector is not None, "Cannot create CCheckerDetector"

    def test_has_mcc_flag(self):
        """HAS_MCC runtime flag must be True when cv2.mcc is present."""
        from ColorCorrectionPipeline.core.utils import HAS_MCC
        import cv2
        if hasattr(cv2, "mcc"):
            assert HAS_MCC is True, "HAS_MCC should be True when cv2.mcc exists"


# ── Numba ─────────────────────────────────────────────────────────────────

class TestNumba:
    """Verify numba (core dependency since v1.4.1) is importable."""

    def test_numba_imports(self):
        import numba
        assert numba.__version__, "numba imported but __version__ is empty"

    def test_numba_njit_available(self):
        from numba import njit
        assert callable(njit)

    def test_has_numba_flag(self):
        from ColorCorrectionPipeline.core.accel import HAS_NUMBA
        assert HAS_NUMBA is True, "HAS_NUMBA should be True when numba is installed"


# ── Core scientific stack ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "colour",
    ],
)
def test_scientific_stack_imports(module_name):
    """Each core scientific library must be importable."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ── Package self-import ───────────────────────────────────────────────────

class TestPackageImport:
    """Verify that the package itself imports cleanly."""

    def test_top_level_import(self):
        import ColorCorrectionPipeline
        assert hasattr(ColorCorrectionPipeline, "__version__")

    def test_core_submodule_import(self):
        from ColorCorrectionPipeline import core
        assert hasattr(core, "HAS_MCC")
        assert hasattr(core, "to_float64")

    def test_main_classes_importable(self):
        from ColorCorrectionPipeline import ColorCorrection, Config, MyModels
        assert ColorCorrection is not None
        assert Config is not None
        assert MyModels is not None
