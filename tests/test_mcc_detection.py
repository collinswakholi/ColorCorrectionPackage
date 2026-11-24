"""Legacy OpenCV MCC chart detection must work across OpenCV API generations."""

from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not hasattr(cv2, "mcc"), reason="opencv-contrib mcc module missing")
CHART = Path(__file__).resolve().parents[1] / "Images" / "Image_1.JPG"


def test_core_mcc_extracts_classic24():
    from ColorCorrectionPipeline.core.utils import _chart_cache, _extract_color_chart_mcc

    _chart_cache.clear() if hasattr(_chart_cache, "clear") else None
    patches, drawn, _ = _extract_color_chart_mcc(cv2.imread(str(CHART)))
    assert patches is not None and np.asarray(patches).shape == (24, 3)
    assert drawn is not None
