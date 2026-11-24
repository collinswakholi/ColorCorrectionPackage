"""Characterization tests: outputs must match the 2.2.0 baseline (see tests/golden)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import capture  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", list(capture._scenarios()))
def test_matches_baseline(name):
    record, thumb = capture.run_scenario(name)
    assert capture.compare(name, record, thumb) == []
