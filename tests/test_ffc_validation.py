"""Legacy FFC must reject mismatched chart/white shapes with a recorded ValueError."""

import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config


def test_mismatched_white_image_fails_cleanly():
    config = Config(
        fitting_policy="legacy-v1",
        do_ffc=True,
        do_gc=False,
        do_wb=False,
        do_cc=False,
        check_saturation=False,
    )
    pipe = ColorCorrection()
    image = np.full((40, 60, 3), 0.5)
    white = np.full((30, 60, 3), 0.9)
    _, _, failed = pipe.run(image, white, name_="x", config=config)
    assert failed
    errors = [e for e in pipe.last_run["errors"] if e["stage"] == "FFC"]
    assert errors and errors[0]["type"] == "ValueError"
    assert "same shape" in errors[0]["message"]
