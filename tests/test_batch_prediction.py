"""predict_images must equal a loop of predict_image without starving its kernels."""

import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, Config, _cpu_workers

IMAGES = Path(__file__).resolve().parents[1] / "Images"
CHART, WHITE, SCENE = (str(IMAGES / n) for n in ("Image_1.JPG", "white.JPG", "Sample_1.JPG"))


def _recording_pipeline(monkeypatch, budget):
    """An unfitted pipeline whose predict_image records the thread and kernel budget."""
    seen = []
    pipe = ColorCorrection()
    monkeypatch.setattr(_cpu_workers, "image_worker_limit", lambda: budget)

    def predict_image(self, image, show=False, *, stages=None, chunk_pixels=None):
        thread = threading.current_thread()
        seen.append(
            (thread is threading.main_thread(), getattr(_cpu_workers._budget, "limit", None))
        )
        return {"CC": image, "stages": stages}

    monkeypatch.setattr(ColorCorrection, "predict_image", predict_image)
    return pipe, seen


def test_one_worker_runs_on_the_calling_thread(monkeypatch):
    pipe, seen = _recording_pipeline(monkeypatch, budget=8)
    progress = []
    images = [np.full((2, 2, 3), i, dtype=np.float64) for i in range(3)]
    results = pipe.predict_images(
        images,
        on_progress=lambda done, total, name: progress.append((done, total, name)),
        max_workers=1,
        stages="final",
    )
    assert seen == [(True, None)] * 3
    assert [r["CC"][0, 0, 0] for r in results] == [0, 1, 2]
    assert all(r["stages"] == "final" for r in results)
    assert progress == [(1, 3, "image_0"), (2, 3, "image_1"), (3, 3, "image_2")]


@pytest.mark.parametrize("workers, budget, expected", [(2, 8, 4), (4, 8, 2), (3, 8, 2), (4, 2, 1)])
def test_image_threads_share_the_callers_kernel_budget(monkeypatch, workers, budget, expected):
    pipe, seen = _recording_pipeline(monkeypatch, budget=budget)
    images = [np.full((2, 2, 3), i, dtype=np.float64) for i in range(6)]
    results = pipe.predict_images(images, max_workers=workers)
    assert [r["CC"][0, 0, 0] for r in results] == list(range(6))  # input order
    assert all(r["stages"] is None for r in results)
    assert seen and all(not main and limit == expected for main, limit in seen)


@pytest.fixture(scope="module")
def fitted():
    pipe = ColorCorrection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, failed = pipe.run(
            CHART,
            WHITE,
            name_="chart",
            config=Config(chart={"device": "cpu"}, input={"frame_index": 0}),
        )
    assert not failed, pipe.last_run["errors"]
    return pipe


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("stages", [None, "final"])
def test_batch_matches_single_image_prediction(fitted, workers, stages):
    scenes = [SCENE, CHART, SCENE]
    expected = [fitted.predict_image(s, stages=stages) for s in scenes]
    actual = fitted.predict_images(scenes, max_workers=workers, stages=stages)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert list(got) == list(want)
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])
