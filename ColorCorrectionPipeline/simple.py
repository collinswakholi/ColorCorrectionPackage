"""Two-call API for the common case: calibrate on a chart, then correct images.

    >>> from ColorCorrectionPipeline import calibrate, correct
    >>> cal = calibrate("chart.jpg", white="white.jpg", save_to="calibration.ccpb")
    >>> cal.metrics["chart_CC"]["delta_e00_after_mean"]   # doctest: +SKIP
    >>> corrected = correct("scene.jpg", cal)              # float RGB, 0-1
    >>> corrected = correct("scene.jpg", "calibration.ccpb")

Everything here is a thin layer over :class:`Config` and :class:`ColorCorrection`; use
those directly for full control. ``calibrate`` defaults to ``frame_index=0`` because many
camera JPEGs carry a second (preview) frame, which the strict v2 reader otherwise rejects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union, cast

import numpy as np

__all__ = ["Calibration", "calibrate", "correct", "PRESETS"]

from .config import PRESETS, Config

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class Calibration:
    """A fitted pipeline plus the metrics of the fit that produced it."""

    pipeline: Any
    metrics: Dict[str, Any]

    def correct(self, image: Union[PathLike, np.ndarray], *, stage: str = "final") -> np.ndarray:
        """Correct one image; see :func:`correct`."""
        return correct(image, self, stage=stage)

    def save(self, path: PathLike) -> str:
        """Write a portable ``.ccpb`` bundle (no pickle) and return its path."""
        self.pipeline.save_bundle(path)
        return os.fspath(path)

    @property
    def delta_e(self) -> Optional[float]:
        """Mean CIEDE2000 of the chart after correction, when the fit reported it."""
        for stage in ("CC", "WB", "GC", "FFC"):
            for name, values in self.metrics.items():
                if name.endswith("_" + stage) and isinstance(values, Mapping):
                    value = values.get("delta_e00_after_mean")
                    if value is not None:
                        return float(value)
        return None


def calibrate(
    chart: Union[PathLike, np.ndarray],
    white: Union[PathLike, np.ndarray, None] = None,
    *,
    model: str = "rpcc2",
    frame_index: Optional[int] = 0,
    device: str = "auto",
    save_to: Optional[PathLike] = None,
    config: Any = None,
    name: str = "chart",
    **options: Any,
) -> Calibration:
    """Fit a calibration on a ColorChecker Classic 24 capture.

    Args:
        chart: Chart photograph (path or RGB array).
        white: Matching flat-field (white) capture. Omit to skip flat-field correction.
        model: Preset name: ``rpcc2`` (default), ``rpcc3``, ``ccm``/``3x3``, ``pls`` or
            ``high-precision``.
        frame_index: Frame to read from multi-frame files; ``None`` requires single-frame input.
        device: Chart detection device, ``auto`` or ``cpu``.
        save_to: Optional ``.ccpb`` path; the bundle is written after a successful fit.
        config: A ready :class:`Config`; when given, ``model``, ``frame_index``, ``device`` and
            ``options`` are not used.
        name: Label used in metric keys and saved artifacts.
        **options: Further :class:`Config` keyword arguments (e.g. ``numerics={...}``).

    Returns:
        :class:`Calibration` with the fitted pipeline and fit metrics.

    Raises:
        CalibrationError: an enabled stage failed (for example, no chart was found).
    """
    from .pipeline import ColorCorrection

    if config is None:
        config = Config.from_preset(
            model, white=white is not None, frame_index=frame_index, device=device, **options
        )
    elif options:
        raise TypeError("pass either config or Config keyword options, not both")
    pipeline = ColorCorrection()
    result = pipeline.fit_or_raise(chart, white, name_=name, config=config)
    calibration = Calibration(pipeline=pipeline, metrics=result.metrics)
    if save_to is not None:
        calibration.save(save_to)
    return calibration


def correct(
    image: Union[PathLike, np.ndarray],
    calibration: Union[Calibration, PathLike, Any],
    *,
    stage: str = "final",
) -> np.ndarray:
    """Apply a calibration to one image and return float RGB in 0-1.

    ``calibration`` may be a :class:`Calibration`, a fitted :class:`ColorCorrection`, or a
    path to a ``.ccpb`` bundle. ``stage`` selects ``final`` (default) or one stage name
    (``FFC``, ``GC``, ``WB``, ``CC``).
    """
    from .pipeline import ColorCorrection

    if isinstance(calibration, Calibration):
        pipeline = calibration.pipeline
    elif isinstance(calibration, ColorCorrection):
        pipeline = calibration
    elif isinstance(calibration, (str, os.PathLike)):
        pipeline = ColorCorrection().load_bundle(calibration)
    else:
        raise TypeError("calibration must be a Calibration, a ColorCorrection or a .ccpb path")
    outputs = pipeline.predict_image(image, stages=stage)
    key = "CC" if stage == "final" else stage
    return cast(np.ndarray, outputs[key])
