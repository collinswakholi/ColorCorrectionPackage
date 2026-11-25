"""Synthetic chart photographs for multi-chart tests.

Renders a chart from its built-in reference values, places it on a grey table, and
"photographs" it through a simulated camera: a colour cast, a tone curve that is not
sRGB, and lens vignetting. A matching white-field photo lets FFC remove the vignette.
"""

import cv2
import numpy as np

from ColorCorrectionPipeline.chart_references import builtin_reference
from ColorCorrectionPipeline.colorimetry import ColorSemantics, linear_to_srgb, srgb_to_linear

SIZE = (1200, 900)
_SEMANTICS = ColorSemantics(version="explicit-v1", fitting_domain="linear-rgb")


def chart_card(chart, cell=64, gap=10, frame=28, frame_grey=0.12):
    """Encoded-sRGB card with the chart's patches in detector row-major order."""
    ref = builtin_reference(chart)
    rgb = _SEMANTICS.reference_to_working_rgb(ref.canonical_lab_d50, clip=True)
    rows, cols = ref.layout
    h, w = rows * cell + (rows - 1) * gap + 2 * frame, cols * cell + (cols - 1) * gap + 2 * frame
    card = np.full((h, w, 3), frame_grey, np.float32)
    for i, colour in enumerate(rgb):
        r, c = divmod(i, cols)
        y, x = frame + r * (cell + gap), frame + c * (cell + gap)
        card[y : y + cell, x : x + cell] = colour
    return ref, card


def scene(cards, angle=0.0):
    """Place one or more cards side by side on a grey table (encoded sRGB, float)."""
    canvas = np.full((SIZE[1], SIZE[0], 3), 0.55, np.float32)
    n = len(cards)
    for k, card in enumerate(cards):
        h, w = card.shape[:2]
        slot_w = SIZE[0] / n
        scale = min(slot_w * 0.8 / w, SIZE[1] * 0.7 / h)
        cx, cy = slot_w * (k + 0.5), SIZE[1] / 2
        half = np.float32([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2 * scale
        t = np.deg2rad(angle)
        rot = np.float32([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        dst = half @ rot.T + [cx, cy]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        m = cv2.getPerspectiveTransform(src, np.float32(dst))
        warped = cv2.warpPerspective(card, m, SIZE, flags=cv2.INTER_AREA, borderValue=(-1, -1, -1))
        mask = warped[..., 0] >= 0
        canvas[mask] = warped[mask]
    return canvas


def camera(encoded, gains=(1.12, 1.0, 0.82), tone=1.18, vignette=0.35):
    """Photograph encoded sRGB through a cast, a non-sRGB tone curve and vignetting."""
    linear = srgb_to_linear(np.clip(encoded, 0, 1)) * np.asarray(gains, np.float32)
    h, w = linear.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    linear = linear * (1 - vignette * r2 / 2)[..., None]
    recorded = np.clip(linear, 0, 1) ** tone
    return np.clip(linear_to_srgb(recorded) * 255 + 0.5, 0, 255).astype(np.uint8)


def white_field(gains=(1.12, 1.0, 0.82), tone=1.18, vignette=0.35, level=0.85):
    flat = np.full((SIZE[1], SIZE[0], 3), level, np.float32)
    return camera(flat, gains, tone, vignette)


def write(path, rgb8):
    cv2.imwrite(str(path), rgb8[..., ::-1])
    return str(path)
