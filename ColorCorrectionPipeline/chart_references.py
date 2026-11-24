"""Built-in references for the three charts the detector recognises.

Values are CIE Lab D50/2 degrees in the detector's row-major cell order (row 1 is the
assembled top of the chart), copied from ColorCorrect Studio's audited catalog
(ColorCorrectionPackage_UI ``app/ColorCorrect.Engine/ReferenceTemplates.json``, which
matches its Rust ``ccp_core/src/refs.rs``). Grey patches for GC and WB are Studio's
named neutrals (``refs/neutrals.json``); they are never chosen by low chroma.

Sources (not manufacturer-certified values):

* Classic 24: the package's existing UI-native table (``ui_native.default_reference``).
* SpyderCHECKR 24: darktable's "after 2018" V2 table, checked against Datacolor's
  published chart. Datacolor row D is the assembled top row, so the white-to-black
  ramp A1..A6 is the bottom row.
* ColorChecker Digital SG: BabelColor "ColorCheckerSG After Nov 2014". Border
  repeats (columns A and N, rows 1 and 10) are evenness patches, not the grey scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

__all__ = ["ChartSpec", "CHARTS", "chart_spec", "builtin_reference", "is_builtin_reference"]


@dataclass(frozen=True)
class ChartSpec:
    """One supported chart: detector template, layout, names, values and neutrals."""

    template_id: str
    label: str
    rows: int
    cols: int
    profile_id: str
    names: Tuple[str, ...]
    lab_d50: Tuple[Tuple[float, float, float], ...]
    neutral_ids: Tuple[str, ...]
    source: str

    @property
    def cell_ids(self) -> Tuple[str, ...]:
        return tuple("r%dc%d" % (r + 1, c + 1) for r in range(self.rows) for c in range(self.cols))


SPYDER24_NAMES = (
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
)

SPYDER24_LAB_D50 = (
    (70.19, -31.85, 1.98),
    (54.38, 8.84, -25.71),
    (42.03, -15.78, 22.93),
    (48.82, -5.11, -23.08),
    (65.1, 18.14, 18.68),
    (36.13, 14.15, 15.78),
    (60.94, 38.21, 61.31),
    (37.8, 7.3, -43.04),
    (49.81, 48.5, 15.76),
    (28.88, 19.36, -24.48),
    (72.45, -23.57, 60.47),
    (71.65, 23.74, 72.28),
    (47.12, -32.5, -28.75),
    (50.49, 53.45, -13.55),
    (83.61, 3.36, 87.02),
    (41.05, 60.75, 31.17),
    (54.14, -40.8, 34.75),
    (24.75, 13.78, -49.48),
    (96.04, 2.16, 2.6),
    (80.44, 1.17, 2.05),
    (65.52, 0.69, 1.86),
    (49.62, 0.58, 1.56),
    (33.55, 0.35, 1.4),
    (16.91, 1.43, -0.81),
)

DIGITAL_SG_NAMES = tuple(
    "%s%d" % (chr(ord("A") + col), row) for row in range(1, 11) for col in range(14)
)

DIGITAL_SG_LAB_D50 = (
    (96.71, -0.62, 2.06),
    (49.7, 0.12, 0.68),
    (8.13, 0.15, -0.76),
    (96.79, -0.66, 1.99),
    (49.79, 0.13, 0.66),
    (8.09, 0.19, -0.69),
    (96.7, -0.66, 1.97),
    (49.74, 0.14, 0.68),
    (8.1, 0.19, -0.93),
    (96.69, -0.67, 1.95),
    (49.72, 0.12, 0.64),
    (8.08, 0.13, -0.81),
    (49.71, 0.12, 0.62),
    (96.79, -0.67, 1.97),
    (8.05, 0.17, -0.69),
    (33.02, 52.0, -10.3),
    (19.65, 20.42, -18.82),
    (84.0, -1.7, -8.37),
    (32.77, 19.91, 22.33),
    (63.88, 20.34, 19.93),
    (45.84, -3.74, -25.32),
    (38.29, -17.44, 30.22),
    (51.36, 9.52, -26.98),
    (68.71, -35.41, -1.11),
    (85.68, 10.75, 18.39),
    (23.03, 33.95, 8.88),
    (42.52, 63.55, 11.43),
    (49.78, 0.12, 0.65),
    (49.76, 0.11, 0.72),
    (61.4, 27.14, -18.42),
    (41.7, 18.9, -37.42),
    (85.48, 15.15, 0.79),
    (62.28, 37.56, 68.87),
    (35.28, 12.93, -51.17),
    (47.6, 53.66, 22.15),
    (20.76, 31.66, -28.04),
    (71.62, -24.77, 64.1),
    (70.39, 19.37, 79.73),
    (89.35, -16.38, 6.41),
    (44.35, 67.94, 50.62),
    (18.09, 32.61, -5.9),
    (8.23, 0.18, -0.82),
    (96.72, -0.63, 2.06),
    (30.54, 50.39, -41.79),
    (20.25, 0.26, -36.44),
    (84.56, -19.74, -1.13),
    (19.92, 25.07, -61.05),
    (52.75, -44.12, 38.68),
    (36.88, 65.72, 41.63),
    (81.43, 2.41, 88.98),
    (48.75, 57.24, -14.45),
    (47.42, -30.91, -32.27),
    (84.59, 5.21, -5.87),
    (60.91, 36.55, 4.15),
    (40.66, 65.54, 31.98),
    (96.73, -0.67, 1.99),
    (8.17, 0.15, -0.65),
    (49.56, -13.9, -49.65),
    (60.13, -17.88, -32.08),
    (85.26, 13.37, 7.95),
    (96.78, -0.66, 2.01),
    (79.65, -0.08, 0.62),
    (65.22, -0.27, 0.16),
    (49.71, 0.12, 0.69),
    (34.85, -0.21, 0.73),
    (15.43, -0.24, -0.25),
    (83.63, -12.47, -8.89),
    (62.2, 37.45, 18.18),
    (53.13, 68.44, 49.57),
    (49.8, 0.11, 0.67),
    (49.68, 0.14, 0.74),
    (60.62, -29.91, -27.54),
    (19.75, -17.79, -22.37),
    (84.38, -11.97, 27.16),
    (8.07, 0.12, -0.93),
    (30.32, -0.1, 0.22),
    (39.55, -0.37, -0.09),
    (60.04, 0.09, 0.05),
    (75.36, 0.35, 0.26),
    (88.85, -0.59, 0.25),
    (70.6, -0.24, 0.07),
    (63.33, 51.3, 81.88),
    (82.08, 23.39, 87.24),
    (8.18, 0.15, -0.84),
    (96.6, -0.62, 2.11),
    (20.13, -24.81, -7.5),
    (60.43, -5.12, -32.79),
    (62.35, 29.94, 36.89),
    (77.37, 20.28, 24.27),
    (63.46, 13.53, 26.37),
    (44.49, 16.06, 26.79),
    (67.6, 14.47, 17.12),
    (45.14, 26.38, 41.24),
    (64.0, 25.09, 27.14),
    (45.14, -0.04, 0.86),
    (73.74, -11.45, 85.07),
    (82.5, 5.29, 96.68),
    (96.73, -0.65, 2.01),
    (7.99, 0.21, -0.75),
    (60.32, -40.29, -13.25),
    (50.46, -47.9, -11.56),
    (64.17, 21.34, 19.36),
    (74.01, 29.0, 25.8),
    (64.44, 14.31, 17.63),
    (64.97, 15.89, 16.79),
    (64.75, 17.3, 18.88),
    (36.2, 16.7, 27.06),
    (66.65, 22.21, 28.81),
    (20.33, 0.4, -0.21),
    (62.35, 1.96, 57.52),
    (71.9, -17.32, 77.72),
    (49.75, 0.13, 0.67),
    (49.67, 0.15, 0.73),
    (19.62, 1.77, 11.99),
    (60.53, -40.75, 19.37),
    (50.48, -53.21, 12.65),
    (20.33, -23.98, 7.2),
    (60.05, -44.0, 7.27),
    (60.77, -30.19, 40.76),
    (51.26, -50.65, 43.8),
    (61.65, -54.33, 46.18),
    (62.05, 16.45, 51.74),
    (62.33, -14.54, 54.58),
    (72.77, -29.09, 71.26),
    (21.95, 13.41, 16.36),
    (8.11, 0.15, -0.9),
    (96.51, -0.63, 2.11),
    (49.68, 0.15, 0.78),
    (8.09, 0.19, -0.69),
    (96.57, -0.64, 2.0),
    (49.72, 0.14, 0.71),
    (8.08, 0.18, -0.78),
    (96.71, -0.64, 2.01),
    (49.76, 0.14, 0.71),
    (7.97, 0.14, -0.8),
    (96.71, -0.64, 2.02),
    (49.74, 0.14, 0.69),
    (8.13, 0.15, -0.86),
    (49.74, 0.12, 0.69),
    (96.55, -0.64, 2.02),
)

CHARTS: Dict[str, ChartSpec] = {
    "spyder_4x6": ChartSpec(
        template_id="spyder_4x6",
        label="SpyderCHECKR 24",
        rows=4,
        cols=6,
        profile_id="ui-native-spyder24",
        names=SPYDER24_NAMES,
        lab_d50=SPYDER24_LAB_D50,
        neutral_ids=("r4c1", "r4c2", "r4c3", "r4c4", "r4c5", "r4c6"),
        source="darktable SpyderCHECKR 24 after-2018 V2 (CIE Lab D50/2deg), via ColorCorrect Studio",
    ),
    "digital_sg_10x14": ChartSpec(
        template_id="digital_sg_10x14",
        label="ColorChecker Digital SG",
        rows=10,
        cols=14,
        profile_id="ui-native-digital-sg140",
        names=DIGITAL_SG_NAMES,
        lab_d50=DIGITAL_SG_LAB_D50,
        neutral_ids=(
            "r5c5",
            "r5c6",
            "r5c7",
            "r5c8",
            "r5c9",
            "r5c10",
            "r6c5",
            "r6c6",
            "r6c7",
            "r6c8",
            "r6c9",
            "r6c10",
        ),
        source="BabelColor ColorCheckerSG After Nov 2014 (CIE Lab D50/2deg), via ColorCorrect Studio",
    ),
}

#: Detector chart types, in the order "auto" prefers them when a photo holds several.
TEMPLATES = ("classic_4x6", "spyder_4x6", "digital_sg_10x14")

_ALIASES = {
    "classic": "classic_4x6",
    "classic24": "classic_4x6",
    "passport": "classic_4x6",
    "classic_4x6": "classic_4x6",
    "spyder": "spyder_4x6",
    "spyder24": "spyder_4x6",
    "spyder_4x6": "spyder_4x6",
    "digital-sg": "digital_sg_10x14",
    "digital_sg": "digital_sg_10x14",
    "sg": "digital_sg_10x14",
    "digital_sg_10x14": "digital_sg_10x14",
}

#: Values accepted by ``chart={"type": ...}``.
CHART_TYPES = ("auto",) + tuple(sorted(_ALIASES))

_BUILTIN_PROFILE_IDS = {
    "classic_4x6": "ui-native-classic24",
    "spyder_4x6": "ui-native-spyder24",
    "digital_sg_10x14": "ui-native-digital-sg140",
}


def template_for(chart_type: str) -> str:
    """Resolve a chart type name or alias to its detector template id."""
    key = str(chart_type).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            "chart type must be auto, classic, spyder or digital-sg (got %r)" % (chart_type,)
        )
    return _ALIASES[key]


def chart_spec(chart_type: str) -> ChartSpec:
    """The Spyder or Digital SG specification (Classic lives in ``ui_native``)."""
    template = template_for(chart_type)
    if template not in CHARTS:
        raise ValueError("chart_spec covers spyder and digital-sg; use builtin_reference")
    return CHARTS[template]


def builtin_reference(chart_type: str = "classic"):
    """The built-in ``ReferenceProfile`` for a chart type.

    Classic returns exactly ``ui_native.default_reference()``, so Classic fits are
    unchanged. Spyder and Digital SG are built the same way from Studio's catalog.
    """
    template = template_for(chart_type)
    from .ui_native import D50, default_reference

    if template == "classic_4x6":
        return default_reference()
    from .contracts import ReferenceProfile

    spec = CHARTS[template]
    return ReferenceProfile.from_values(
        spec.lab_d50,
        color_space="lab",
        white=D50,
        scale=1,
        profile_id=spec.profile_id,
        revision=1,
        names=spec.names,
        cell_ids=spec.cell_ids,
        neutral_ids=spec.neutral_ids,
        retain_source=True,
        template_id=spec.template_id,
        layout=(spec.rows, spec.cols),
        name=spec.label,
        provenance={"source": spec.source},
    )


def is_builtin_reference(profile) -> bool:
    """True when ``profile`` is one of the built-in references, value for value."""
    if profile is None:
        return True
    template = str(getattr(profile, "template_id", ""))
    if _BUILTIN_PROFILE_IDS.get(template) != getattr(profile, "profile_id", None):
        return False
    return bool(profile.digest == builtin_reference(template).digest)


# -- user-supplied reference values (same contract as the Studio SDK) ---------

#: CIE standard whites (2 degree xy) accepted by name, as in the Studio SDK.
WHITE_POINTS = {
    "D50": (0.34567, 0.35850),
    "D55": (0.33242, 0.34743),
    "D60": (0.32168, 0.33767),
    "D65": (0.31271, 0.32902),
    "D75": (0.29902, 0.31485),
    "A": (0.44757, 0.40745),
    "C": (0.31006, 0.31616),
    "E": (1.0 / 3.0, 1.0 / 3.0),
}


def _white(white):
    if isinstance(white, str):
        key = white.strip().upper()
        if key in ("D50", "D65"):
            return key  # the converter's own named whites
        if key not in WHITE_POINTS:
            raise ValueError("white must be one of %s or an xy pair" % ", ".join(WHITE_POINTS))
        return WHITE_POINTS[key]
    return white


def _layout(chart):
    """(template, rows, cols, names, neutral_ids) for a chart name."""
    template = template_for(chart)
    if template == "classic_4x6":
        from .ui_native import default_reference

        base = default_reference()
        return template, 4, 6, tuple(base.names), tuple(base.neutral_ids)
    spec = CHARTS[template]
    return template, spec.rows, spec.cols, spec.names, spec.neutral_ids


def _cell_for(item, names, cell_ids, chart):
    """A cell id (``r4c1``) or a patch name: exact, case-insensitive, then unique prefix."""
    if item in cell_ids:
        return item
    text = str(item).strip().lower()
    for match in (
        [i for i, n in enumerate(names) if n == item],
        [i for i, n in enumerate(names) if n.lower() == text],
        [i for i, n in enumerate(names) if n.lower().startswith(text)],
    ):
        if len(match) == 1:
            return cell_ids[match[0]]
    raise ValueError("unknown or ambiguous neutral cell %r for %s" % (item, chart))


def custom_reference(
    values,
    *,
    chart="classic",
    color_space="lab",
    white="D50",
    scale=1,
    order="row-major",
    name="Custom reference",
    neutral_ids=None,
    profile_id=None,
    revision=1,
):
    """Build a reference from your own measured values for one of the three charts.

    Args:
        values: One triplet per chart cell: 24 for Classic/Spyder, 140 for Digital SG.
        chart: ``classic`` (or ``passport``), ``spyder`` or ``digital-sg``.
        color_space: ``lab``, ``xyz``, ``srgb`` or ``linear-srgb``.
        white: The values' white point: a name (D50, D55, D60, D65, D75, A, C, E) or xy.
            sRGB values use D65.
        scale: Lab 1; XYZ 1 or 100; RGB 1, 255 or 65535.
        order: ``row-major`` (left to right, then the next row; the default) or
            ``column-major`` (top to bottom, then the next column), as the chart
            appears with its detector orientation.
        neutral_ids: Grey cells for GC and WB, as cell ids (``r4c1``) or chart names
            (``A1``, ``white``). Defaults to the chart's built-in grey patches.

    Returns:
        A ``ReferenceProfile`` to pass as ``Config(reference=...)``.
    """
    from .contracts import ReferenceProfile

    template, rows, cols, names, default_neutrals = _layout(chart)
    if order not in ("row-major", "column-major"):
        raise ValueError("order must be row-major or column-major")
    cells = [tuple(row) for row in values]
    if len(cells) != rows * cols:
        raise ValueError(
            "%s needs %d values (%d rows x %d columns); got %d"
            % (chart, rows * cols, rows, cols, len(cells))
        )
    if order == "column-major":
        cells = [cells[col * rows + row] for row in range(rows) for col in range(cols)]
    cell_ids = tuple("r%dc%d" % (r + 1, c + 1) for r in range(rows) for c in range(cols))
    neutrals = [
        _cell_for(item, names, cell_ids, chart)
        for item in (default_neutrals if neutral_ids is None else neutral_ids)
    ]
    return ReferenceProfile.from_values(
        cells,
        color_space=color_space,
        white=_white(white),
        scale=scale,
        profile_id=profile_id or "custom-" + template.replace("_", "-"),
        revision=revision,
        names=names,
        cell_ids=cell_ids,
        neutral_ids=tuple(neutrals),
        retain_source=True,
        template_id=template,
        layout=(rows, cols),
        name=name,
    )


def reference_from_text(text, *, chart="classic", order="row-major", **options):
    """Parse a reference table pasted from a spreadsheet or read from a CSV.

    Accepted rows (comma, tab or semicolon separated, with one optional header):
    three values per cell in ``order``; ``cell,v1,v2,v3`` with a cell such as
    ``A1`` (letter = column, number = row); or 1-based ``row,column,v1,v2,v3``.
    Coordinate rows may come in any order but must cover every cell once.
    Other keywords go to :func:`custom_reference`.
    """
    import csv
    import io

    _, rows, cols, _, _ = _layout(chart)
    dialect: Any
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    table = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not table:
        raise ValueError("reference table is empty")
    try:
        float(table[0][-1])
    except ValueError:
        table = table[1:]  # header row
    widths = {len(r) for r in table}
    if len(widths) != 1 or widths.pop() not in (3, 4, 5):
        raise ValueError("expected three values per row, or a cell/row,column plus three values")
    width = len(table[0])
    if width == 3:
        return custom_reference(
            [[float(v) for v in r] for r in table], chart=chart, order=order, **options
        )
    placed = {}
    for r in table:
        if width == 5:
            row, col = int(r[0]), int(r[1])
        else:
            cell = r[0].strip().upper()
            if len(cell) < 2 or not "A" <= cell[0] <= "Z":
                raise ValueError("invalid cell coordinate %r" % r[0])
            row, col = int(cell[1:]), ord(cell[0]) - ord("A") + 1
        if not (1 <= row <= rows and 1 <= col <= cols) or (row, col) in placed:
            raise ValueError("duplicate or out-of-range cell (row %d, column %d)" % (row, col))
        placed[row, col] = [float(v) for v in r[-3:]]
    if len(placed) != rows * cols:
        raise ValueError("reference table is missing chart cells")
    ordered = [placed[r, c] for r in range(1, rows + 1) for c in range(1, cols + 1)]
    return custom_reference(ordered, chart=chart, order="row-major", **options)


def reference_from_csv(path, *, encoding="utf-8-sig", **options):
    """Read :func:`reference_from_text` input from a file."""
    from pathlib import Path

    text = Path(path).read_text(encoding=encoding)
    options.setdefault("name", Path(path).stem)
    return reference_from_text(text, **options)
