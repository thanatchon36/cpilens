"""Figure house style: one module every figure script uses.

- Type: Libertinus Serif (SIL OFL, vendored in ``figures/fonts``), the paper's body face.
  Sizes are set so 8 pt in the figure prints at 8 pt when placed at its design width.
- Colour: Okabe-Ito (colour-blind safe), one fixed slot per method family so a family keeps its
  colour across every chart. Ordered categories (label budgets) use one blue ramp, which also
  survives greyscale printing.
- Layout: the title states the chart's message, top left. A source line, and an optional note
  line, sit under every chart. Light horizontal grid, no top/right spines.
- Export: PDF and SVG (text as outlines) for the paper, PNG at 300 dpi for the web.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

from cpilens.paths import ROOT

FONT_DIR = ROOT / "figures" / "fonts"
OUT = ROOT / "figures" / "out"
FONT = "Libertinus Serif"

INK, MUTED, RULE, GRID = "#1a1a1a", "#5a5a5a", "#3a3a3a", "#e6e6e6"
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}
FAMILY = {
    "official": INK,
    "embedding": OKABE_ITO["blue"],
    "finetune": OKABE_ITO["green"],
    "llm": OKABE_ITO["vermillion"],
    "cascade": OKABE_ITO["purple"],
    "human": OKABE_ITO["orange"],
}
BLUE_RAMP = ("#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b")

SOURCE_CPI = "TPSO consumer price indices and weights (base 2566)"
SOURCE_OFF = "Open Food Facts (ODbL)"
SOURCE_GOLD = "author's gold labels"
SOURCE_CALC = "author's calculations"


def source(*parts: str) -> str:
    return "; ".join(parts) + "."


# widths in inches: A4 with 2.5 cm margins (16 cm text block), one column of a two-column
# layout, and a 16:9 slide
WIDTH = {"full": 16.0 / 2.54, "column": 7.8 / 2.54, "slide": 10.0}
TITLE_IN = 0.34  # space reserved above the axes for the title line
LINE_IN = 0.17  # per source/note line below the axes


def setup() -> None:
    for path in sorted(FONT_DIR.glob("LibertinusSerif-*.otf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "svg.hashsalt": "cpilens",
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "font.family": FONT,
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 8.5,
            "axes.titlelocation": "left",
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.handlelength": 1.4,
            "lines.linewidth": 1.4,
            "figure.dpi": 100,
            "savefig.facecolor": "white",
        }
    )


def figure(
    height_in: float,
    size: str = "full",
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """A figure of the given width class; ``finish`` reserves the title and footer space."""
    setup()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH[size], height_in), layout="constrained", **kwargs
    )
    return fig, axes


def finish(fig: Figure, title: str, source: str, note: str | None = None) -> None:
    """Title-as-message at the top left; source and note lines (wrapped) at the bottom left."""
    import textwrap

    w, h = fig.get_figwidth(), fig.get_figheight()
    width = int(w * 22)  # characters per footer line at 7 pt
    fig.text(0.005, 1 - 0.06 / h, title, ha="left", va="top", fontsize=10, weight="semibold")
    lines = textwrap.wrap(f"Source: {source}", width)
    if note:
        lines += textwrap.wrap(f"Note: {note}", width)
    for i, line in enumerate(lines):
        fig.text(
            0.005,
            (len(lines) - i - 1) * LINE_IN / h + 0.03 / h,
            line,
            ha="left",
            va="bottom",
            fontsize=7,
            color=MUTED,
        )
    top = TITLE_IN / h
    bottom = (len(lines) * LINE_IN + 0.04) / h
    engine: Any = fig.get_layout_engine()
    engine.set(rect=(0, bottom, 1, 1 - top - bottom), w_pad=0.04, h_pad=0.04)


def dirs() -> tuple[Path, Path]:
    """``results`` and output directories: the repository defaults, or argv[1:3]."""
    import sys

    from cpilens.paths import RESULTS

    a = sys.argv[1:]
    return (Path(a[0]) if a else RESULTS), (Path(a[1]) if len(a) > 1 else OUT)


def save(fig: Figure, name: str, out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.{ext}" for ext in ("pdf", "svg", "png")]
    for p in paths:
        fig.savefig(p, dpi=300 if p.suffix == ".png" else None, metadata=_meta(p.suffix))
    plt.close(fig)
    return paths


def _meta(suffix: str) -> dict[str, str | None]:
    # fixed metadata so regenerated figures are byte-stable in Git
    if suffix == ".pdf":
        return {"CreationDate": None, "Producer": None}
    if suffix == ".svg":
        return {"Date": None}
    return {}
