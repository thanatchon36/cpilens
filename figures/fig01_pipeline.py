"""Figure 1: the pipeline, from public data to basis points of measured inflation."""

from __future__ import annotations

import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cpilens.viz import style

W, H = 16.0, 6.0
BOX_W, BOX_H = 2.75, 1.25
COLS = (0.1, 3.3, 6.5, 9.7, 12.9)  # left edge of each column


def box(ax, x, y, title, body, color=style.INK, w=BOX_W, h=BOX_H):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            lw=1.0,
        )
    )
    ax.add_patch(
        FancyBboxPatch((x, y + h - 0.06), w, 0.06, boxstyle="square,pad=0", facecolor=color, lw=0)
    )
    ax.text(x + 0.1, y + h - 0.16, title, ha="left", va="top", fontsize=7, weight="semibold")
    ax.text(
        x + 0.1,
        y + h - 0.47,
        body,
        ha="left",
        va="top",
        fontsize=6,
        color=style.MUTED,
        linespacing=1.25,
    )
    return (x, y, w, h)


def arrow(ax, a, b, up=False):  # type: ignore[no-untyped-def]
    """Right side of ``a`` to left side of ``b``; with ``up``, top of ``a`` to bottom of ``b``."""
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    start, end = (ax0 + aw, ay0 + ah / 2), (bx0, by0 + bh / 2)
    if up:
        start, end = (ax0 + aw / 2, ay0 + ah), (bx0 + bw / 2, by0)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>,head_length=3,head_width=1.8",
            color=style.RULE,
            lw=0.7,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    results, out = style.dirs()
    n_items = pl.read_csv(results / "coverage.csv").filter(pl.col("label") != "OOS").height
    fig, ax = style.figure(3.3)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    f = style.FAMILY
    c0, c1, c2, c3, c4 = COLS
    off = box(ax, c0, 4.1, "Open Food Facts", "Thai and non-Thai products:\nname, brand, quantity")
    moc = box(
        ax, c0, 2.2, "Ministry of Commerce", "official Thai product\nspecifications, labelled"
    )
    tax = box(
        ax, c1, 3.0, "Taxonomy", f"{n_items} CPI items + out of scope;\nweak labels by crosswalk"
    )
    models = [
        box(ax, c2, y, t, b, f[k], h=1.05)
        for y, t, b, k in (
            (4.85, "Embeddings", "e5-large, bge-m3\nk-NN, logistic, MLP", "embedding"),
            (3.65, "Fine-tuned encoder", "XLM-R", "finetune"),
            (2.45, "LLM prompting", "Claude Haiku 5.5,\nClaude Opus 5.5", "llm"),
            (1.25, "Cascade", "cheap model; the LLM only\nwhen it is unsure", "cascade"),
        )
    ]
    ev = box(ax, c3, 3.0, "Evaluation on gold", "CPI-weighted accuracy,\ncoverage, cost; bootstrap")
    sim = box(ax, c4, 3.0, "Inflation simulator", "confusion matrix → item\nrelatives → bp error")
    gold = box(
        ax, c3, 0.3, "Gold labels", "stratified Thai sample,\nlabelled by hand, blind", f["human"]
    )
    tpso = box(ax, c4, 0.3, "TPSO", "CPI indices and weights\n(base 2566)")
    arrow(ax, off, tax)
    arrow(ax, moc, tax)
    for m in models:
        arrow(ax, tax, m)
        arrow(ax, m, ev)
    arrow(ax, ev, sim)
    arrow(ax, gold, ev, up=True)
    arrow(ax, tpso, sim, up=True)
    style.finish(
        fig,
        "From public product text to basis points of measured inflation",
        style.source("author's design"),
        "Every result in the paper is computed from public data by the code in the repository.",
    )
    style.save(fig, "fig01_pipeline", out)


if __name__ == "__main__":
    main()
