"""Figure 1: the pipeline, from public data to basis points of measured inflation."""

from __future__ import annotations

import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cpilens.viz import style

W, H = 16.0, 6.0
BOX_W, BOX_H = 2.9, 1.25
COLS = (0.05, 3.28, 6.51, 9.74, 12.97)  # left edge of each column


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
    ax.add_patch(  # colour bar inset so it stays inside the rounded corners
        FancyBboxPatch(
            (x + 0.07, y + h - 0.06), w - 0.14, 0.06, boxstyle="square,pad=0", facecolor=color, lw=0
        )
    )
    ax.text(x + 0.1, y + h - 0.16, title, ha="left", va="top", fontsize=8, weight="semibold")
    ax.text(
        x + 0.1,
        y + h - 0.5,
        body,
        ha="left",
        va="top",
        fontsize=7,
        color=style.MUTED,
        linespacing=1.25,
    )
    return (x, y, w, h)


def arrow(ax, a, b, up=False, a_frac=0.5, b_frac=0.5):  # type: ignore[no-untyped-def]
    """Right side of ``a`` to left side of ``b``, at the given fraction of each box's height
    (fanned out when several arrows share an edge); with ``up``, top of ``a`` to bottom of ``b``.
    """
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    start, end = (ax0 + aw, ay0 + ah * a_frac), (bx0, by0 + bh * b_frac)
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
    fig, ax = style.figure(3.6)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    f = style.FAMILY
    c0, c1, c2, c3, c4 = COLS
    off = box(ax, c0, 4.1, "Open Food Facts", "Thai and other products:\nname, brand, quantity")
    moc = box(
        ax, c0, 2.2, "Ministry of Commerce", "official Thai product\nspecifications, labelled"
    )
    tax = box(ax, c1, 3.0, "Taxonomy", f"{n_items} CPI items and OOS;\nweak labels by crosswalk")
    models = [
        box(ax, c2, y, t, b, f[k], h=1.05)
        for y, t, b, k in (
            (4.85, "Embeddings", "e5-large, bge-m3\nk-NN, logistic, MLP", "embedding"),
            (3.65, "Fine-tuned encoder", "XLM-R", "finetune"),
            (2.45, "LLM prompting", "Claude Haiku 5.5,\nClaude Opus 5.5", "llm"),
            (1.25, "Cascade", "cheap model; the LLM only\nwhen it is unsure", "cascade"),
        )
    ]
    ev = box(ax, c3, 3.0, "Evaluation", "CPI-weighted accuracy,\ncoverage, cost; bootstrap")
    sim = box(ax, c4, 3.0, "Inflation simulator", "confusion matrix → item\nrelatives → bp error")
    ref = box(ax, c3, 0.3, "Reference set", "TPSO specifications of\nheld-out sub-items")
    tpso = box(ax, c4, 0.3, "TPSO", "CPI indices and weights\n(base 2566)")
    arrow(ax, off, tax)
    arrow(ax, moc, tax)
    for i, m in enumerate(models):  # fan the four arrows along the shared box edges
        frac = (len(models) - i) / (len(models) + 1)
        arrow(ax, tax, m, a_frac=frac)
        arrow(ax, m, ev, b_frac=frac)
    arrow(ax, ev, sim)
    arrow(ax, ref, ev, up=True)
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
