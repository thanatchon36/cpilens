"""Figure 6: per-item recall on the reference set, ranked, against cumulative CPI weight."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cpilens.viz import names, select, style


def panel(ax: Any, r: pl.DataFrame, title: str) -> None:
    d = r.filter(pl.col("weight") > 0).sort("recall", descending=True)
    lab = names.labels()
    x = np.arange(d.height)
    ax.bar(x, d["recall"], color=style.FAMILY[select.family(d["method"][0])], width=0.75)
    # recall and cumulative weight are both shares, so one axis serves both
    cum = np.cumsum(d["weight"].to_numpy()) / d["weight"].sum()
    ax.plot(x, cum, color=style.MUTED, lw=1.0, marker=".", ms=3)
    ax.set_xticks(x, [lab[k] for k in d["label"]], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.02)
    style.percent_axis(ax)
    ax.set_title(title, fontsize=8.5)


def main() -> None:
    results, out = style.dirs()
    r = pl.read_csv(results / "gold_label_recall.csv", schema_overrides={"label": pl.Utf8})
    cheap, best = select.best_cheap(results), select.best_overall(results)
    fig, axes = style.figure(4.4, ncols=2, sharey=True)
    for ax, m, tag in zip(axes, (cheap, best), ("(a)", "(b)"), strict=True):
        panel(ax, r.filter(pl.col("method") == m), f"{tag} {names.method(m)}")
    axes[0].set_ylabel("Recall (bars); cumulative CPI weight (line)")
    d = r.filter((pl.col("method") == cheap) & (pl.col("weight") > 0))
    low = d.filter(pl.col("recall") < 0.5)["weight"].sum() / d["weight"].sum()
    style.finish(
        fig,
        f"{100 * low:.0f}% of evaluated CPI weight sits in items that {names.method(cheap)} "
        "recalls less than half the time",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_REF, style.SOURCE_CALC),
        "Bars: share of each item's held-out official specifications classified correctly; "
        "grey line: cumulative evaluated CPI weight in bar order. Items without specifications "
        "are omitted.",
    )
    style.save(fig, "fig06_item_recall", out)


if __name__ == "__main__":
    main()
