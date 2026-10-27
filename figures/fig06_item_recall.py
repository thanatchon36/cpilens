"""Figure 6: per-item recall on gold, ranked, against cumulative CPI weight."""

from __future__ import annotations

import numpy as np
import polars as pl

from cpilens.viz import names, select, style


def panel(ax: object, r: pl.DataFrame, title: str, right_label: bool) -> None:
    d = r.filter(pl.col("weight") > 0).sort("recall", descending=True)
    lab = names.labels()
    x = np.arange(d.height)
    ax.bar(x, d["recall"], color=style.FAMILY[select.family(d["method"][0])], width=0.75)  # type: ignore[attr-defined]
    ax.set_xticks(x, [lab[k] for k in d["label"]], rotation=90, fontsize=6.5)  # type: ignore[attr-defined]
    ax.set_ylim(0, 1.02)  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=8.5, weight="semibold")  # type: ignore[attr-defined]
    cum = np.cumsum(d["weight"].to_numpy()) / d["weight"].sum()
    ax2 = ax.twinx()  # type: ignore[attr-defined]
    ax2.plot(x, cum, color=style.INK, lw=1.0, marker=".", ms=3)
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    if right_label:
        ax2.set_ylabel("Cumulative share of in-scope CPI weight (line)", fontsize=7.5)
    else:
        ax2.set_yticklabels([])


def main() -> None:
    results, out = style.dirs()
    r = pl.read_csv(results / "gold_label_recall.csv", schema_overrides={"label": pl.Utf8})
    cheap, best = select.best_cheap(results), select.best_overall(results)
    fig, axes = style.figure(4.2, ncols=2, sharey=True)
    for i, (ax, m) in enumerate(zip(axes, (cheap, best), strict=True)):
        panel(ax, r.filter(pl.col("method") == m), names.method(m), right_label=i == 1)
    axes[0].set_ylabel("Recall on gold (Horvitz–Thompson)")
    d = r.filter((pl.col("method") == cheap) & (pl.col("weight") > 0))
    low = d.filter(pl.col("recall") < 0.5)["weight"].sum() / d["weight"].sum()
    style.finish(
        fig,
        f"{100 * low:.0f}% of in-scope CPI weight sits in items that {names.method(cheap)} "
        "recalls less than half the time",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_GOLD, style.SOURCE_CALC),
        "Bars: share of the Thai product universe of each item classified correctly; line: "
        "cumulative CPI weight in bar order. Items without gold products are omitted.",
    )
    style.save(fig, "fig06_item_recall", out)


if __name__ == "__main__":
    main()
