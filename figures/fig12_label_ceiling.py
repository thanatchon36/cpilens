"""Figure 12: the label ceiling. Blind relabelling of the double subset: disagreement by
hierarchy level, and item-level disagreement by CPI group."""

from __future__ import annotations

import json

import polars as pl

from cpilens.viz import names, style

MIN_N = 5  # groups with fewer double-labelled products are left out of panel (b)
LEVELS = (("item", "Item"), ("group", "Group"), ("division", "Division"))


def main() -> None:
    results, out = style.dirs()
    meta = json.loads((results / "rq4_meta.json").read_text())
    a = pl.read_csv(results / "rq4_agreement.csv")
    by = pl.read_csv(results / "rq4_by_class.csv", schema_overrides={"class": pl.Utf8})
    by = by.filter((pl.col("level") == "group") & (pl.col("n") >= MIN_N)).sort("disagree")
    c = style.FAMILY["human"]
    fig, (ax, bx) = style.figure(3.2, ncols=2, width_ratios=(1, 1.6))
    lv = {r["level"]: r for r in a.iter_rows(named=True)}
    for y, (k, _) in enumerate(LEVELS[::-1]):
        r = lv[k]
        ax.plot(
            [100 * (1 - r["agree_hi"]), 100 * (1 - r["agree_lo"])],
            [y, y],
            color=c,
            lw=1.6,
            alpha=0.45,
        )
        ax.plot(100 * (1 - r["agree"]), y, "o", color=c, ms=4)
    ax.set_yticks(range(len(LEVELS)), [t for _, t in LEVELS[::-1]])
    ax.set_title("(a) By level of the CPI hierarchy")
    ax.set_xlabel("Labels changed on relabelling (%)")
    gn = names.groups()
    for y, r in enumerate(by.iter_rows(named=True)):
        bx.plot(
            [100 * r["disagree_lo"], 100 * r["disagree_hi"]], [y, y], color=c, lw=1.6, alpha=0.45
        )
        bx.plot(100 * r["disagree"], y, "o", color=c, ms=4)
    bx.set_yticks(
        range(by.height),
        [f"{gn.get(k, k)} (n={n})" for k, n in zip(by["class"], by["n"], strict=True)],
    )
    bx.set_title("(b) Item labels changed, by CPI group")
    bx.set_xlabel("Labels changed on relabelling (%)")
    for x in (ax, bx):
        x.set_xlim(left=-1)
        x.grid(axis="x")
        x.grid(axis="y", visible=False)
    item, group = 100 * (1 - lv["item"]["agree"]), 100 * (1 - lv["group"]["agree"])
    style.finish(
        fig,
        f"Relabelling blind, the labeller changes {item:.0f}% of item labels "
        f"and {group:.0f}% of group labels",
        style.source(style.SOURCE_OFF, style.SOURCE_GOLD, style.SOURCE_CALC),
        f"{meta['n_pairs']:,} products labelled twice by the same person, at least 48 hours "
        f"apart; {meta['n_unk_either']} with 'cannot tell' in either pass excluded. Groups by the "
        f"first-pass label; groups with fewer than {MIN_N} products omitted. Lines: 95% "
        "bootstrap intervals.",
    )
    style.save(fig, "fig12_label_ceiling", out)


if __name__ == "__main__":
    main()
