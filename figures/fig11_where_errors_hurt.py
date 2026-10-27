"""Figure 11: the confusions that move measured headline inflation most (best cheap method)."""

from __future__ import annotations

import polars as pl
from matplotlib.patches import Patch

from cpilens.taxonomy.weights import PRICE_REF_DATE
from cpilens.viz import names, select, style

TOP = 12
# flow types are ordered by how far a product travels, so the ordinal blue ramp encodes them
# (the method-family colours would suggest methods)
TYPE_COLOR = {
    "within_group": style.BLUE_RAMP[1],
    "within_division": style.BLUE_RAMP[2],
    "cross_division": style.BLUE_RAMP[4],
    "oos_leak": style.BLUE_RAMP[5],
    "empty_item": style.MUTED,
}
TYPE_NAME = {
    "within_group": "within group",
    "within_division": "within division",
    "cross_division": "across divisions",
    "oos_leak": "out-of-scope leak",
    "empty_item": "empty item",
}


def main() -> None:
    results, out = style.dirs()
    m = select.best_cheap(results)
    d = pl.read_csv(
        results / "rq3_decomposition.csv", schema_overrides={"true": pl.Utf8, "pred": pl.Utf8}
    )
    d = d.filter((pl.col("method") == m) & (pl.col("aggregate") == "headline"))
    top = d.sort("mean_abs_bp", descending=True).head(TOP)[::-1]
    lab = names.labels()
    fig, ax = style.figure(3.6)
    y = range(top.height)
    ax.barh(y, top["mean_bp"], color=[TYPE_COLOR[t] for t in top["flow_type"]], height=0.7)
    ax.set_yticks(
        y,
        [f"{lab.get(t, t)} → {lab[p]}" for t, p in zip(top["true"], top["pred"], strict=True)],
        fontsize=7,
    )
    ax.axvline(0, color=style.RULE, lw=0.6)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        f"Mean contribution to the headline index error since {PRICE_REF_DATE:%b %Y} (basis points)"
    )
    handles = [
        Patch(color=TYPE_COLOR[t], label=TYPE_NAME[t])
        for t in TYPE_COLOR
        if t in set(top["flow_type"])
    ]
    ft = pl.read_csv(results / "rq3_flow_types.csv").filter(
        (pl.col("method") == m) & (pl.col("aggregate") == "headline")
    )
    share = top["mean_abs_bp"].sum() / ft["mean_abs_bp"].sum()
    style.finish(
        fig,
        f"For {names.method(m)}, {TOP} confusions carry {100 * share:.0f}% of the headline error",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_REF, style.SOURCE_CALC),
        "Contribution = item weight × misallocated product share × gap between the true and the "
        "predicted item's price relative (true → predicted). Posterior-mean confusion matrix "
        "on the held-out official specifications; products in proportion to CPI weights.",
        legend=(handles, [h.get_label() for h in handles]),
        legend_cols=5,
    )
    style.save(fig, "fig11_where_errors_hurt", out)


if __name__ == "__main__":
    main()
