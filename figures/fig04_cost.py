"""Figure 4: cost-accuracy frontier on gold; cascades traced across thresholds."""

from __future__ import annotations

import polars as pl

from cpilens.evaluate.frontier import GPU_USD_PER_HOUR
from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.viz import names, style

POOL = "nonthai+thaiweak+mocall"
# label offsets (points) so neighbouring labels do not collide
OFFSET = {CASCADE_CHEAP[0]: (5, -10), CASCADE_CHEAP[1]: (5, 4), CASCADE_CHEAP[2]: (5, -3)}


def main() -> None:
    results, out = style.dirs()
    sc = pl.read_csv(results / "gold_scores.csv").filter(pl.col("weights") == "equal")
    sw = pl.read_csv(results / "gold_cascade_sweep.csv")
    fig, ax = style.figure(3.4)
    pts = sc.filter(
        (
            (pl.col("scenario") == POOL)
            & pl.col("family").is_in(["embedding", "finetune"])
            & (pl.col("text_variant").fill_null("full") == "full")
        )
        | (pl.col("family") == "llm")
    )
    for r in pts.iter_rows(named=True):
        c = style.FAMILY[r["family"]]
        ax.errorbar(
            r["usd_per_million"],
            r["acc_w"],
            yerr=[[r["acc_w"] - r["acc_w_lo"]], [r["acc_w_hi"] - r["acc_w"]]],
            fmt="o",
            color=c,
            ms=3.5,
            elinewidth=0.8,
            capsize=0,
        )
        if r["method"] in (*CASCADE_CHEAP, *LLMS):
            ax.annotate(
                names.method(r["method"]),
                (r["usd_per_million"], r["acc_w"]),
                xytext=OFFSET.get(r["method"], (5, -3)),
                textcoords="offset points",
                fontsize=7,
                color=c,
            )
    for m in LLMS:
        d = sw.filter((pl.col("cheap") == CASCADE_CHEAP[0]) & (pl.col("llm") == m)).sort(
            "threshold"
        )
        c = style.FAMILY["cascade"]
        ax.plot(d["usd_per_million"], d["acc_w"], color=c, lw=1.0)
        ch = d.filter(pl.col("chosen"))
        ax.plot(ch["usd_per_million"], ch["acc_w"], "D", color=c, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per million products (USD, log scale)")
    ax.set_ylabel("Weighted accuracy (full coverage)")
    ax.grid(axis="x", which="major")
    chosen = sw.filter(pl.col("chosen") & (pl.col("cheap") == CASCADE_CHEAP[0])).sort("acc_w")
    best = chosen.row(-1, named=True)
    style.finish(
        fig,
        f"A cascade routing {100 * best['routed']:.0f}% of products to "
        f"{names.LLM[best['llm']]} costs USD {best['usd_per_million']:,.0f} per million products",
        style.source(
            style.SOURCE_OFF, style.SOURCE_GOLD, "Anthropic API price list", style.SOURCE_CALC
        ),
        f"API cost measured per call; local compute at an assumed USD {GPU_USD_PER_HOUR:.2f} per "
        "GPU-hour. Line: cascade over thresholds; diamond: threshold chosen on development data.",
    )
    style.save(fig, "fig04_cost", out)


if __name__ == "__main__":
    main()
