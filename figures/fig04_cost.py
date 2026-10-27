"""Figure 4: cost-accuracy frontier on gold; cascades traced across thresholds."""

from __future__ import annotations

import polars as pl
from matplotlib.ticker import FuncFormatter, NullFormatter

from cpilens.evaluate.frontier import GPU_USD_PER_HOUR
from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.viz import names, style

POOL = "nonthai+thaiweak+mocall"
NAMED = (*CASCADE_CHEAP, *LLMS)  # in the legend, with their own markers; the rest are hollow dots
MARKER = dict(zip(NAMED, ("o", "s", "^", "o", "s"), strict=True))
LLM_LS = dict(zip(LLMS, ("-", (0, (3, 1.5))), strict=True))


def main() -> None:
    results, out = style.dirs()
    sc = pl.read_csv(results / "gold_scores.csv").filter(pl.col("weights") == "equal")
    sw = pl.read_csv(results / "gold_cascade_sweep.csv")
    fig, ax = style.figure(3.8)
    pts = sc.filter(
        (
            (pl.col("scenario") == POOL)
            & pl.col("family").is_in(["embedding", "finetune"])
            & (pl.col("text_variant").fill_null("full") == "full")
        )
        | (pl.col("family") == "llm")
    )
    n_other = 0
    for r in pts.iter_rows(named=True):
        c = style.FAMILY[r["family"]]
        named = r["method"] in NAMED
        n_other += not named
        ax.errorbar(
            r["usd_per_million"],
            r["acc_w"],
            yerr=[[r["acc_w"] - r["acc_w_lo"]], [r["acc_w_hi"] - r["acc_w"]]],
            fmt=MARKER.get(r["method"], "o"),
            color=c,
            mfc=c if named else "white",
            ms=4.5 if named else 3,
            elinewidth=0.8,
            capsize=0,
            alpha=1.0 if named else 0.6,
            zorder=3 if named else 2,
            label=names.method(r["method"]) if named else "_nolegend_",
        )
    if n_other:
        ax.plot(
            [],
            [],
            "o",
            color=style.FAMILY["embedding"],
            mfc="white",
            ms=3,
            alpha=0.6,
            label=f"Other embedding classifiers ({n_other})",
        )
    c = style.FAMILY["cascade"]
    for m in LLMS:
        d = sw.filter((pl.col("cheap") == CASCADE_CHEAP[0]) & (pl.col("llm") == m)).sort(
            "threshold"
        )
        ax.plot(
            d["usd_per_million"],
            d["acc_w"],
            color=c,
            lw=1.0,
            ls=LLM_LS[m],
            label=names.method(f"cascade__{CASCADE_CHEAP[0]}__{m}"),
        )
        ch = d.filter(pl.col("chosen"))
        ax.plot(ch["usd_per_million"], ch["acc_w"], "D", color=c, ms=4, label="_nolegend_")
    ax.plot([], [], "D", color=c, ms=4, lw=0, label="Threshold chosen on development data")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Cost per million products (USD, log scale)")
    ax.set_ylabel(style.ACC_LABEL)
    style.percent_axis(ax)
    ax.grid(axis="x", which="major")
    chosen = sw.filter(pl.col("chosen") & (pl.col("cheap") == CASCADE_CHEAP[0])).sort("acc_w")
    best = chosen.row(-1, named=True)
    style.finish(
        fig,
        f"A cascade routing {100 * best['routed']:.0f}% of products to "
        f"{names.LLM[best['llm']]} costs USD {best['usd_per_million']:,.0f} per million products",
        style.source(
            style.SOURCE_OFF, style.SOURCE_REF, "Anthropic API price list", style.SOURCE_CALC
        ),
        f"Full coverage throughout. API cost measured per call; local compute at an assumed USD "
        f"{GPU_USD_PER_HOUR:.2f} per GPU-hour. Lines: the cascade over its thresholds.",
        legend=ax,
        legend_cols=3,
    )
    style.save(fig, "fig04_cost", out)


if __name__ == "__main__":
    main()
