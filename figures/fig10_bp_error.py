"""Figure 10: year-on-year inflation error (RMSE, bp) implied by each method's gold confusion
matrix, headline / food / core, with 95% posterior intervals."""

from __future__ import annotations

import json

import polars as pl

from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.viz import names, select, style

AGGS = (("headline", "Headline"), ("food", "Food"), ("core", "Core proxy"))


def order() -> list[str]:
    cascades = [f"cascade__{c}__{m}" for m in LLMS for c in CASCADE_CHEAP]
    return [*CASCADE_CHEAP, *LLMS, *cascades]


def main() -> None:
    results, out = style.dirs()
    n_draws = json.loads((results / "rq3_meta.json").read_text())["n_draws"]
    e = pl.read_csv(results / "rq3_errors.csv").filter(
        (pl.col("variant") == "full") & (pl.col("weights") == "equal") & (pl.col("kind") == "yoy")
    )
    ms = [m for m in order() if m in set(e["method"])][::-1]
    fig, axes = style.figure(3.0, ncols=3, sharey=True)
    for ax, (agg, title) in zip(axes, AGGS, strict=True):
        d = e.filter(pl.col("aggregate") == agg)
        for y, m in enumerate(ms):
            r = d.filter(pl.col("method") == m).row(0, named=True)
            c = style.FAMILY[select.family(m)]
            ax.plot([r["rmse_bp_lo"], r["rmse_bp_hi"]], [y, y], color=c, lw=1.6, alpha=0.45)
            ax.plot(r["rmse_bp"], y, "o", color=c, ms=4)
        ax.set_title(title, fontsize=8.5, weight="semibold")
        ax.set_xlim(left=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("RMSE of year-on-year inflation error (basis points)")
    axes[0].set_yticks(range(len(ms)), [names.method(m) for m in ms])
    head = e.filter((pl.col("aggregate") == "headline") & pl.col("method").is_in(ms)).sort(
        "rmse_bp"
    )
    best = head.row(0, named=True)
    style.finish(
        fig,
        f"Lowest headline error: {names.method(best['method'])}, "
        f"{best['rmse_bp']:.0f} bp RMSE on year-on-year inflation",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_GOLD, style.SOURCE_CALC),
        f"Dots: posterior-mean confusion matrix; lines: 95% interval over {n_draws:,} "
        "Dirichlet draws. Out-of-scope items held at official values.",
    )
    style.save(fig, "fig10_bp_error", out)


if __name__ == "__main__":
    main()
