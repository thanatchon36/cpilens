"""Figure 10: year-on-year inflation error (RMSE, bp) implied by each method's reference-set
confusion matrix, headline / food / core, with two 95% intervals: Dirichlet draws of the
confusion matrix (specifications independent) and resampled sub-items (errors clustered)."""

from __future__ import annotations

import json

import polars as pl
from matplotlib.lines import Line2D

from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.viz import names, select, style

AGGS = (("headline", "(a) Headline"), ("food", "(b) Food"), ("core", "(c) Core proxy"))


def order() -> list[str]:
    cascades = [f"cascade__{c}__{m}" for m in LLMS for c in CASCADE_CHEAP]
    return [*CASCADE_CHEAP, *LLMS, *cascades]


def main() -> None:
    results, out = style.dirs()
    n_draws = json.loads((results / "rq3_meta.json").read_text())["n_draws"]
    allv = pl.read_csv(results / "rq3_errors.csv").filter(
        (pl.col("weights") == "equal") & (pl.col("universe") == "cpi") & (pl.col("kind") == "yoy")
    )
    e = allv.filter(pl.col("variant") == "full")
    cl = allv.filter(pl.col("variant") == "cluster")
    ms = [m for m in order() if m in set(e["method"])][::-1]
    fig, axes = style.figure(3.6, ncols=3, sharey=True)
    for ax, (agg, title) in zip(axes, AGGS, strict=True):
        d = e.filter(pl.col("aggregate") == agg)
        dc = cl.filter(pl.col("aggregate") == agg)
        for y, m in enumerate(ms):
            r = d.filter(pl.col("method") == m).row(0, named=True)
            rc = dc.filter(pl.col("method") == m).row(0, named=True)
            c = style.FAMILY[select.family(m)]
            ax.plot([rc["rmse_bp_lo"], rc["rmse_bp_hi"]], [y, y], color=c, lw=0.8)
            ax.plot([r["rmse_bp_lo"], r["rmse_bp_hi"]], [y, y], color=c, lw=3.2, alpha=0.4)
            ax.plot(r["rmse_bp"], y, "o", color=c, ms=4, clip_on=False, zorder=3)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(left=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("RMSE of year-on-year inflation error (basis points)")
    axes[0].set_yticks(range(len(ms)), [names.method(m) for m in ms])
    # ranked by the upper end of the interval: a point estimate at the edge of a wide
    # interval is not a safe basis for "lowest"
    head = cl.filter((pl.col("aggregate") == "headline") & pl.col("method").is_in(ms)).sort(
        "rmse_bp_hi"
    )
    best = head.row(0, named=True)
    cheap = head.filter(pl.col("method").is_in(CASCADE_CHEAP)).row(0, named=True)
    fams = list(dict.fromkeys(select.family(m) for m in ms[::-1]))
    key = [
        Line2D([], [], color=style.FAMILY[f], lw=3.2, alpha=0.6, marker="o", ms=4, mew=0)
        for f in fams
    ]
    style.finish(
        fig,
        f"Headline error, upper end of the wider interval: {names.method(best['method'])} "
        f"{best['rmse_bp_hi']:.1f} bp; best cheap method {cheap['rmse_bp_hi']:.1f} bp",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_REF, style.SOURCE_CALC),
        "Dots: posterior-mean confusion matrix from the held-out official specifications. "
        f"Thick lines: 95% interval over {n_draws:,} Dirichlet draws of the matrix "
        "(specifications independent); thin lines: 95% interval over resampled official "
        "sub-items (errors clustered). Products in proportion to CPI weights. Out-of-scope "
        "items and items without specifications held at official values. Panels have their "
        "own scales.",
        legend=(key, [names.FAMILY[f] for f in fams]),
    )
    style.save(fig, "fig10_bp_error", out)


if __name__ == "__main__":
    main()
