"""Figure 9: official vs simulated measured year-on-year inflation, best cheap and best overall
method, with posterior bands."""

from __future__ import annotations

import matplotlib.dates as mdates
import polars as pl

from cpilens.viz import names, select, style


def main() -> None:
    results, out = style.dirs()
    s = pl.read_csv(results / "rq3_series.csv", try_parse_dates=True).filter(
        pl.col("kind") == "yoy"
    )
    cheap, best = select.best_cheap(results), select.best_overall(results)
    fig, axes = style.figure(3.0, ncols=2, sharey=False)
    for ax, agg, title in zip(
        axes, ("headline", "food"), ("(a) Headline", "(b) Food"), strict=True
    ):
        d = s.filter(pl.col("aggregate") == agg)
        for m, ls in ((cheap, (0, (3, 1.5))), (best, "-")):
            e = d.filter(pl.col("method") == m).sort("date")
            c = style.FAMILY[select.family(m)]
            ax.fill_between(
                e["date"],
                e["measured_lo_bp"] / 100,
                e["measured_hi_bp"] / 100,
                color=c,
                alpha=0.15,
                lw=0,
            )
            # the best method is drawn wide under the official line, visible where they coincide
            wide = 2.6 if m == best else 1.2
            y = e["measured_bp"] / 100
            ax.plot(e["date"], y, color=c, ls=ls, lw=wide, label=names.method(m))
        o = d.filter(pl.col("method") == cheap).sort("date")
        ax.plot(  # drawn last, so a method that tracks the official series cannot hide it
            o["date"],
            o["official_bp"] / 100,
            color=style.FAMILY["official"],
            lw=1.0,
            zorder=5,
            label="Official",
        )
        ax.axhline(0, color=style.RULE, lw=0.6)
        ax.set_title(title, fontsize=8.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Year-on-year inflation (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    e = (
        pl.read_csv(results / "rq3_errors.csv")
        .filter(
            (pl.col("variant") == "full")
            & (pl.col("weights") == "equal")
            & (pl.col("universe") == "cpi")
            & (pl.col("kind") == "yoy")
            & (pl.col("aggregate") == "food")
            & (pl.col("method") == cheap)
        )
        .row(0, named=True)
    )
    style.finish(
        fig,
        f"{names.method(cheap)} would misstate food inflation by {e['max_abs_bp']:.0f} bp at worst",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_REF, style.SOURCE_CALC),
        "Simulated with each method's confusion matrix on the held-out official "
        "specifications, products in proportion to CPI weights; bands: 95% over Dirichlet "
        "draws. Out-of-scope items and items without specifications held at official values. "
        "Panels have their own scales.",
        legend=(handles[::-1], labels[::-1]),  # official first
    )
    style.save(fig, "fig09_series", out)


if __name__ == "__main__":
    main()
