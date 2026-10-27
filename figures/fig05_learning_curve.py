"""Figure 5: reference-set weighted accuracy against the number of official Thai labels,
starting from zero-shot cross-lingual transfer."""

from __future__ import annotations

import math

import polars as pl

from cpilens.evaluate.run import LLMS
from cpilens.viz import names, style

BUDGETS = ("100", "250", "500", "1000", "all")
SERIES = (
    ("e5-large", "mlp", "nonthai+thaiweak+moc", "-", "non-Thai + Thai weak + $n$ official"),
    ("e5-large", "mlp", "moc", (0, (3, 1.5)), "$n$ official only"),
)
STEPS = 2  # categorical steps before the numeric budgets: zero-shot, + Thai weak


def xpos(n: int) -> float:
    """Budgets on a log scale, 100 labels one unit right of the categorical steps."""
    return STEPS + math.log10(n / 100)


def main() -> None:
    results, out = style.dirs()
    sc = pl.read_csv(results / "gold_scores.csv").filter(
        (pl.col("weights") == "equal") & (pl.col("text_variant").fill_null("full") == "full")
    )
    grid = pl.read_csv(results / "rq1_dev_grid.csv").select(
        pl.col("run").alias("method"), "n_official"
    )
    sc = sc.join(grid, on="method", how="left")
    n_all = int(sc.filter(pl.col("scenario") == "nonthai+thaiweak+mocall")["n_official"].max())
    xs = {b: xpos(n_all if b == "all" else int(b)) for b in BUDGETS}
    fig, ax = style.figure(3.4)
    c = style.FAMILY["embedding"]
    for emb, clf, prefix, ls, label in SERIES:
        d = sc.filter((pl.col("emb") == emb) & (pl.col("clf") == clf))
        pts = [d.filter(pl.col("scenario") == f"{prefix}{b}").row(0, named=True) for b in BUDGETS]
        x = [xs[b] for b in BUDGETS]
        if prefix.startswith("nonthai"):
            z = [
                d.filter(pl.col("scenario") == s).row(0, named=True)
                for s in ("nonthai", "nonthai+thaiweak")
            ]
            pts, x = z + pts, [0, 1, *x]
        y = [p["acc_w"] for p in pts]
        ax.fill_between(
            x, [p["acc_w_lo"] for p in pts], [p["acc_w_hi"] for p in pts], color=c, alpha=0.10, lw=0
        )
        ax.plot(
            x,
            y,
            color=c,
            ls=ls,
            marker="o",
            ms=3,
            label=f"{names.method(f'{emb}__{clf}__x')}: {label}",
        )
    for m, ls in zip(LLMS, ("-", (0, (3, 1.5))), strict=True):
        r = sc.filter(pl.col("method") == m).row(0, named=True)
        ax.axhline(
            r["acc_w"],
            color=style.FAMILY["llm"],
            ls=ls,
            lw=1.0,
            label=f"{names.method(m)} (few-shot prompt, no training)",
        )
    ax.axvline(STEPS - 0.5, color=style.RULE, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(
        [0, 1, *(xs[b] for b in BUDGETS)],
        [
            "Zero-shot\n(non-Thai)",
            "+ Thai\nweak",
            *(f"{int(b):,}" for b in BUDGETS[:-1]),
            f"all\n({n_all:,})",
        ],
    )
    ax.set_xlabel("Official Thai (TPSO) labels in training (log scale right of the dotted line)")
    ax.set_ylabel(style.ACC_LABEL)
    style.percent_axis(ax)
    e5 = sc.filter((pl.col("emb") == "e5-large") & (pl.col("clf") == "mlp"))
    z = e5.filter(pl.col("scenario") == "nonthai")["acc_w"][0]
    k = e5.filter(pl.col("scenario") == "nonthai+thaiweak+moc1000")["acc_w"][0]
    a = e5.filter(pl.col("scenario") == "nonthai+thaiweak+mocall")["acc_w"][0]
    style.finish(
        fig,
        f"No Thai labels: {100 * z:.0f}%; adding 1,000 official labels: {100 * k:.0f}%; "
        f"adding all: {100 * a:.0f}%",
        style.source(
            style.SOURCE_CPI,
            style.SOURCE_OFF,
            style.SOURCE_REF,
            style.SOURCE_CALC,
        ),
        "Budgets count every official label a run touches (training and validation). "
        "Bands: 95% stratified bootstrap.",
        legend=ax,
        legend_cols=2,
    )
    style.save(fig, "fig05_learning_curve", out)


if __name__ == "__main__":
    main()
