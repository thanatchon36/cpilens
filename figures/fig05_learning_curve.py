"""Figure 5: gold weighted accuracy against the number of official Thai labels, starting from
zero-shot cross-lingual transfer."""

from __future__ import annotations

import polars as pl

from cpilens.evaluate.run import LLMS
from cpilens.viz import names, style

BUDGETS = ("100", "250", "500", "1000", "all")
SERIES = (
    ("e5-large", "mlp", "nonthai+thaiweak+moc", "-", "Non-Thai + Thai weak + n official"),
    ("e5-large", "mlp", "moc", (0, (3, 1.5)), "n official only"),
)


def main() -> None:
    results, out = style.dirs()
    sc = pl.read_csv(results / "gold_scores.csv").filter(
        (pl.col("weights") == "equal") & (pl.col("text_variant").fill_null("full") == "full")
    )
    grid = pl.read_csv(results / "rq1_dev_grid.csv").select(
        pl.col("run").alias("method"), "n_official"
    )
    sc = sc.join(grid, on="method", how="left")
    fig, ax = style.figure(3.2)
    xs = {b: i + 2 for i, b in enumerate(BUDGETS)}  # 0: zero-shot, 1: + Thai weak
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
            label=f"{names.method(m)} (no training labels)",
        )
    n_all = sc.filter(pl.col("scenario") == "nonthai+thaiweak+mocall")["n_official"].max()
    ax.set_xticks(
        range(len(BUDGETS) + 2),
        ["Zero-shot\n(non-Thai)", "+ Thai\nweak", *BUDGETS[:-1], f"all\n({n_all:,})"],
    )
    ax.set_xlabel("Official Thai (MOC) labels used in training")
    ax.set_ylabel("Weighted accuracy on gold")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2, fontsize=7)
    e5 = sc.filter((pl.col("emb") == "e5-large") & (pl.col("clf") == "mlp"))
    z = e5.filter(pl.col("scenario") == "nonthai")["acc_w"][0]
    a = e5.filter(pl.col("scenario") == "nonthai+thaiweak+mocall")["acc_w"][0]
    style.finish(
        fig,
        f"Thai labels lift e5-large MLP from {100 * z:.0f}% (zero-shot) "
        f"to {100 * a:.0f}% weighted accuracy",
        style.source(
            style.SOURCE_CPI,
            style.SOURCE_OFF,
            "Ministry of Commerce product specifications",
            style.SOURCE_GOLD,
            style.SOURCE_CALC,
        ),
        "Budgets count every official label a run touches (training and validation). "
        "Bands: 95% stratified bootstrap.",
    )
    style.save(fig, "fig05_learning_curve", out)


if __name__ == "__main__":
    main()
