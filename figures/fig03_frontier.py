"""Figure 3: expenditure-weighted accuracy-coverage frontier on the reference set,
with bootstrap bands."""

from __future__ import annotations

import json

import polars as pl

from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.viz import names, select, style

SHOWN = (*CASCADE_CHEAP, *LLMS, *(f"cascade__{CASCADE_CHEAP[0]}__{m}" for m in LLMS))


def main() -> None:
    results, out = style.dirs()
    n_boot = json.loads((results / "gold_summary.json").read_text())["n_boot"]
    f = pl.read_csv(results / "gold_frontier.csv").drop_nulls("acc_w")
    fig, ax = style.figure(3.4)
    dashes = {}
    for m in SHOWN:
        d = f.filter(pl.col("method") == m).sort("coverage_w")
        if d.height == 0:
            continue
        fam = select.family(m)
        ls = "-" if fam not in dashes else (0, (3, 1.5))
        dashes[fam] = True
        c = style.FAMILY[fam]
        if ls == "-":  # one band per family keeps the chart readable
            ax.fill_between(
                d["coverage_w"], d["acc_w_lo"], d["acc_w_hi"], color=c, alpha=0.10, lw=0
            )
        ax.plot(d["coverage_w"], d["acc_w"], color=c, ls=ls, label=names.method(m))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Evaluated CPI weight classified (%)")
    ax.set_ylabel(style.ACC_LABEL)
    style.percent_axis(ax, "x")
    style.percent_axis(ax)
    full = f.filter(pl.col("threshold") == 0).filter(pl.col("method").is_in(SHOWN)).sort("acc_w")
    top = full.row(-1, named=True)
    style.finish(
        fig,
        f"At full coverage, {names.method(top['method'])} classifies "
        f"{100 * top['acc_w']:.0f}% of evaluated CPI weight correctly",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_REF, style.SOURCE_CALC),
        "Accuracy is on the classified products; those below a confidence threshold are left "
        f"unclassified. Bands (first method of each family): 95% stratified bootstrap "
        f"({n_boot:,} resamples). Weights: equal split within residual cells.",
        legend=ax,
    )
    style.save(fig, "fig03_frontier", out)


if __name__ == "__main__":
    main()
