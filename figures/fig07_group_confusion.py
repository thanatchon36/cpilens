"""Figure 7: row-normalised confusion between CPI groups on gold (best cheap method)."""

from __future__ import annotations

import numpy as np
import polars as pl

from cpilens.paths import TAXONOMY
from cpilens.viz import names, select, style


def group_names() -> dict[str, str]:
    w = pl.read_csv(TAXONOMY / "thcpi_weights_2566.csv", schema_overrides={"code": pl.Utf8})
    out = {
        c[:3]: n.split(" (")[0]
        for c, n in zip(w["code"], w["name_en"], strict=True)
        if c.endswith("00")
    }
    # groups the published weights table does not list (only their divisions)
    extra = {"421": "Personal care products", "721": "Alcoholic beverages"}
    return {**extra, **out, "OOS": "Out of scope", "ABSTAIN": "Abstain"}


def main() -> None:
    results, out = style.dirs()
    m = select.best_cheap(results)
    c = pl.read_csv(
        results / "gold_confusion_group.csv", schema_overrides={"true": pl.Utf8, "pred": pl.Utf8}
    )
    c = c.filter(pl.col("method") == m)
    rows = sorted(set(c["true"]), key=lambda k: (k == "OOS", k))
    cols = [*rows, *(["ABSTAIN"] if "ABSTAIN" in set(c["pred"]) else [])]
    cols += sorted(set(c["pred"]) - set(cols))
    mat = np.zeros((len(rows), len(cols)))
    for r in c.iter_rows(named=True):
        mat[rows.index(r["true"]), cols.index(r["pred"])] = r["share"]
    gn = group_names()
    fig, ax = style.figure(4.6)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if mat[i, j] >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{100 * mat[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] > 0.55 else style.INK,
                )
    ax.set_xticks(
        range(len(cols)), [gn.get(k, k) for k in cols], rotation=60, ha="right", fontsize=6.5
    )
    ax.set_yticks(range(len(rows)), [gn.get(k, k) for k in rows], fontsize=6.5)
    ax.set_xlabel("Predicted group")
    ax.set_ylabel("True group")
    ax.spines[:].set_visible(False)
    diag = np.mean([mat[i, cols.index(k)] for i, k in enumerate(rows)])
    style.finish(
        fig,
        f"{names.method(m)}: on average {100 * diag:.0f}% of a group's products "
        "stay in the right group",
        style.source(style.SOURCE_CPI, style.SOURCE_OFF, style.SOURCE_GOLD, style.SOURCE_CALC),
        "Cells: % of the true group's Thai product universe (Horvitz–Thompson weighted); "
        "rows sum to 100.",
    )
    style.save(fig, "fig07_group_confusion", out)


if __name__ == "__main__":
    main()
