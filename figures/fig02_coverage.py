"""Figure 2: the CPI basket by weight, with the items a product classifier can reach."""

from __future__ import annotations

import textwrap

import polars as pl
from matplotlib.patches import Rectangle

from cpilens.viz import style
from cpilens.viz.treemap import squarify

OOS_GREY = "#d9d9d9"
MIN_LABEL = 0.9  # label item blocks of at least this weight (% of headline)
SHORT = {"Tobacco and alcoholic beverages": "Tobacco and alcohol"}
MIN_W = 1.0  # ... and at least this wide (layout units; the canvas is 16 wide)


def fit(text: str, width: float) -> str:
    """Wrap ``text`` to a block ``width`` layout units wide (about 8 characters per unit)."""
    return textwrap.fill(text, max(6, int(width * 8)))


def main() -> None:
    results, out = style.dirs()
    c = pl.read_csv(results / "coverage.csv", schema_overrides={"division": pl.Utf8})
    div = (
        c.group_by("division", "division_name")
        .agg(pl.col("w_equal").sum())
        .sort("w_equal", descending=True)
    )
    fig, ax = style.figure(3.6)
    W, H = 16.0, 7.0
    for (d, name, _), (x, y, w, h) in zip(
        div.iter_rows(), squarify(div["w_equal"].to_list(), (0, 0, W, H)), strict=True
    ):
        items = c.filter(pl.col("division") == d)
        # in-scope items first (largest first), then the out-of-scope remainder
        order = pl.concat(
            [
                items.filter(pl.col("label") != "OOS").sort("w_equal", descending=True),
                items.filter(pl.col("label") == "OOS"),
            ]
        )
        for r, (rx, ry, rw, rh) in zip(
            order.iter_rows(named=True),
            squarify(order["w_equal"].to_list(), (x, y, w, h)),
            strict=True,
        ):
            oos = r["label"] == "OOS"
            ax.add_patch(
                Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor=OOS_GREY if oos else style.FAMILY["embedding"],
                    edgecolor="white",
                    lw=0.6,
                )
            )
            if not oos and r["w_equal"] >= MIN_LABEL and rw >= MIN_W:
                ax.text(
                    rx + 0.08,
                    ry + rh - 0.08,
                    f"{fit(r['name_en'].split(' (')[0], rw)}\n{r['w_equal']:.1f}",
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="white",
                )
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", lw=2.0))
        ax.text(
            x + w - 0.1,
            y + h - 0.1,
            f"{fit(SHORT.get(name, name), min(w, 2.4))}\n{sum(items['w_equal']):.1f}",
            ha="right",
            va="top",
            fontsize=6.5,
            color=style.INK,
            weight="semibold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
        )
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ins = c.filter(pl.col("label") != "OOS")
    share = ins["w_equal"].sum()
    food = ins.filter(pl.col("division") == "1")["w_equal"].sum() / share
    style.finish(
        fig,
        f"Packaged-product items make up {share:.0f}% of the CPI basket; "
        f"{100 * food:.0f}% of that is food",
        style.source(style.SOURCE_CPI, style.SOURCE_CALC),
        f"Area: price-updated CPI weight (% of headline). Blue: the {ins.height} items in the "
        "classifier's "
        "scope; grey: the rest of each division. Item weights inside an official weight cell "
        "are an equal split (estimated).",
    )
    style.save(fig, "fig02_coverage", out)


if __name__ == "__main__":
    main()
