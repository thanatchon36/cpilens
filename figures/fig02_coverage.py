"""Figure 2: the CPI basket by weight, with the items a product classifier can reach."""

from __future__ import annotations

import textwrap

import polars as pl
from matplotlib.patches import Rectangle

from cpilens.viz import style
from cpilens.viz.treemap import squarify

OOS_GREY = "#d9d9d9"
SHORT = {
    "Tobacco and alcoholic beverages": "Tobacco, alcohol",
    "Recreation, reading, education and religion": "Recreation and education",
    "Apparel and footwear": "Apparel, footwear",
}
STRIP = 0.27  # height of one line of a division name (layout units; canvas 16 x 7)
LINE = 0.25  # height of one 7 pt line of an item label, in layout units
MIN_W = 0.8  # item cells narrower than this carry no label


def fit(text: str, width: float) -> str | None:
    """Wrap ``text`` to a block ``width`` layout units wide (about 8 characters per unit);
    None when a word is too long for the block (a label must never cross a cell border)."""
    lines = textwrap.wrap(text, max(4, int(width * 8)), break_long_words=False)
    if max(len(line) for line in lines) > width * 8.5:
        return None
    return "\n".join(lines)


def main() -> None:
    results, out = style.dirs()
    c = pl.read_csv(results / "coverage.csv", schema_overrides={"division": pl.Utf8})
    div = (
        c.group_by("division", "division_name")
        .agg(pl.col("w_equal").sum())
        .sort("w_equal", descending=True)
    )
    fig, ax = style.figure(4.3)
    W, H = 16.0, 7.0
    unlabelled = 0
    for (d, name, _), (x, y, w, h) in zip(
        div.iter_rows(), squarify(div["w_equal"].to_list(), (0, 0, W, H)), strict=True
    ):
        items = c.filter(pl.col("division") == d)
        # the division's name sits in a strip of its own above its items, so it covers no cell
        name = name.replace(" ,", ",")
        head = fit(SHORT.get(name, name), max(w - 0.2, 1.4)) or SHORT.get(name, name)
        strip = min(STRIP * (head.count("\n") + 1) + 0.08, 0.6 * h)
        ax.text(
            x + 0.1,
            y + h - 0.08,
            f"{head} {items['w_equal'].sum():.1f}",
            ha="left",
            va="top",
            fontsize=7.5,
            weight="semibold",
            color=style.INK,
        )
        # in-scope items first (largest first), then the out-of-scope remainder
        order = pl.concat(
            [
                items.filter(pl.col("label") != "OOS").sort("w_equal", descending=True),
                items.filter(pl.col("label") == "OOS"),
            ]
        )
        for r, (rx, ry, rw, rh) in zip(
            order.iter_rows(named=True),
            squarify(order["w_equal"].to_list(), (x, y, w, h - strip)),
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
            if oos:
                continue
            text = fit(f"{r['name_en'].split(' (')[0]} {r['w_equal']:.1f}", rw - 0.15)
            if text and rw >= MIN_W and rh >= LINE * (text.count("\n") + 1) + 0.06:
                ax.text(
                    rx + 0.08, ry + rh - 0.05, text, ha="left", va="top", fontsize=7, color="white"
                )
            else:
                unlabelled += 1
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", lw=2.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ins = c.filter(pl.col("label") != "OOS")
    share = ins["w_equal"].sum()
    food = ins.filter(pl.col("division") == "1")["w_equal"].sum() / share
    small = ""
    if unlabelled:
        small = f" The {unlabelled} smallest are unlabelled; the appendix lists every item."
    style.finish(
        fig,
        f"Packaged-product items make up {share:.0f}% of the CPI basket; "
        f"{100 * food:.0f}% of that is food",
        style.source(style.SOURCE_CPI, style.SOURCE_CALC),
        f"Area: price-updated CPI weight (% of headline). Blue: the {ins.height} items in the "
        f"classifier's scope; grey: the rest of each division.{small} Item weights inside an "
        "official weight cell are an equal split (estimated).",
    )
    style.save(fig, "fig02_coverage", out)


if __name__ == "__main__":
    main()
