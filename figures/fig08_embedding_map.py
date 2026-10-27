"""Figure 8: cross-lingual alignment in the e5-large space: (a) t-SNE map of Thai and non-Thai
products, (b) per-item share of a Thai product's nearest non-Thai neighbours with its label."""

from __future__ import annotations

import polars as pl

from cpilens.embed.projection import K
from cpilens.viz import names, style


def main() -> None:
    results, out = style.dirs()
    mp = pl.read_csv(results / "embedding_map.csv", schema_overrides={"label": pl.Utf8})
    al = pl.read_csv(results / "crosslingual_alignment.csv", schema_overrides={"label": pl.Utf8})
    al = al.filter(pl.col("alignment").is_not_null() & (pl.col("label") != "OOS")).sort("alignment")
    lab = names.labels()
    fig, (a, b) = style.figure(4.6, ncols=2, gridspec_kw={"width_ratios": [1.1, 1]})
    for lang, label, col, z in (  # grey and the embedding family's blue: no method colour
        ("Other", "Non-Thai products", "#b0b0b0", 1),
        ("Thai", "Thai products", style.FAMILY["embedding"], 2),
    ):
        d = mp.filter(pl.col("lang") == lang)
        a.scatter(d["x"], d["y"], s=2.5, color=col, alpha=0.55, lw=0, zorder=z, label=label)
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
    a.spines[:].set_visible(False)
    a.set_title("(a) t-SNE map of product texts", fontsize=8.5)
    y = range(al.height)
    b.barh(y, al["alignment"], color=style.FAMILY["embedding"], height=0.7)
    b.set_yticks(y, [lab[k] for k in al["label"]], fontsize=6.5)
    b.set_xlim(0, 1)
    style.percent_axis(b, "x")
    b.grid(axis="x")
    b.grid(axis="y", visible=False)
    b.set_xlabel(
        f"Share of a Thai product's {K} nearest non-Thai\nneighbours that carry its item (%)"
    )
    b.set_title("(b) Cross-lingual alignment by item", fontsize=8.5)
    hi, lo = al.row(-1, named=True), al.row(0, named=True)
    style.finish(
        fig,
        f"Cross-lingual alignment ranges from {100 * lo['alignment']:.0f}% ({lab[lo['label']]}) "
        f"to {100 * hi['alignment']:.0f}% ({lab[hi['label']]})",
        style.source(style.SOURCE_OFF, style.SOURCE_SPECS, style.SOURCE_CALC),
        "Labels: official (TPSO) or weak (Open Food Facts crosswalk); reference set excluded. "
        "Items with no non-Thai products are omitted. The map is for display only.",
        legend=a,
        legend_kw={"markerscale": 4},
    )
    style.save(fig, "fig08_embedding_map", out)


if __name__ == "__main__":
    main()
