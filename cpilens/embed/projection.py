"""Cross-lingual alignment of the embedding space: does a Thai product sit next to non-Thai
products of the same CPI item?

Alignment of label k = mean, over sampled Thai products of k, of the share of their 10
nearest non-Thai neighbours (cosine) that carry label k. Labels are official (MOC) or weak
(OFF crosswalk); gold products are excluded. The 2-D map is a t-SNE of a balanced sample
(fixed seed), for display only; alignment is computed in the full embedding space.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cpilens.classify.data import corpus
from cpilens.embed.encode import embed
from cpilens.paths import RESULTS

SEED = 20261027
K = 10


def _sample(d: pl.DataFrame, by: list[str], n: int) -> pl.DataFrame:
    return (
        d.with_columns(pl.int_range(pl.len()).shuffle(SEED).over(by).alias("_r"))
        .filter(pl.col("_r") < n)
        .drop("_r")
    )


def run(model: str = "e5-large", per_label: int = 200, per_cell: int = 40) -> None:
    from sklearn.manifold import TSNE

    c = corpus().filter((pl.col("split") != "gold") & pl.col("label").is_not_null())
    c = c.with_columns(
        pl.when(pl.col("region") == "thai")
        .then(pl.lit("Thai"))
        .otherwise(pl.lit("Other"))
        .alias("lang")
    )
    x = embed(c["text"].to_list(), model)
    other = (c["lang"] == "Other").to_numpy()
    xo, yo = x[other], np.asarray(c.filter(pl.col("lang") == "Other")["label"].to_list())
    thai = _sample(c.with_row_index("row").filter(pl.col("lang") == "Thai"), ["label"], per_label)
    xt = x[thai["row"].to_numpy()]
    share = np.empty(len(xt))
    for s in range(0, len(xt), 512):
        sim = xt[s : s + 512] @ xo.T
        nn = np.argpartition(-sim, K, axis=1)[:, :K]
        share[s : s + 512] = (
            yo[nn] == np.asarray(thai["label"][s : s + 512].to_list())[:, None]
        ).mean(1)
    n_other = c.filter(pl.col("lang") == "Other").group_by("label").len().rename({"len": "n_other"})
    (
        thai.with_columns(pl.Series("share_same", share))
        .group_by("label")
        .agg(pl.col("share_same").mean().alias("alignment"), pl.len().alias("n_thai_sampled"))
        .join(n_other, on="label", how="left")
        .with_columns(pl.col("n_other").fill_null(0))
        # no non-Thai products of the label: alignment is undefined, not zero
        .with_columns(pl.when(pl.col("n_other") > 0).then(pl.col("alignment")).alias("alignment"))
        .sort("label")
        .write_csv(RESULTS / "crosslingual_alignment.csv")
    )
    m = _sample(c.with_row_index("row"), ["label", "lang"], per_cell)
    xy = TSNE(2, init="pca", random_state=SEED, perplexity=30).fit_transform(x[m["row"].to_numpy()])
    m.select("id", "label", "lang", "source").with_columns(
        pl.Series("x", xy[:, 0]), pl.Series("y", xy[:, 1])
    ).write_csv(RESULTS / "embedding_map.csv")
