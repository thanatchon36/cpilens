"""How much of the CPI basket the classifier can reach: in-scope item weights by division,
with each division's out-of-scope remainder as one block (figure 2)."""

from __future__ import annotations

import polars as pl

from cpilens.paths import TAXONOMY


def table() -> pl.DataFrame:
    w = pl.read_csv(TAXONOMY / "thcpi_weights_2566.csv", schema_overrides={"code": pl.Utf8})
    div = w.filter(pl.col("kind") == "division").select(
        pl.col("code").str.slice(0, 1).alias("division"),
        pl.col("name_en").alias("division_name"),
        pl.col("w_pr").alias("division_weight"),
    )
    lw = pl.read_csv(TAXONOMY / "thcpi_leaf_weights_2566.csv", schema_overrides={"leaf": pl.Utf8})
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    leaf = lw.join(
        scope.select("label", "name_en", "in_scope"), left_on="leaf", right_on="label", how="left"
    ).with_columns(
        pl.col("leaf").str.slice(0, 1).alias("division"),
        pl.col("in_scope").fill_null(False),
        pl.col("residual_cell").alias("weight_estimated"),
    )
    inside = leaf.filter(pl.col("in_scope")).select(
        "division",
        pl.col("leaf").alias("label"),
        "name_en",
        "w_equal",
        "w_nnls",
    )
    outside = (
        leaf.filter(~pl.col("in_scope"))
        .group_by("division")
        .agg(pl.col("w_equal").sum(), pl.col("w_nnls").sum())
        .with_columns(
            pl.lit("OOS").alias("label"),
            pl.lit("Out of scope").alias("name_en"),
        )
        .select(inside.columns)
    )
    return (
        pl.concat([inside, outside]).join(div, on="division", how="left").sort("division", "label")
    )
