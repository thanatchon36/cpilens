"""Command-line entry points; each step reads public inputs and writes a versioned output."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import typer

from cpilens.data import tpso
from cpilens.data.tpso_release import PRICE_REF, build_from_dir
from cpilens.paths import RAW, TAXONOMY
from cpilens.taxonomy.hierarchy import Hierarchy
from cpilens.taxonomy.weights import cells, leaf_weights

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def fetch_tpso(start: str = "2562-01", end: str = "2569-09") -> None:
    """Download monthly national CPI indices (BE year-month range, inclusive)."""
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    paths = tpso.fetch_months((ys, ms), (ye, me))
    (tpso.API_DIR / "fetched_utc.txt").write_text(datetime.now(UTC).isoformat() + "\n")
    typer.echo(f"{len(paths)} monthly files in {tpso.API_DIR}")


@app.command()
def weights() -> None:
    """Official aggregate weights and leaf weights (two labelled sensitivities)."""
    idx = tpso.load_index()
    h = Hierarchy.from_index(idx)
    agg = pl.read_csv(TAXONOMY / "thcpi_weight_aggregates.csv", schema_overrides={"code": pl.Utf8})
    w, months = build_from_dir(RAW / "tpso" / "releases", idx, agg)
    used = ";".join(f"{y}-{m:02d}" for y, m in months)
    w.select("code", "name_en", "kind", "w0", "w0_spread", "w_pr").with_columns(
        pl.lit(used).alias("release_months_be"),
        pl.lit(f"{PRICE_REF[0]}-{PRICE_REF[1]:02d}").alias("price_ref_be"),
    ).write_csv(TAXONOMY / "thcpi_weights_2566.csv")

    official = dict(zip(w["code"], w["w_pr"], strict=True))
    lw = (
        cells(h, official)
        .join(
            leaf_weights(h, idx, official, "equal").select(
                "leaf", pl.col("weight").alias("w_equal")
            ),
            on="leaf",
        )
        .join(
            leaf_weights(h, idx, official, "nnls").select("leaf", pl.col("weight").alias("w_nnls")),
            on="leaf",
        )
    )
    lw.with_columns(
        pl.col("leaf").replace_strict(h.names).alias("name_th"),
        pl.col("cell_weight", "w_equal", "w_nnls").round(4),
    ).select("leaf", "name_th", "cell", "residual_cell", "cell_weight", "w_equal", "w_nnls").sort(
        "leaf"
    ).write_csv(TAXONOMY / "thcpi_leaf_weights_2566.csv")
    typer.echo(f"weights from {len(months)} releases ({used}); {lw.height} leaves")


if __name__ == "__main__":
    app()
