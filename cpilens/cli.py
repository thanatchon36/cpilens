"""Command-line entry points; each step reads public inputs and writes a versioned output."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import typer

from cpilens.data import download, moc, off, tpso
from cpilens.data.tpso_release import PRICE_REF, build_from_dir
from cpilens.label import sampling
from cpilens.paths import DATA, INTERIM, RAW, RESULTS, TAXONOMY
from cpilens.taxonomy import coverage
from cpilens.taxonomy.crosswalk import Crosswalk
from cpilens.taxonomy.hierarchy import Hierarchy
from cpilens.taxonomy.weights import cells, leaf_weights

GOLD = DATA / "gold"

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def fetch_raw() -> None:
    """Download the files in data/manifest.csv (if missing) and verify checksums."""
    bad = download.fetch_all()
    if bad:
        typer.echo(f"checksum mismatch (upstream changed?): {bad}", err=True)
        raise typer.Exit(1)
    typer.echo("raw files present and verified")


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
    coverage.table().write_csv(RESULTS / "coverage.csv")
    typer.echo(f"weights from {len(months)} releases ({used}); {lw.height} leaves")


@app.command()
def products() -> None:
    """OFF/OBF extract with silver labels, and MOC specifications with gold labels."""
    p = Crosswalk.load().apply(pl.read_parquet(off.extract()))
    p.write_parquet(off.OUT)
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    h = Hierarchy.from_index(tpso.load_index())
    specs = moc.label_specs(moc.load_specs(), scope, h.leaves)
    specs.write_parquet(INTERIM / "moc_specs.parquet")
    typer.echo(
        f"{p.height} OFF/OBF products ({p['silver_label'].is_not_null().sum()} silver); "
        f"{specs.height} MOC specs"
    )


@app.command()
def sample_gold() -> None:
    """Draw the gold-labelling sample (fixed seed) into data/gold/sample.csv."""
    lc = pl.read_csv(
        TAXONOMY / "thcpi_leaf_weights_2566.csv",
        schema_overrides={"leaf": pl.Utf8, "cell": pl.Utf8},
    )
    fr = sampling.frame(pl.read_parquet(off.OUT), lc)
    s = sampling.draw(fr, dict(zip(lc["cell"], lc["cell_weight"], strict=True)))
    GOLD.mkdir(parents=True, exist_ok=True)
    urls = [
        off.image_url(*r) for r in s.select("barcode", "source", "image_key", "image_rev").rows()
    ]
    s.with_columns(pl.Series("image_url", urls, dtype=pl.Utf8)).select(
        "order",
        "barcode",
        "source",
        "stratum",
        "incl_prob",
        "double",
        "name_th",
        "name_en",
        "brands",
        "quantity",
        "image_url",
    ).write_csv(GOLD / "sample.csv")
    typer.echo(
        f"{s.height} products ({s['double'].sum()} double-labelled) from a frame of "
        f"{fr.height}; seed {sampling.SEED}"
    )


@app.command()
def label(
    pass_no: int = typer.Option(1, "--pass", help="1 = all products; 2 = blind relabel"),
    port: int = 8765,
) -> None:
    """Start the local gold-labelling app (127.0.0.1 only; every label autosaves)."""
    from cpilens.label.app import serve

    serve(pass_no, port)


@app.command()
def models(
    llm: bool = typer.Option(True, help="also run the LLM passes (cached responses are free)"),
) -> None:
    """Embeddings, the RQ1 grid of embedding classifiers, XLM-R fine-tune, LLM predictions."""
    from cpilens.classify import experiments, finetune
    from cpilens.classify import llm as llm_mod
    from cpilens.classify.data import corpus
    from cpilens.embed import encode, projection
    from cpilens.paths import RESULTS

    grid = [experiments.run()]
    main = {"nonthai+thaiweak+mocall"}
    for b, q in ((False, True), (True, False), (False, False)):
        grid.append(experiments.run(only=main, with_brand=b, with_quantity=q))
    ft = [finetune.run(s) for s in ("nonthai", "nonthai+thaiweak+mocall")]
    pl.concat([*grid, pl.DataFrame(ft)], how="diagonal_relaxed").write_csv(
        RESULTS / "rq1_dev_grid.csv"
    )
    sample = corpus().sample(5000, seed=20261027)["text"].to_list()
    timing = {m: encode.seconds_per_item(m, sample) for m in encode.MODELS}
    (RESULTS / "embed_timing.json").write_text(json.dumps(timing, indent=2))
    projection.run()
    if llm:
        llm_mod.run_eval("claude-haiku-5-5", max_usd=3.0)
        llm_mod.run_eval("claude-opus-5-5", max_usd=16.0)


@app.command()
def cascade() -> None:
    """Dev-set cascade sweep: cheap classifier first, LLM for low-confidence products."""
    from cpilens.evaluate import frontier as fr
    from cpilens.evaluate import metrics as m
    from cpilens.evaluate.run import CASCADE_CHEAP
    from cpilens.paths import RESULTS

    lw = m.cpi_label_weights()
    costs = pl.read_csv(RESULTS / "llm_costs.csv")
    grid = pl.read_csv(RESULTS / "rq1_dev_grid.csv")
    embed_s = json.loads((RESULTS / "embed_timing.json").read_text())
    out = []
    for llm_model in ("claude-haiku-5-5", "claude-opus-5-5"):
        lp = pl.read_parquet(RESULTS / "preds" / "llm" / f"{llm_model}__low.parquet")
        llm_cost = fr.api_cost_per_item(costs, llm_model)
        for run in CASCADE_CHEAP:
            row = grid.filter(pl.col("run") == run).row(0, named=True)
            cp = pl.read_parquet(RESULTS / "preds" / "embed" / f"{run}.parquet")
            d = cp.join(
                lp.select("id", "eval_set", pl.col("pred").alias("llm_pred")), on=["id", "eval_set"]
            )
            for s in ("thai_val_weak",):  # the test split is never used to choose
                e = d.filter(pl.col("eval_set") == s)
                y = e["label"].to_list()
                w = m.item_weights(y, lw)

                def score(p: list[str | None], y: list[str] = y, w=w) -> dict[str, float]:  # type: ignore[no-untyped-def]
                    return {"acc": m.accuracy(y, p), "acc_w": m.accuracy(y, p, w)}

                sw = fr.sweep(
                    e["pred"].to_list(),
                    e["conf"].to_list(),
                    e["llm_pred"].to_list(),
                    score,
                    fr.cheap_cost_per_item(row, embed_s),
                    llm_cost,
                )
                out.append(
                    sw.with_columns(
                        pl.lit(run).alias("cheap"),
                        pl.lit(llm_model).alias("llm"),
                        pl.lit(s).alias("eval_set"),
                        pl.lit(e.height).alias("n"),
                    )
                )
    pl.concat(out).write_csv(RESULTS / "cascade_dev.csv")


@app.command("eval")
def eval_() -> None:
    """Score every method on the reference set (held-out official specifications)."""
    from cpilens.evaluate.run import run_eval

    run_eval()


@app.command()
def econ() -> None:
    """RQ3: basis-point inflation errors implied by each method's confusion matrix."""
    from cpilens.econ.run import run_econ

    run_econ()


@app.command()
def agreement() -> None:
    """RQ4: pass-1 vs pass-2 agreement on the double-labelled subset (the label ceiling)."""
    from cpilens.evaluate.agreement import run_agreement

    run_agreement()


@app.command()
def report() -> None:
    """Data summary and ``paper/numbers.tex``: every number the paper prints, from results/."""
    from cpilens.report import numbers

    numbers.describe()
    typer.echo(f"{numbers.write()} macros -> {numbers.OUT.relative_to(numbers.ROOT)}")
    typer.echo(f"tables -> {', '.join(numbers.write_tables())}")
    from cpilens.report import manifest

    typer.echo(f"manifest -> {manifest.write().relative_to(numbers.ROOT)}")


if __name__ == "__main__":
    app()
