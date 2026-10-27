"""Shared fixtures."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from cpilens.paths import TAXONOMY


@pytest.fixture
def synthetic_eval(tmp_path):
    """(reference table, results dir): a synthetic reference set and a results tree in the
    shape ``cpilens models`` writes, small enough for CI. 120 fake specifications with random
    labels over the in-scope items and OOS; a grid of the three cascade cheap models, both
    LLMs, their dev tables and cost log; Haiku's predictions are the labels themselves, the
    others carry random errors."""
    from cpilens.evaluate import run

    rng = np.random.default_rng(0)
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    pool = [*scope.filter(pl.col("in_scope"))["label"].to_list(), "OOS"]
    ids = [f"moc:{i:04d}" for i in range(120)]
    labels = [str(rng.choice(pool)) for _ in ids]
    ref = pl.DataFrame({"id": ids, "label": labels}).with_columns(
        pl.col("label").alias("stratum"),
        pl.lit(1.0).alias("incl_prob"),
        pl.lit(False).alias("unsure"),
    )
    res = tmp_path / "results"
    (res / "preds" / "embed").mkdir(parents=True)
    (res / "preds" / "llm").mkdir()
    grid = []
    for r in run.CASCADE_CHEAP:
        emb, clf, scenario = r.split("__")
        grid.append(
            {"run": r, "emb": emb, "clf": clf, "scenario": scenario, "text_variant": "full"}
            | {"pred_s_per_item": 1e-4}
        )
    pl.DataFrame(grid).write_csv(res / "rq1_dev_grid.csv")
    (res / "embed_timing.json").write_text(json.dumps({"e5-large": 1e-3, "bge-m3": 1e-3}))
    pl.DataFrame(
        {"model": list(run.LLMS), "purpose": ["eval:x", "eval:y"], "usd": [1e-4, 1e-3]}
    ).write_csv(res / "llm_costs.csv")
    dev = [
        {
            "threshold": t,
            "routed": 0.1,
            "usd_per_million": 1.0,
            "acc": 0.9,
            "acc_w": 0.9 if t == 0.7 else 0.8,  # 0.7 must be the dev-chosen threshold
            "cheap": c,
            "llm": m,
            "eval_set": "thai_val_weak",
            "n": 10,
        }
        for c in run.CASCADE_CHEAP
        for m in run.LLMS
        for t in (0.5, 0.7)
    ]
    pl.DataFrame(dev).write_csv(res / "cascade_dev.csv")

    def preds(flip: float) -> pl.DataFrame:
        pred = [str(rng.choice(pool)) if rng.random() < flip else y for y in labels]
        return pl.DataFrame(
            {
                "id": ids,
                "eval_set": ["moc_test"] * len(ids),
                "label": labels,
                "pred": pred,
                "conf": rng.random(len(ids)),
            }
        )

    for r in run.CASCADE_CHEAP:
        preds(0.3).write_parquet(res / "preds" / "embed" / f"{r}.parquet")
    preds(0.0).write_parquet(res / "preds" / "llm" / "claude-haiku-5-5__low.parquet")
    preds(0.1).write_parquet(res / "preds" / "llm" / "claude-opus-5-5__low.parquet")
    return ref, res
