"""Which methods a figure shows, chosen by a fixed rule from the reference-set scores.

- best cheap: the cascade base model (no API) with the highest reference-set acc_w;
- best overall: the headline method (LLMs, cascades, cascade base models) with the highest
  reference-set acc_w, ties broken by lower cost.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cpilens.evaluate.run import CASCADE_CHEAP, LLMS


def family(m: str) -> str:
    if m in LLMS:
        return "llm"
    if m.startswith("cascade__"):
        return "cascade"
    return "finetune" if m.startswith("xlmr") else "embedding"


def _scores(results: Path) -> pl.DataFrame:
    return pl.read_csv(results / "gold_scores.csv").filter(pl.col("weights") == "equal")


def best_cheap(results: Path) -> str:
    s = _scores(results).filter(pl.col("method").is_in(CASCADE_CHEAP))
    return str(s.sort(["acc_w", "usd_per_million"], descending=[True, False])["method"][0])


def best_overall(results: Path) -> str:
    s = _scores(results).filter(
        pl.col("family").is_in(["llm", "cascade"]) | pl.col("method").is_in(CASCADE_CHEAP)
    )
    return str(s.sort(["acc_w", "usd_per_million"], descending=[True, False])["method"][0])
