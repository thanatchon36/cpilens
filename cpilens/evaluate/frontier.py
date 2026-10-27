"""Cost model and the confidence cascade (cheap classifier first, LLM for the rest).

Cost per million products = API cost (mean logged USD per billed call, cache writes
included) + local compute at an *assumed* GPU price of ``GPU_USD_PER_HOUR`` applied to
measured wall-clock seconds per item (embedding + prediction). The assumption is stated in
the paper; the API part is measured.

Cascade: keep the cheap prediction when its calibrated confidence is at least ``t``, route
the product to the LLM otherwise. Every product pays the cheap cost; routed products also
pay the LLM cost. Sweeping ``t`` from 0 (never route) to above 1 (always route) traces the
cost-accuracy curve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

GPU_USD_PER_HOUR = 1.00
THRESHOLDS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01)


def local_cost_per_item(seconds_per_item: float) -> float:
    return seconds_per_item * GPU_USD_PER_HOUR / 3600


def api_cost_per_item(costs: pl.DataFrame, model: str, purpose_prefix: str = "eval:") -> float:
    d = costs.filter((pl.col("model") == model) & pl.col("purpose").str.starts_with(purpose_prefix))
    return float(np.mean(d["usd"].to_numpy())) if d.height else float("nan")


def cascade(
    cheap_pred: Sequence[str | None],
    cheap_conf: Sequence[float],
    llm_pred: Sequence[str | None],
    t: float,
) -> tuple[list[str | None], float]:
    """Predictions and routed share at threshold ``t``."""
    conf = np.asarray(cheap_conf, dtype=float)
    route = conf < t
    pred = [lp if r else cp for cp, lp, r in zip(cheap_pred, llm_pred, route, strict=True)]
    return pred, float(route.mean())


def cascade_cost(routed: float, cheap_cost: float, llm_cost: float) -> float:
    return cheap_cost + routed * llm_cost


def cheap_cost_per_item(run: dict[str, object], embed_s: dict[str, float]) -> float:
    """Local cost of one grid run: prediction time plus, for frozen-embedding classifiers,
    encoding time (a fine-tuned encoder's prediction time already includes its encoder)."""
    s = float(run["pred_s_per_item"]) + embed_s.get(str(run["emb"]), 0.0)  # type: ignore[arg-type]
    return local_cost_per_item(s)


def sweep(
    cheap_pred: Sequence[str | None],
    cheap_conf: Sequence[float],
    llm_pred: Sequence[str | None],
    score: Callable[[list[str | None]], dict[str, float]],
    cheap_cost: float,
    llm_cost: float,
    thresholds: Sequence[float] = THRESHOLDS,
) -> pl.DataFrame:
    """Cost (USD per million products) and scores of the cascade at each threshold."""
    rows = []
    for t in thresholds:
        pred, routed = cascade(cheap_pred, cheap_conf, llm_pred, t)
        rows.append(
            {
                "threshold": t,
                "routed": routed,
                "usd_per_million": 1e6 * cascade_cost(routed, cheap_cost, llm_cost),
                **score(pred),
            }
        )
    return pl.DataFrame(rows)
