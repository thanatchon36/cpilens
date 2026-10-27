"""RQ3 on the reference set: each method's confusion matrix -> basis-point errors in measured
inflation.

For every method the confusion counts on the held-out official specifications give a
smoothed point confusion matrix and ``N_DRAWS`` Dirichlet posterior draws. The product
universe is an assumption, stated per ``universe``:

- ``cpi`` (primary): products arrive in proportion to CPI expenditure weights, with the
  weight outside the evaluated items as out-of-scope products;
- ``list``: products arrive in proportion to the office's specification list, i.e. the
  shares of the reference set.

In-scope items without official specifications cannot be evaluated; they are held at their
official values, like out-of-scope items. Variants:

- ``full``: the method's confusion matrix.
- ``coverage_only``: a perfect classifier on the same universe (a check: zero error).
- ``noise_s{sigma}``: ``full`` plus product-level sampling noise with an *assumed* annual
  dispersion ``sigma``, the reference set's size standing in for the number of products.
  The estimate is the draw median, since one noise realisation is not a point estimate.

Smoothing uses an empirical-Bayes prior (``simulate.eb_prior``), so a perfect classifier
has zero error in every draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.econ import simulate as s
from cpilens.evaluate.gold import reference_table
from cpilens.evaluate.metrics import cpi_label_weights, group_of
from cpilens.evaluate.run import is_headline, methods
from cpilens.paths import RESULTS

N_DRAWS = 1000
SEED = 20261027
ALPHA = 1.0
SIGMAS = (0.05, 0.10)
UNIVERSES = ("cpi", "list")  # the first is the primary one (series and decompositions)
STATS = ("mean_bp", "rmse_bp", "max_abs_bp")
MIN_FLOW_BP = 0.01  # flows kept in the detailed decomposition (mean |contribution|)


def flow_type(true: str, pred: str) -> str:
    if true == "EMPTY":
        return "empty_item"
    if true == "OOS":
        return "oos_leak"
    if group_of(true) == group_of(pred):
        return "within_group"
    return "within_division" if true[0] == pred[0] else "cross_division"


def universe_shares(
    off: s.Official, lw: dict[str, float], y: list[str], pi: list[float], universe: str
) -> np.ndarray:
    """Shares of the true rows (``off.labels``, then OOS) under an assumed product universe."""
    if universe == "list":  # the office's list: design-weighted shares of the reference set
        return s.counts_from_predictions(off.labels, y, list(y), pi)[1]
    w = np.array([lw[k] for k in off.labels])  # % of headline
    n = np.append(w, 100.0 - sum(lw.values()))  # the rest of the basket is out of scope
    return n / n.sum()


def _summarise(point: pl.DataFrame | None, draws: list[pl.DataFrame]) -> pl.DataFrame:
    """Point estimate (or, if None, the draw median) with 95% draw intervals."""
    allc = pl.concat(draws)
    if point is None:
        point = allc.group_by("aggregate", "kind").agg(
            *[pl.col(c).median() for c in STATS], pl.col("n_months").first()
        )
    d = allc.group_by("aggregate", "kind").agg(
        *[
            pl.col(c).quantile(q).alias(f"{c}_{tag}")
            for c in STATS
            for q, tag in ((0.025, "lo"), (0.975, "hi"))
        ]
    )
    return point.join(d, on=["aggregate", "kind"])


def _series(est: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    lag = 1 if kind == "mom" else 12
    return {a: 1e4 * (v[lag:] / v[:-lag] - 1) for a, v in est.items()}


def run_econ(
    ref: pl.DataFrame | None = None,
    results: Path = RESULTS,
    out: Path = RESULTS,
    n_draws: int = N_DRAWS,
    weights: tuple[str, ...] = ("equal", "nnls"),
    universes: tuple[str, ...] = UNIVERSES,
) -> None:
    g = reference_table() if ref is None else ref
    g = g.filter(pl.col("label") != "UNK")
    y = g["label"].to_list()
    observed = set(y)
    pi = g["incl_prob"].to_list()
    n_universe = float((1 / g["incl_prob"]).sum())
    ms = methods(results)
    rng = np.random.default_rng(SEED)
    err_rows, series_rows, dec_rows, type_rows = [], [], [], []
    unobserved: list[str] = []
    for wm in weights:
        lw = cpi_label_weights(wm)
        unobserved = [k for k in lw if k not in observed]
        off = s.official(wm, exclude=unobserved)
        k = len(off.labels)
        for uni in universes:
            primary = uni == universes[0] and wm == "equal"
            n = universe_shares(off, lw, y, pi, uni)
            tag = {"weights": wm, "universe": uni}
            P_perfect = np.zeros((k + 1, k + 2))
            P_perfect[np.arange(k + 1), np.arange(k + 1)] = 1.0
            cov = s.simulate(off, P_perfect, n).with_columns(
                pl.lit("perfect").alias("method"), pl.lit("coverage_only").alias("variant")
            )
            err_rows.append(cov.with_columns(**{c: pl.lit(v) for c, v in tag.items()}))
            for name, (p, meta) in ms.items():
                pred = g.select("id").join(p, on="id", how="left")["pred"].to_list()
                counts, _ = s.counts_from_predictions(off.labels, y, pred, pi)
                prior = s.eb_prior(counts, ALPHA)
                P = s.point_confusion(counts, prior)
                draws = s.confusion_draws(counts, n_draws, prior, seed=SEED)
                point = s.simulate(off, P, n)
                tab = _summarise(point, [s.simulate(off, d, n) for d in draws])
                err_rows.append(
                    tab.with_columns(
                        pl.lit(name).alias("method"),
                        pl.lit("full").alias("variant"),
                        **{c: pl.lit(v) for c, v in tag.items()},
                    )
                )
                for sigma in SIGMAS:
                    sims = []
                    for d in draws:
                        nz = s.product_noise(off, s.inflow(off, d, n), n_universe, sigma, rng)
                        sims.append(s.simulate(off, d, n, nz))
                    tab = _summarise(None, sims)
                    err_rows.append(
                        tab.with_columns(
                            pl.lit(name).alias("method"),
                            pl.lit(f"noise_s{sigma:.2f}").alias("variant"),
                            **{c: pl.lit(v) for c, v in tag.items()},
                        )
                    )
                if not primary:
                    continue
                dec = pl.concat(
                    [
                        s.decompose(off, P, n, a).with_columns(pl.lit(a).alias("aggregate"))
                        for a in s.AGGREGATES
                    ]
                ).with_columns(
                    pl.lit(name).alias("method"),
                    pl.struct("true", "pred")
                    .map_elements(lambda r: flow_type(r["true"], r["pred"]), return_dtype=pl.Utf8)
                    .alias("flow_type"),
                )
                type_rows.append(
                    dec.group_by("method", "aggregate", "flow_type").agg(
                        pl.col("mean_bp").sum(),
                        pl.col("mean_abs_bp").sum(),
                        pl.len().alias("n_flows"),
                    )
                )
                if not is_headline(name, meta):
                    continue
                dec_rows.append(dec.filter(pl.col("mean_abs_bp") >= MIN_FLOW_BP))
                # monthly series with draw bands (figure: official vs measured inflation)
                est = s.measured_index(off, s.measured_relatives(off, P, n)[0])
                dr = [s.measured_index(off, s.measured_relatives(off, d, n)[0]) for d in draws]
                for kind in ("mom", "yoy"):
                    lag = 1 if kind == "mom" else 12
                    o = _series(off.agg_index, kind)
                    e = _series(est, kind)
                    band = {a: np.stack([_series(x, kind)[a] for x in dr]) for a in est}
                    for a in est:
                        lo, hi = np.quantile(band[a], [0.025, 0.975], axis=0)
                        for i, dt in enumerate(off.dates[lag - 1 :]):
                            series_rows.append(
                                {
                                    "method": name,
                                    "aggregate": a,
                                    "kind": kind,
                                    "date": dt,
                                    "official_bp": o[a][i],
                                    "measured_bp": e[a][i],
                                    "measured_lo_bp": lo[i],
                                    "measured_hi_bp": hi[i],
                                }
                            )
    meta = {
        "n_draws": n_draws,
        "alpha": ALPHA,
        "sigmas": SIGMAS,
        "n_universe": n_universe,
        "universes": list(universes),
        "unobserved_items": unobserved,
    }
    (out / "rq3_meta.json").write_text(json.dumps(meta, indent=2))
    pl.concat(err_rows, how="diagonal_relaxed").write_csv(out / "rq3_errors.csv")
    pl.DataFrame(series_rows).write_csv(out / "rq3_series.csv")
    pl.concat(dec_rows).write_csv(out / "rq3_decomposition.csv")
    pl.concat(type_rows).write_csv(out / "rq3_flow_types.csv")
