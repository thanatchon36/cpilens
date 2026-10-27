"""RQ3 on gold: each method's confusion matrix -> basis-point errors in measured inflation.

For every method the gold confusion counts give a smoothed point confusion matrix and
``N_DRAWS`` Dirichlet posterior draws; universe shares of the true labels are the
Horvitz-Thompson estimates from the gold sample. Variants:

- ``full``: the method's confusion matrix.
- ``coverage_only``: a perfect classifier on the same universe. It is not exact, because
  labels with no gold products are empty and renormalised within their group. ``full``
  minus ``coverage_only`` is the part due to classification.
- ``noise_s{sigma}``: ``full`` plus product-level sampling noise with an *assumed* annual
  dispersion ``sigma``. The universe size is the Horvitz-Thompson estimate (sum of 1/pi).
  The estimate is the draw median, since one noise realisation is not a point estimate.

Smoothing uses an empirical-Bayes prior (``simulate.eb_prior``), so a perfect classifier
has zero error in every draw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cpilens.econ import simulate as s
from cpilens.evaluate.gold import gold_table
from cpilens.evaluate.metrics import group_of
from cpilens.evaluate.run import methods
from cpilens.label import store
from cpilens.paths import RESULTS

N_DRAWS = 1000
SEED = 20261027
ALPHA = 1.0
SIGMAS = (0.05, 0.10)
STATS = ("mean_bp", "rmse_bp", "max_abs_bp")


def flow_type(true: str, pred: str) -> str:
    if true == "EMPTY":
        return "empty_item"
    if true == "OOS":
        return "oos_leak"
    if group_of(true) == group_of(pred):
        return "within_group"
    return "within_division" if true[0] == pred[0] else "cross_division"


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
    gold: Path = store.GOLD,
    results: Path = RESULTS,
    out: Path = RESULTS,
    n_draws: int = N_DRAWS,
    weights: tuple[str, ...] = ("equal", "nnls"),
) -> None:
    g = gold_table(gold)
    if g.height == 0:
        raise SystemExit("no gold labels yet: label pass 1 first (cpilens label --pass 1)")
    g = g.filter(pl.col("label") != "UNK")
    y = g["label"].to_list()
    pi = g["incl_prob"].to_list()
    n_universe = float((1 / g["incl_prob"]).sum())
    ms = methods(results)
    rng = np.random.default_rng(SEED)
    err_rows, series_rows, dec_rows = [], [], []
    for wm in weights:
        off = s.official(wm)
        perfect_pred: list[str | None] = list(y)
        _, n = s.counts_from_predictions(off.labels, y, perfect_pred, pi)
        k = len(off.labels)
        P_perfect = np.zeros((k + 1, k + 2))
        P_perfect[np.arange(k + 1), np.arange(k + 1)] = 1.0
        cov = s.simulate(off, P_perfect, n).with_columns(
            pl.lit("perfect").alias("method"), pl.lit("coverage_only").alias("variant")
        )
        err_rows.append(cov.with_columns(pl.lit(wm).alias("weights")))
        for name, (p, _meta) in ms.items():
            pred = g.select("id").join(p, on="id", how="left")["pred"].to_list()
            counts, n = s.counts_from_predictions(off.labels, y, pred, pi)
            prior = s.eb_prior(counts, ALPHA)
            P = s.point_confusion(counts, prior)
            draws = s.confusion_draws(counts, n_draws, prior, seed=SEED)
            point = s.simulate(off, P, n)
            tab = _summarise(point, [s.simulate(off, d, n) for d in draws])
            err_rows.append(
                tab.with_columns(
                    pl.lit(name).alias("method"),
                    pl.lit("full").alias("variant"),
                    pl.lit(wm).alias("weights"),
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
                        pl.lit(wm).alias("weights"),
                    )
                )
            if wm != "equal":
                continue
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
            for a in ("headline", "food", "core"):
                dec = s.decompose(off, P, n, a)
                if dec.height:
                    dec_rows.append(
                        dec.with_columns(
                            pl.lit(name).alias("method"),
                            pl.lit(a).alias("aggregate"),
                            pl.struct("true", "pred")
                            .map_elements(
                                lambda r: flow_type(r["true"], r["pred"]), return_dtype=pl.Utf8
                            )
                            .alias("flow_type"),
                        )
                    )
    pl.concat(err_rows, how="diagonal_relaxed").write_csv(out / "rq3_errors.csv")
    pl.DataFrame(series_rows).write_csv(out / "rq3_series.csv")
    if dec_rows:
        pl.concat(dec_rows).write_csv(out / "rq3_decomposition.csv")
