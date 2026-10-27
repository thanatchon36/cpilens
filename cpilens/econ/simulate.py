"""RQ3 materiality simulator: what a classifier's confusion matrix does to measured inflation.

Setting. The Thai CPI is a Lowe index whose price reference is the link month L (Dec 2567):
for any aggregate A with members i, I_{A,t} / I_{A,L} = sum_i (w_i / W_A) R_{i,t}, where
R_{i,t} = I_{i,t} / I_{i,L} and w are price-updated weights. Official item relatives are taken
as the true item price movements.

Classification error. Products of true item k occur with universe share n_k. A classifier
with row-stochastic confusion P (true k -> predicted j, or abstain) sends mass n_k P_kj to
item j, so item j is measured from the mix

    R~_{j,t} = sum_k phi_kj R_{k,t},    phi_kj = n_k P_kj / sum_k' n_k' P_k'j.

Abstained products (and products sent to OOS) are dropped. If nothing flows into j, its
weight is renormalised within its CPI group, i.e. j takes the weighted mean of the measured
relatives of the other in-scope items of the group; if the whole group is empty, j is held
at its official relative. True OOS products that leak into an item move with the official
non-food aggregate (80000), a stated assumption. Items outside the scope are perfectly
measured, so

    I^_{A,t} = I_{A,t} + I_{A,L} sum_{j in A, in scope} (w_j / W_A) (R~_{j,t} - R_{j,t}).

A perfect classifier gives R~ = R and reproduces every official series exactly. The error
decomposes exactly into pairs: R~_j - R_j = sum_k phi_kj (R_k - R_j), so the contribution
of (k -> j) is weight x misallocation mass x divergence of price movements.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import polars as pl

from cpilens.data.tpso import load_index
from cpilens.evaluate.metrics import cpi_label_weights, group_of
from cpilens.paths import TAXONOMY
from cpilens.taxonomy.weights import PRICE_REF_DATE, relatives

OOS_PROXY = "80000"
FRESH = ("111", "112", "113", "114")  # 91000 = 11100 + 11200 + 11300 + 11400 (verified)
AGGREGATES = {"headline": "00000", "food": "10000", "core": "93000"}


@dataclass
class Official:
    labels: list[str]  # in-scope labels, column order of R
    dates: list[object]  # months after L
    R: np.ndarray  # months x labels, true relatives
    r_oos: np.ndarray  # months, relative of the OOS proxy
    w: np.ndarray  # leaf weights (% of headline)
    agg_index: dict[str, np.ndarray]  # aggregate -> index levels, L first then months
    agg_w: dict[str, float]  # aggregate -> official price-updated weight
    members: dict[str, np.ndarray]  # aggregate -> boolean mask over labels


def official(method: str = "equal", exclude: Collection[str] = ()) -> Official:
    """The official series; ``exclude``: in-scope items held at official values (not simulated)."""
    lw = cpi_label_weights(method)
    labels = [k for k in lw if k not in set(exclude)]
    idx = load_index()
    codes = [*labels, OOS_PROXY, *AGGREGATES.values()]
    rel = relatives(idx, codes)
    wide = (
        idx.filter(
            pl.col("code").is_in(list(AGGREGATES.values())) & (pl.col("date") >= PRICE_REF_DATE)
        )
        .pivot(on="code", index="date", values="index")
        .sort("date")
        .drop_nulls()
    )
    wide = wide.head(rel.shape[0] + 1)
    wtab = dict(
        pl.read_csv(TAXONOMY / "thcpi_weights_2566.csv", schema_overrides={"code": pl.Utf8})
        .select("code", "w_pr")
        .rows()
    )
    food = np.array([lab.startswith("1") for lab in labels])
    fresh = np.array([lab[:3] in FRESH for lab in labels])
    return Official(
        labels=labels,
        dates=wide["date"].to_list()[1:],
        R=rel[:, : len(labels)],
        r_oos=rel[:, len(labels)],
        w=np.array([lw[k] for k in labels]),
        agg_index={a: wide[c].to_numpy() for a, c in AGGREGATES.items()},
        agg_w={a: float(wtab[c]) for a, c in AGGREGATES.items()},
        members={
            "headline": np.ones(len(labels), bool),
            "food": food,
            "core": ~fresh,
        },  # no in-scope label is energy
    )


def measured_relatives(
    off: Official, P: np.ndarray, n: np.ndarray, noise: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """R~ (months x labels) and phi (true rows incl. OOS x labels).

    ``P``: (L+1) x (L+2) row-stochastic, rows = labels + OOS, columns = labels + OOS +
    ABSTAIN. ``n``: universe shares of the L+1 true rows. ``noise``: optional months x
    labels multiplicative factor (product-level sampling noise, see ``product_noise``),
    applied before empty items are renormalised.
    """
    k = len(off.labels)
    mass = n[:, None] * P[:, :k]  # true row -> predicted in-scope label
    inflow = mass.sum(0)
    phi = np.divide(mass, inflow, out=np.zeros_like(mass), where=inflow > 0)
    r_true = np.column_stack([off.R, off.r_oos])  # months x (L+1)
    rt = r_true @ phi
    if noise is not None:
        rt = rt * noise
    empty = inflow <= 0
    if empty.any():
        groups = np.array([group_of(lab) for lab in off.labels])
        for j in np.flatnonzero(empty):
            peers = (groups == groups[j]) & ~empty
            if peers.any():
                rt[:, j] = rt[:, peers] @ off.w[peers] / off.w[peers].sum()
            else:
                rt[:, j] = off.R[:, j]
    return rt, phi


def inflow(off: Official, P: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Universe share of products measured in each in-scope label."""
    return (n[:, None] * P[:, : len(off.labels)]).sum(0)


def product_noise(
    off: Official, flow: np.ndarray, n_products: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative noise from measuring item j with a finite set of products.

    Each product's log price deviates from its item's by a random walk with annual standard
    deviation ``sigma``; item j averages ``n_products * flow_j`` products, so its log
    relative carries a random walk with monthly variance sigma^2 / (12 n_products flow_j).
    An item with any inflow is measured from at least one product.
    """
    m, k = off.R.shape
    n_j = np.maximum(n_products * flow, 1.0)
    sd = np.where(flow > 0, sigma / np.sqrt(12 * n_j), 0.0)
    return np.exp(np.cumsum(rng.normal(size=(m, k)) * sd, axis=0))


def measured_index(off: Official, rt: np.ndarray) -> dict[str, np.ndarray]:
    """Measured aggregate index levels (L first), outside-scope items held at official."""
    out = {}
    for a, lev in off.agg_index.items():
        mbr = off.members[a]
        dev = (rt[:, mbr] - off.R[:, mbr]) @ off.w[mbr] / off.agg_w[a]
        out[a] = np.concatenate([[lev[0]], lev[1:] + lev[0] * dev])
    return out


def inflation_errors_bp(off: Official, est: dict[str, np.ndarray]) -> pl.DataFrame:
    """Per aggregate: month-on-month and year-on-year errors in basis points."""
    rows = []
    for a, lev in off.agg_index.items():
        m = est[a]
        mom = 1e4 * (m[1:] / m[:-1] - lev[1:] / lev[:-1])
        yoy = 1e4 * (m[12:] / m[:-12] - lev[12:] / lev[:-12])
        for kind, e in (("mom", mom), ("yoy", yoy)):
            rows.append(
                {
                    "aggregate": a,
                    "kind": kind,
                    "mean_bp": float(e.mean()),
                    "rmse_bp": float(np.sqrt((e**2).mean())),
                    "max_abs_bp": float(np.abs(e).max()),
                    "n_months": len(e),
                }
            )
    return pl.DataFrame(rows)


def simulate(
    off: Official, P: np.ndarray, n: np.ndarray, noise: np.ndarray | None = None
) -> pl.DataFrame:
    rt, _ = measured_relatives(off, P, n, noise)
    return inflation_errors_bp(off, measured_index(off, rt))


def decompose(
    off: Official,
    P: np.ndarray,
    n: np.ndarray,
    aggregate: str = "headline",
    true_names: list[str] | None = None,
) -> pl.DataFrame:
    """Contribution of each (true k -> predicted j) flow to the index-level error of an
    aggregate, in basis points of its relative since L: mean over months and mean absolute.
    Exact for items with inflow; renormalised (empty) items are reported as ``k = EMPTY``."""
    rt, phi = measured_relatives(off, P, n)
    mbr = off.members[aggregate]
    r_true = np.column_stack([off.R, off.r_oos])
    rows_names = true_names or [*off.labels, "OOS"]
    scale = 1e4 * off.w / off.agg_w[aggregate]
    rows = []
    for j in np.flatnonzero(mbr):
        if phi[:, j].sum() == 0:
            c = scale[j] * (rt[:, j] - off.R[:, j])
            if np.abs(c).max() > 0:
                rows.append(
                    {
                        "true": "EMPTY",
                        "pred": off.labels[j],
                        "mean_bp": float(c.mean()),
                        "mean_abs_bp": float(np.abs(c).mean()),
                    }
                )
            continue
        for k in np.flatnonzero(phi[:, j] > 0):
            if rows_names[k] == off.labels[j]:
                continue
            c = scale[j] * phi[k, j] * (r_true[:, k] - off.R[:, j])
            rows.append(
                {
                    "true": rows_names[k],
                    "pred": off.labels[j],
                    "mean_bp": float(c.mean()),
                    "mean_abs_bp": float(np.abs(c).mean()),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "true": pl.Utf8,
                "pred": pl.Utf8,
                "mean_bp": pl.Float64,
                "mean_abs_bp": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("mean_abs_bp", descending=True)


def eb_prior(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Empirical-Bayes Dirichlet prior for each confusion row, worth ``alpha`` products.

    Row k's prior mean puts the method's pooled accuracy rho on its own column k and spreads
    1 - rho over the other columns in proportion to where the method's errors go overall
    (pooled over rows, column k excluded). Sparse rows are shrunk towards the method's own
    behaviour, not towards a flat matrix: a flat prior would send mass to arbitrary items and
    manufacture inflation error even for a perfect classifier, whereas a perfect classifier
    (rho = 1) keeps an identity prior and so no error.
    """
    r = counts.shape[0]
    diag = counts[np.arange(r), np.arange(r)]
    total = counts.sum()
    rho = diag.sum() / total if total > 0 else 1.0
    wrong = counts.copy()
    wrong[np.arange(r), np.arange(r)] = 0
    err_cols = wrong.sum(0)
    prior = np.zeros_like(counts, dtype=float)
    for k in range(r):
        q = err_cols.copy()
        q[k] = 0
        if q.sum() > 0:
            prior[k] = alpha * (1 - rho) * q / q.sum()
            prior[k, k] = alpha * rho
        else:
            prior[k, k] = alpha
    return prior


def confusion_draws(
    counts: np.ndarray, n_draws: int, prior: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Posterior draws of a row-stochastic confusion matrix, row k ~ Dirichlet(counts_k +
    prior_k). Columns with zero parameter are exactly zero (gamma representation)."""
    rng = np.random.default_rng(seed)
    a = counts + prior
    pos = a > 0
    g = np.zeros((n_draws, *a.shape))
    g[:, pos] = rng.gamma(a[pos], size=(n_draws, int(pos.sum())))
    return g / g.sum(2, keepdims=True)


def point_confusion(counts: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Posterior-mean confusion matrix (the smoothed point estimate)."""
    a = counts + prior
    return a / a.sum(1, keepdims=True)


def counts_from_predictions(
    labels: list[str], y: list[str], yhat: list[str | None], incl_prob: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Confusion counts (rows labels + OOS; columns labels + OOS + ABSTAIN) and
    Horvitz-Thompson universe shares of the true rows. Items whose true label is not a
    label or OOS (e.g. UNK) are skipped."""
    rows = [*labels, "OOS"]
    pos = {k: i for i, k in enumerate(rows)}
    counts = np.zeros((len(rows), len(rows) + 1))
    mass = np.zeros(len(rows))
    inv = np.ones(len(y)) if incl_prob is None else 1 / np.asarray(incl_prob, float)
    for t, p, v in zip(y, yhat, inv, strict=True):
        if t not in pos:
            continue
        counts[pos[t], len(rows) if p is None or p not in pos else pos[p]] += 1
        mass[pos[t]] += v
    return counts, mass / mass.sum()
