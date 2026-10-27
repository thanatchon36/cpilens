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


def official(method: str = "equal") -> Official:
    lw = cpi_label_weights(method)
    labels = list(lw)
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
    off: Official, P: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """R~ (months x labels) and phi (true rows incl. OOS x labels).

    ``P``: (L+1) x (L+2) row-stochastic, rows = labels + OOS, columns = labels + OOS +
    ABSTAIN. ``n``: universe shares of the L+1 true rows.
    """
    k = len(off.labels)
    mass = n[:, None] * P[:, :k]  # true row -> predicted in-scope label
    inflow = mass.sum(0)
    phi = np.divide(mass, inflow, out=np.zeros_like(mass), where=inflow > 0)
    r_true = np.column_stack([off.R, off.r_oos])  # months x (L+1)
    rt = r_true @ phi
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


def simulate(off: Official, P: np.ndarray, n: np.ndarray) -> pl.DataFrame:
    rt, _ = measured_relatives(off, P, n)
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


def confusion_draws(
    counts: np.ndarray, n_draws: int, alpha: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Posterior draws of a row-stochastic confusion matrix: each row ~ Dirichlet(counts +
    alpha / C). Rows without observations draw from the flat prior and should carry no
    universe mass (n_k = 0) in the simulation."""
    rng = np.random.default_rng(seed)
    a = counts + alpha / counts.shape[1]
    return np.stack([np.vstack([rng.dirichlet(row) for row in a]) for _ in range(n_draws)])


def point_confusion(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Posterior-mean confusion matrix (the smoothed point estimate)."""
    a = counts + alpha / counts.shape[1]
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
