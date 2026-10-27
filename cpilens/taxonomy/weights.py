"""Leaf-level weights: official where published, estimated within official cells.

TPSO publishes weights for 29 aggregates only. We use the price-updated shares ``w_pr`` (see
``cpilens.data.tpso_release``), which add up the hierarchy. Leaves are partitioned into
*cells*: each official node O owns the leaves whose deepest official ancestor-or-self is O.
The cell weight is w_O minus the weights of the official nodes nested directly below O, so cell
weights sum to the official totals. Inside a cell, leaf weights are not public and are
estimated (``method``); they are always reported as a sensitivity, never as official.

Lowe identity used for estimation, with relatives R_t = I_t / I_L to the price reference L:
    R_cell,t = sum_j s_j R_j,t,   s_j >= 0,   sum_j s_j = 1,
and for a residual cell R_cell,t = (w_O R_O,t - sum_D w_D R_D,t) / w_cell.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import nnls

from cpilens.data.tpso_release import PRICE_REF
from cpilens.taxonomy.hierarchy import ROOT, Hierarchy

PRICE_REF_DATE = date(PRICE_REF[0] - 543, PRICE_REF[1], 1)


def _official_parent(h: Hierarchy, code: str, official: dict[str, float]) -> str:
    return next((a for a in h.ancestors(code) if a in official), ROOT)


def nested_official(h: Hierarchy, official: dict[str, float]) -> dict[str, list[str]]:
    """Official node -> official nodes directly nested below it (analytic codes excluded)."""
    out: dict[str, list[str]] = {}
    for c in official:
        if c in h.names and c != ROOT:
            out.setdefault(_official_parent(h, c, official), []).append(c)
    return out


def cells(h: Hierarchy, official: dict[str, float]) -> pl.DataFrame:
    """One row per leaf: its cell (deepest official ancestor-or-self) and the cell weight."""
    nested = nested_official(h, official)
    cell_w = {
        o: official[o] - sum(official[d] for d in nested.get(o, []))
        for o in [ROOT, *official]
        if o == ROOT or o in h.names
    }
    rows = [
        {"leaf": lf, "cell": lf if lf in official else _official_parent(h, lf, official)}
        for lf in h.leaves
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("cell").replace_strict(cell_w, return_dtype=pl.Float64).alias("cell_weight"),
        pl.col("cell").is_in(list(nested)).alias("residual_cell"),
    )


def relatives(index: pl.DataFrame, codes: list[str]) -> np.ndarray:
    """Months after the price reference x codes, as I_t / I_L (rows with any gap dropped)."""
    wide = (
        index.filter(pl.col("code").is_in(codes) & (pl.col("date") >= PRICE_REF_DATE))
        .pivot(on="code", index="date", values="index")
        .sort("date")
        .drop_nulls()
    )
    x = wide.select(codes).to_numpy()
    if wide["date"][0] != PRICE_REF_DATE:
        raise ValueError(f"price-reference month missing for {codes}")
    return x[1:] / x[0]


def cell_relative(
    index: pl.DataFrame, cell: str, official: dict[str, float], nested: list[str]
) -> np.ndarray:
    x = relatives(index, [cell, *nested])
    w = np.array([official[c] for c in [cell, *nested]])
    return (w[0] * x[:, 0] - x[:, 1:] @ w[1:]) / (w[0] - w[1:].sum())


def leaf_weights(
    h: Hierarchy,
    index: pl.DataFrame,
    official: dict[str, float],
    method: str = "nnls",
    penalty: float = 1e3,
) -> pl.DataFrame:
    """Leaf weights (% of all items). ``method``: "nnls" (sum-constrained NNLS) or "equal"."""
    nested = nested_official(h, official)
    out = []
    for (cell,), grp in cells(h, official).group_by("cell", maintain_order=True):
        leaves = sorted(grp["leaf"])
        if method == "equal" or len(leaves) == 1:
            s = np.full(len(leaves), 1 / len(leaves))
        elif method == "nnls":
            y = cell_relative(index, str(cell), official, nested.get(str(cell), []))
            x = relatives(index, leaves)
            n = min(len(y), len(x))
            a = np.vstack([x[-n:], penalty * np.ones(len(leaves))])
            s, _ = nnls(a, np.concatenate([y[-n:], [penalty]]))
            s = s / s.sum()
        else:
            raise ValueError(method)
        out += [
            {"leaf": lf, "cell": cell, "weight": grp["cell_weight"][0] * si, "method": method}
            for lf, si in zip(leaves, s, strict=True)
        ]
    return pl.DataFrame(out).sort("leaf")
