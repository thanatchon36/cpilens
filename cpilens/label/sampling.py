"""Gold-set sample: Thai products with a Thai-script name, stratified by CPI group and weight.

Strata are CPI weight cells of the silver label (``none`` when the crosswalk gives no label,
``OOS`` when it says out of scope). Allocation: a floor per stratum, then half of the
remainder in proportion to stratum size and half in proportion to the stratum's CPI weight
(``none``/``OOS`` get the size half only). Inclusion probabilities are stored so that gold
estimates can be design-weighted back to the frame. A random subset is labelled twice blind.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SEED = 20261027
N_GOLD = 600
N_DOUBLE = 150
FLOOR = 10


def frame(products: pl.DataFrame, leaf_cells: pl.DataFrame) -> pl.DataFrame:
    """Thai-market products with a Thai-script name, one row per barcode, with stratum."""
    cell_of = dict(zip(leaf_cells["leaf"], leaf_cells["cell"], strict=True))
    f = (
        products.filter((pl.col("market") == "thailand") & pl.col("name_th").is_not_null())
        .sort("source", "barcode")
        .unique("barcode", keep="first", maintain_order=True)
    )
    return f.with_columns(
        pl.when(pl.col("silver_label").is_null())
        .then(pl.lit("none"))
        .when(pl.col("silver_label") == "OOS")
        .then(pl.lit("OOS"))
        .otherwise(pl.col("silver_label").replace_strict(cell_of, default=None))
        .alias("stratum")
    )


def allocate(
    sizes: dict[str, int], weights: dict[str, float], n: int = N_GOLD, floor: int = FLOOR
) -> dict[str, int]:
    """Sample size per stratum (never above the stratum size); sums to ``n``."""
    alloc = {h: min(floor, s) for h, s in sizes.items()}
    rest = n - sum(alloc.values())
    tot_n, tot_w = sum(sizes.values()), sum(weights.get(h, 0.0) for h in sizes)
    share = {h: 0.5 * s / tot_n + 0.5 * weights.get(h, 0.0) / tot_w for h, s in sizes.items()}
    for h in sizes:
        alloc[h] = min(sizes[h], alloc[h] + int(rest * share[h]))
    # Hand out what rounding and caps left, largest remaining room first.
    while (left := n - sum(alloc.values())) > 0:
        room = sorted((h for h in sizes if alloc[h] < sizes[h]), key=lambda h: (-share[h], h))
        if not room:
            raise ValueError("frame smaller than the requested sample")
        for h in room[:left]:
            alloc[h] += 1
    return alloc


def draw(
    fr: pl.DataFrame,
    cell_weights: dict[str, float],
    n: int = N_GOLD,
    n_double: int = N_DOUBLE,
    seed: int = SEED,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = dict(fr.group_by("stratum").len().iter_rows())
    alloc = allocate(sizes, {h: w for h, w in cell_weights.items() if h in sizes}, n)
    parts = []
    for h in sorted(alloc):
        g = fr.filter(pl.col("stratum") == h).sort("barcode")
        idx = rng.choice(g.height, size=alloc[h], replace=False)
        parts.append(g[np.sort(idx)].with_columns(pl.lit(alloc[h] / sizes[h]).alias("incl_prob")))
    s = pl.concat(parts)
    order = rng.permutation(s.height)
    s = s.with_columns(pl.Series("order", np.argsort(order) + 1))
    double = set(rng.choice(s.height, size=n_double, replace=False).tolist())
    return s.with_columns(pl.Series("double", [i in double for i in range(s.height)])).sort("order")
