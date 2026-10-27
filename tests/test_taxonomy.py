"""Hierarchy and weights: synthetic checks (always run) and checks on committed tables."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from cpilens.paths import RAW, TAXONOMY
from cpilens.taxonomy.hierarchy import ROOT, Hierarchy, parent_code
from cpilens.taxonomy.weights import PRICE_REF_DATE, cells, leaf_weights

# 2566 division weights printed in TPSO's rebasing note (data/SOURCES.md: TPSO-NOTE).
NOTE_DIVISIONS = {
    "10000": 38.74,
    "20000": 2.13,
    "30000": 24.77,
    "40000": 6.43,
    "50000": 22.71,
    "60000": 3.99,
    "70000": 1.23,
}


def test_parent_code_skips_missing_levels() -> None:
    pool = {"00000", "10000", "11000", "11400", "11411"}
    assert parent_code("11411", pool) == "11400"
    assert parent_code("11400", pool) == "11000"
    assert parent_code("11500", pool) == "11000"
    assert parent_code("20000", pool) == ROOT
    assert parent_code(ROOT, pool) is None


def synthetic_index() -> tuple[pl.DataFrame, dict[str, float]]:
    """Two-division tree; parent indices are exact Lowe aggregates of known leaf weights."""
    rng = np.random.default_rng(0)
    leaf_w = {"11110": 6.0, "11120": 14.0, "11210": 10.0, "21000": 30.0, "22000": 40.0}
    tree = {
        "11100": ["11110", "11120"],
        "11200": ["11210"],
        "10000": ["11110", "11120", "11210"],
        "20000": ["21000", "22000"],
        ROOT: list(leaf_w),
    }
    n = 24
    rel = {c: np.cumprod(1 + rng.normal(0.003, 0.01, n)) for c in leaf_w}
    for p, ch in tree.items():
        rel[p] = sum(leaf_w[c] * rel[c] for c in ch) / sum(leaf_w[c] for c in ch)
    months = [
        date(
            PRICE_REF_DATE.year + (PRICE_REF_DATE.month + k - 1) // 12,
            (PRICE_REF_DATE.month + k - 1) % 12 + 1,
            1,
        )
        for k in range(n + 1)
    ]
    rows = [
        {"code": c, "name_th": c, "date": d, "index": 100.0 * (1 if k == 0 else r[k - 1])}
        for c, r in rel.items()
        for k, d in enumerate(months)
    ]
    official = {ROOT: 100.0, "10000": 30.0, "11100": 20.0, "20000": 70.0}
    return pl.DataFrame(rows), official


def test_cells_partition_official_totals() -> None:
    idx, official = synthetic_index()
    h = Hierarchy.from_index(idx)
    c = cells(h, official)
    assert c.unique("cell")["cell_weight"].sum() == pytest.approx(100.0)
    assert dict(zip(c["leaf"], c["cell"], strict=True))["11210"] == "10000"
    assert c.filter(pl.col("leaf") == "11210")["residual_cell"][0]


@pytest.mark.parametrize("method", ["equal", "nnls"])
def test_leaf_weights_sum_to_cells(method: str) -> None:
    idx, official = synthetic_index()
    lw = leaf_weights(Hierarchy.from_index(idx), idx, official, method)
    assert lw["weight"].sum() == pytest.approx(100.0)
    by_cell = lw.group_by("cell").agg(pl.col("weight").sum())
    assert dict(zip(by_cell["cell"], by_cell["weight"], strict=True))["11100"] == pytest.approx(20)


def test_nnls_recovers_known_weights() -> None:
    idx, official = synthetic_index()
    lw = leaf_weights(Hierarchy.from_index(idx), idx, official, "nnls")
    got = dict(zip(lw["leaf"], lw["weight"], strict=True))
    assert got["11110"] == pytest.approx(6.0, abs=0.05)
    assert got["21000"] == pytest.approx(30.0, abs=0.05)


def test_hierarchy_has_no_orphans_or_duplicates() -> None:
    idx, _ = synthetic_index()
    h = Hierarchy.from_index(idx)
    assert len(set(h.names)) == len(h.names)
    for c in h.names:
        assert c == ROOT or h.ancestors(c)[-1] == ROOT


# --- committed tables (no raw data needed) ---


def official_table() -> pl.DataFrame:
    return pl.read_csv(TAXONOMY / "thcpi_weights_2566.csv", schema_overrides={"code": pl.Utf8})


def test_price_updated_weights_add_up_the_hierarchy() -> None:
    w = dict(zip(*official_table().select("code", "w_pr").to_dict().values(), strict=True))
    groups = ["11100", "11200", "11300", "11400", "11500", "11600", "11700", "12000"]
    assert sum(w[c] for c in NOTE_DIVISIONS) == pytest.approx(100.0, abs=0.01)
    assert sum(w[c] for c in groups) == pytest.approx(w["10000"], abs=0.01)
    assert w["12300"] + w["12400"] + w["12500"] == pytest.approx(w["12000"], abs=0.01)
    assert w["91000"] + w["92000"] == pytest.approx(w["90000"], abs=0.01)
    assert w["90000"] + w["93000"] == pytest.approx(100.0, abs=0.01)


def test_base_weights_match_rebasing_note() -> None:
    w = dict(zip(*official_table().select("code", "w0").to_dict().values(), strict=True))
    for code, note in NOTE_DIVISIONS.items():
        assert w[code] == pytest.approx(note, abs=0.05), code
    assert w["91000"] == pytest.approx(17.43, abs=0.05)
    assert w["92000"] == pytest.approx(12.89, abs=0.05)
    assert official_table()["w0_spread"].max() < 0.02


def test_leaf_table_sums_to_official_cells() -> None:
    lw = pl.read_csv(
        TAXONOMY / "thcpi_leaf_weights_2566.csv",
        schema_overrides={"leaf": pl.Utf8, "cell": pl.Utf8},
    )
    assert lw["leaf"].n_unique() == lw.height
    for col in ["w_equal", "w_nnls"]:
        assert lw[col].min() >= 0
        assert lw[col].sum() == pytest.approx(100.0, abs=0.02)
        per = lw.group_by("cell").agg(pl.col(col).sum(), pl.first("cell_weight"))
        assert (per[col] - per["cell_weight"]).abs().max() < 1e-3


# --- raw public data (skipped when data/raw is absent, e.g. in CI smoke) ---

needs_raw = pytest.mark.skipif(not (RAW / "tpso" / "api").exists(), reason="raw TPSO data absent")


@needs_raw
def test_leaf_weights_reproduce_published_headline() -> None:
    from cpilens.data.tpso import load_index
    from cpilens.taxonomy.weights import relatives

    idx = load_index()
    lw = pl.read_csv(TAXONOMY / "thcpi_leaf_weights_2566.csv", schema_overrides={"leaf": pl.Utf8})
    r = relatives(idx, [*lw["leaf"], ROOT])
    for col, tol in [("w_nnls", 5e-4), ("w_equal", 5e-3)]:
        ww = lw[col].to_numpy()
        assert np.abs(r[:, :-1] @ ww / ww.sum() - r[:, -1]).max() < tol, col


@needs_raw
def test_release_rows_match_api_indices() -> None:
    from cpilens.data.tpso import load_index
    from cpilens.data.tpso_release import build_from_dir

    agg = pl.read_csv(TAXONOMY / "thcpi_weight_aggregates.csv", schema_overrides={"code": pl.Utf8})
    w, months = build_from_dir(RAW / "tpso" / "releases", load_index(), agg)
    assert len(months) >= 5 and w.height == agg.height


def test_coverage_table_partitions_each_division():
    from cpilens.taxonomy import coverage

    t = coverage.table()
    assert t.filter(pl.col("label") != "OOS").height > 0
    # at most one out-of-scope block per division (none where every leaf is in scope)
    assert (t.filter(pl.col("label") == "OOS").group_by("division").len()["len"] == 1).all()
    by = t.group_by("division").agg(
        pl.col("w_equal").sum(), pl.col("w_nnls").sum(), pl.col("division_weight").first()
    )
    assert (by["w_equal"] - by["division_weight"]).abs().max() < 0.01
    assert (by["w_nnls"] - by["division_weight"]).abs().max() < 0.01
