import numpy as np
import pytest

from cpilens.econ import simulate as s
from cpilens.paths import RAW


def synthetic(months: int = 24, seed: int = 0) -> s.Official:
    rng = np.random.default_rng(seed)
    labels = ["11110", "11122", "11600", "42100"]
    R = np.cumprod(1 + rng.normal(0.002, 0.01, (months, 4)), axis=0)
    r_oos = np.cumprod(1 + rng.normal(0.001, 0.005, months))
    w = np.array([2.0, 1.0, 3.0, 1.5])
    # aggregates: in-scope items plus an outside-scope remainder
    rest = np.cumprod(1 + rng.normal(0.001, 0.003, months))

    def level(mask: np.ndarray, w_rest: float) -> tuple[np.ndarray, float]:
        W = w[mask].sum() + w_rest
        rel = (R[:, mask] @ w[mask] + w_rest * rest) / W
        return np.concatenate([[100.0], 100.0 * rel]), W

    food = np.array([True, True, True, False])
    core = np.array([False, True, True, True])
    allm = np.ones(4, bool)
    agg, aw = {}, {}
    for name, mask, wr in (("headline", allm, 92.5), ("food", food, 30.0), ("core", core, 50.0)):
        agg[name], aw[name] = level(mask, wr)
    return s.Official(
        labels=labels,
        dates=list(range(months)),
        R=R,
        r_oos=r_oos,
        w=w,
        agg_index=agg,
        agg_w=aw,
        members={"headline": allm, "food": food, "core": core},
    )


def perfect(k: int) -> np.ndarray:
    P = np.zeros((k + 1, k + 2))
    P[np.arange(k + 1), np.arange(k + 1)] = 1.0
    return P


def test_perfect_classifier_reproduces_official_exactly():
    off = synthetic()
    n = np.full(5, 0.2)
    est = s.measured_index(off, s.measured_relatives(off, perfect(4), n)[0])
    for a in off.agg_index:
        np.testing.assert_allclose(est[a], off.agg_index[a], rtol=0, atol=1e-12)
    err = s.simulate(off, perfect(4), n)
    assert err["rmse_bp"].abs().max() < 1e-8
    assert s.decompose(off, perfect(4), n).height == 0


def test_random_classifier_gives_large_errors():
    off = synthetic()
    P = np.full((5, 6), 1 / 6)
    err = s.simulate(off, P, np.full(5, 0.2))
    assert err.filter(err["kind"] == "yoy")["rmse_bp"].max() > 10


def test_decomposition_adds_up_to_index_error():
    off = synthetic(seed=3)
    rng = np.random.default_rng(1)
    P = np.vstack([rng.dirichlet(np.r_[np.eye(5)[i] * 20 + 0.5, 0.5]) for i in range(5)])
    n = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    rt, _ = s.measured_relatives(off, P, n)
    est = s.measured_index(off, rt)
    for a in ("headline", "food"):
        d = s.decompose(off, P, n, a)
        level_err_bp = 1e4 * (est[a][1:] - off.agg_index[a][1:]) / off.agg_index[a][0]
        assert d["mean_bp"].sum() == pytest.approx(level_err_bp.mean(), abs=1e-9)


def test_abstention_renormalises_within_group_and_oos_leak():
    off = synthetic()
    P = perfect(4)
    P[1] = 0.0
    P[1, -1] = 1.0  # all 11122 products abstained -> 11122 takes its group peer 11110
    rt, _ = s.measured_relatives(off, P, np.full(5, 0.2))
    np.testing.assert_allclose(rt[:, 1], off.R[:, 0])
    P = perfect(4)
    P[4] = 0.0
    P[4, 3] = 1.0  # OOS products leak into 42100
    rt, phi = s.measured_relatives(off, P, np.full(5, 0.2))
    assert phi[4, 3] == pytest.approx(0.5)
    np.testing.assert_allclose(rt[:, 3], 0.5 * off.R[:, 3] + 0.5 * off.r_oos)


def test_confusion_draws_are_seeded_row_stochastic():
    c = np.array([[8, 2, 0], [0, 0, 0]], float)
    pr = s.eb_prior(c)
    d1 = s.confusion_draws(c, 50, pr, seed=4)
    assert np.array_equal(d1, s.confusion_draws(c, 50, pr, seed=4))
    np.testing.assert_allclose(d1.sum(2), 1.0)
    # rho = 0.8; row 0's errors all went to column 1
    assert pr[0].tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert s.point_confusion(c, pr)[0, 0] == pytest.approx(8.8 / 11)
    assert np.all(d1[:, 0, 2] == 0)


def test_eb_prior_keeps_a_perfect_classifier_perfect():
    c = np.zeros((3, 4))
    c[0, 0], c[1, 1] = 5, 1  # row 2 unobserved
    pr = s.eb_prior(c)
    np.testing.assert_allclose(s.point_confusion(c, pr), np.eye(3, 4))
    np.testing.assert_allclose(
        s.confusion_draws(c, 5, pr), np.broadcast_to(np.eye(3, 4), (5, 3, 4))
    )


@pytest.mark.skipif(not (RAW / "tpso" / "api").exists(), reason="raw TPSO data not present")
def test_real_official_perfect_classifier():
    off = s.official()
    k = len(off.labels)
    err = s.simulate(off, perfect(k), np.full(k + 1, 1 / (k + 1)))
    assert err["max_abs_bp"].max() < 1e-6
    assert len(off.dates) == off.R.shape[0] >= 21


def test_counts_from_predictions():
    c, n = s.counts_from_predictions(
        ["a", "b"],
        ["a", "a", "b", "OOS", "UNK"],
        ["a", None, "OOS", "b", "a"],
        [0.5, 0.5, 1.0, 1.0, 1.0],
    )
    assert c.tolist() == [[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    assert n.tolist() == pytest.approx([4 / 6, 1 / 6, 1 / 6])


def test_product_noise_shrinks_with_products():
    off = synthetic()
    rng = np.random.default_rng(1)
    flow = np.array([0.4, 0.3, 0.3, 0.0])
    assert np.allclose(s.product_noise(off, flow, 1e3, 0.0, rng), 1.0)
    small = np.log(s.product_noise(off, flow, 1e2, 0.1, rng))[-1, :3].std()
    big = np.log(s.product_noise(off, flow, 1e6, 0.1, rng))[-1, :3].std()
    assert big < small
    assert np.all(s.product_noise(off, flow, 1e2, 0.1, rng)[:, 3] == 1.0)  # no products
    rt, _ = s.measured_relatives(off, perfect(4), np.full(5, 0.2), np.ones_like(off.R))
    assert np.allclose(rt, off.R)


def test_flow_type():
    from cpilens.econ.run import flow_type

    assert flow_type("11110", "11122") == "within_group"
    assert flow_type("11110", "11600") == "within_division"
    assert flow_type("11110", "42100") == "cross_division"
    assert flow_type("OOS", "11600") == "oos_leak"
    assert flow_type("EMPTY", "11600") == "empty_item"


def test_run_econ_on_synthetic_labels(tmp_path):
    """End to end on the real prediction files with fake labels (Haiku's own answers)."""
    import shutil

    import polars as pl

    from cpilens.econ.run import run_econ
    from cpilens.label import store
    from cpilens.paths import RESULTS

    if not (RESULTS / "cascade_dev.csv").exists() or not (RAW / "tpso").exists():
        pytest.skip("needs cpilens models outputs and raw TPSO data")
    shutil.copy(store.GOLD / "sample.csv", tmp_path / "sample.csv")
    smp = pl.read_csv(tmp_path / "sample.csv", schema_overrides={"barcode": pl.Utf8})
    h = pl.read_parquet(RESULTS / "preds/llm/claude-haiku-5-5__low.parquet").filter(
        pl.col("eval_set") == "gold"
    )
    lab = dict(zip(h["id"], h["pred"].fill_null("UNK"), strict=True))
    for r in smp.iter_rows(named=True):
        store.append_label(
            1,
            {
                "order": r["order"],
                "barcode": r["barcode"],
                "label": lab["off:" + r["barcode"]],
                "unsure": "False",
                "seconds": "1",
                "labelled_utc": "2026-10-27T00:00:00Z",
            },
            tmp_path,
        )
    out = tmp_path / "out"
    out.mkdir()
    run_econ(tmp_path, RESULTS, out, n_draws=10, weights=("equal",))
    e = pl.read_csv(out / "rq3_errors.csv")
    haiku = e.filter(
        (pl.col("method") == "claude-haiku-5-5")
        & (pl.col("variant") == "full")
        & (pl.col("aggregate") == "headline")
        & (pl.col("kind") == "yoy")
    )
    cov = e.filter(
        (pl.col("variant") == "coverage_only")
        & (pl.col("aggregate") == "headline")
        & (pl.col("kind") == "yoy")
    )
    # scored against its own answers, Haiku is a perfect classifier: only coverage error
    assert haiku["rmse_bp"][0] == pytest.approx(cov["rmse_bp"][0], abs=1e-9)
    assert haiku["rmse_bp_hi"][0] == pytest.approx(cov["rmse_bp"][0], abs=1e-9)
    assert haiku["rmse_bp_lo"][0] <= haiku["rmse_bp_hi"][0]
    ser = pl.read_csv(out / "rq3_series.csv")
    assert ser.filter(pl.col("kind") == "yoy")["date"].n_unique() == 10
