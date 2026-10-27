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
    d1 = s.confusion_draws(c, 50, seed=4)
    assert np.array_equal(d1, s.confusion_draws(c, 50, seed=4))
    np.testing.assert_allclose(d1.sum(2), 1.0)
    assert s.point_confusion(c)[0, 0] == pytest.approx((8 + 1 / 3) / 11)


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
