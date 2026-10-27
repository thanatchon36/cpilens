import numpy as np
import polars as pl
import pytest

from cpilens.evaluate import metrics as m


def test_weighted_equals_unweighted_with_uniform_weights():
    y = ["a", "a", "b", "c", "c", "c"]
    yhat = ["a", "b", "b", "c", None, "a"]
    assert m.accuracy(y, yhat, np.ones(6)) == pytest.approx(m.accuracy(y, yhat))
    assert m.accuracy(y, yhat) == pytest.approx(3 / 6)


def test_item_weights_give_cpi_weighted_recall():
    y = ["a", "a", "b", "OOS"]
    yhat = ["a", "b", "b", "OOS"]
    w = m.item_weights(y, {"a": 3.0, "b": 1.0})
    assert w.tolist() == [1.5, 1.5, 1.0, 0.0]
    # recall_a = 0.5 (weight 3), recall_b = 1 (weight 1)
    assert m.accuracy(y, yhat, w) == pytest.approx((3 * 0.5 + 1 * 1.0) / 4)


def test_inclusion_probabilities_reweight_within_label():
    y = ["a", "a"]
    w = m.item_weights(y, {"a": 2.0}, incl_prob=[0.5, 0.25])
    assert w.tolist() == pytest.approx([2 / 3, 4 / 3])
    assert w.sum() == pytest.approx(2.0)


def test_macro_f1_and_abstention():
    y = ["a", "b"]
    assert m.macro_f1(y, ["a", "b"]) == 1.0
    assert m.macro_f1(y, ["a", None]) == pytest.approx((1.0 + 0.0) / 2)
    assert m.macro_f1(y, ["b", "a"], labels=["a", "b"]) == 0.0


def test_coverage_curve_monotone_in_threshold():
    y = ["a", "a", "b", "b"]
    yhat = ["a", "b", "b", None]
    conf = [0.9, 0.4, 0.8, 0.95]
    cc = m.coverage_curve(y, yhat, conf, m.item_weights(y, {"a": 1.0, "b": 1.0}))
    assert cc["coverage"].to_list() == [0.75, 0.5, 0.25]
    assert cc["acc"].to_list() == pytest.approx([2 / 3, 1.0, 1.0])
    assert cc["coverage_w"][0] == pytest.approx(0.75)


def test_bootstrap_is_seeded_and_stratified():
    strata = ["s"] * 5 + ["t"] * 5
    x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], float)
    d1 = m.bootstrap(lambda i: x[i].mean(), strata, n=200, seed=3)
    d2 = m.bootstrap(lambda i: x[i].mean(), strata, n=200, seed=3)
    assert np.array_equal(d1, d2)
    assert np.allclose(d1, 0.5)  # stratum sizes are fixed, so the mean cannot move
    lo, hi = m.ci(m.bootstrap(lambda i: x[i[:3]].mean(), strata, n=200, seed=1))
    assert lo <= hi


def test_confusion_and_groups():
    y = ["11211", "11212", "11310", "OOS"]
    yhat = ["11212", "11212", None, "11310"]
    c = m.confusion(y, yhat, ["11211", "11212", "11310", "OOS"], level=m.group_of)
    assert c.filter(c["true"] == "112")["112"].item() == 2
    assert "ABSTAIN" in c.columns
    assert m.within_group_error_share(y, yhat) == pytest.approx(0.5)


def test_cpi_label_weights_cover_scope():
    w = m.cpi_label_weights()
    assert len(w) == 28 and w["12300"] == pytest.approx(14.777)
    assert set(m.cpi_label_weights("nnls")) == set(w)  # in-scope totals differ: 40.5 vs 43.5


def test_cascade_routes_low_confidence_to_llm():
    from cpilens.evaluate import frontier as f

    pred, routed = f.cascade(["a", "b", "c"], [0.9, 0.5, 0.2], ["x", "y", "z"], t=0.6)
    assert pred == ["a", "y", "z"] and routed == pytest.approx(2 / 3)
    assert f.cascade(["a"], [0.9], ["x"], t=0.0) == (["a"], 0.0)
    assert f.cascade(["a"], [1.0], ["x"], t=1.01) == (["x"], 1.0)
    assert f.cascade_cost(0.5, 1.0, 10.0) == 6.0
    assert f.local_cost_per_item(3600.0) == pytest.approx(f.GPU_USD_PER_HOUR)


def test_api_cost_per_item():
    import polars as pl

    from cpilens.evaluate import frontier as f

    c = pl.DataFrame(
        {"model": ["m", "m", "n"], "purpose": ["eval:m", "probe", "eval:n"], "usd": [0.2, 9.0, 1.0]}
    )
    assert f.api_cost_per_item(c, "m") == pytest.approx(0.2)


def test_gold_scorer_on_synthetic_labels(tmp_path):
    import polars as pl

    from cpilens.evaluate import gold as g
    from cpilens.label import store

    sample = pl.DataFrame(
        {
            "order": [str(i) for i in range(1, 9)],
            "barcode": [f"b{i}" for i in range(8)],
            "stratum": ["s1"] * 4 + ["s2"] * 4,
            "incl_prob": [0.1] * 4 + [0.2] * 4,
            "double": [False] * 8,
        }
    )
    sample.write_csv(tmp_path / "sample.csv")
    labels = ["11110", "11110", "11600", "OOS", "11600", "12300", "UNK", "11600"]
    for o, lab in zip(sample["order"], labels, strict=True):
        store.append_label(
            1,
            {
                "order": o,
                "barcode": "x",
                "label": lab,
                "unsure": False,
                "seconds": 1,
                "labelled_utc": "2026-10-27T00:00:00+00:00",
            },
            tmp_path,
        )
    sc = g.Scorer(g.gold_table(tmp_path), n_boot=50)
    assert len(sc.y) == 7  # UNK dropped
    ids = sc.g["id"].to_list()
    perfect = pl.DataFrame({"id": ids, "pred": sc.y, "conf": [0.9] * 7})
    r = sc.score(perfect)
    assert r["acc_w"]["est"] == 1.0 and r["acc_w"]["lo"] == 1.0
    worse = perfect.with_columns(pl.Series("pred", ["11600"] * 7))
    d = sc.paired(perfect, worse)
    assert d["est"] > 0 and d["p_le_0"] == 0.0
    f = sc.frontier(worse.with_columns(pl.Series("conf", [0.1 * i for i in range(7)])))
    assert f["coverage_w"][0] == 1.0 and f["coverage_w"][-1] == 0.0


def test_cascade_sweep_endpoints():
    from cpilens.evaluate import frontier as fr

    cheap, conf, llm = ["a", "b", "a"], [0.9, 0.2, 0.6], ["a", "a", "b"]
    y = ["a", "a", "b"]

    def score(p):
        return {"acc": sum(a == b for a, b in zip(y, p, strict=True)) / 3}

    sw = fr.sweep(
        cheap, conf, llm, score, cheap_cost=1e-6, llm_cost=1e-4, thresholds=(0, 0.7, 1.01)
    )
    assert sw["routed"].to_list() == [0, 2 / 3, 1]
    assert sw["acc"].to_list() == [1 / 3, 1, 1]
    assert sw["usd_per_million"][0] == 1.0
    assert abs(sw["usd_per_million"][2] - 101.0) < 1e-9


def test_run_gold_on_synthetic_labels(tmp_path):
    """End to end on the real prediction files with fake labels (Haiku's own answers)."""
    import shutil

    from cpilens.evaluate import run
    from cpilens.label import store
    from cpilens.paths import RESULTS

    if not (RESULTS / "cascade_dev.csv").exists() or not (store.GOLD / "sample.csv").exists():
        pytest.skip("needs cpilens models outputs")
    shutil.copy(store.GOLD / "sample.csv", tmp_path / "sample.csv")
    s = pl.read_csv(tmp_path / "sample.csv", schema_overrides={"barcode": pl.Utf8}).head(120)
    h = pl.read_parquet(RESULTS / "preds/llm/claude-haiku-5-5__low.parquet").filter(
        pl.col("eval_set") == "gold"
    )
    lab = dict(zip(h["id"], h["pred"].fill_null("UNK"), strict=True))
    for r in s.iter_rows(named=True):
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
    run.run_gold(tmp_path, RESULTS, out, n_boot=20)
    sc = pl.read_csv(out / "gold_scores.csv")
    haiku = sc.filter((pl.col("method") == "claude-haiku-5-5") & (pl.col("weights") == "equal"))
    assert haiku["acc"][0] == 1.0  # scored against its own answers
    assert set(sc["family"]) == {"embedding", "finetune", "llm", "cascade"}
    assert pl.read_csv(out / "gold_paired.csv").height > 0


def test_cohen_kappa():
    from cpilens.evaluate.agreement import cohen_kappa

    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # agreement no better than chance given the marginals
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-12


def test_run_agreement(tmp_path):
    import json

    from cpilens.evaluate.agreement import run_agreement
    from cpilens.label import store

    pl.DataFrame(
        {
            "order": ["1", "2", "3", "4", "5"],
            "barcode": ["b1", "b2", "b3", "b4", "b5"],
            "stratum": ["11110", "11110", "11200", "11200", "OOS"],
            "incl_prob": [0.5, 0.5, 0.25, 0.25, 0.1],
            "double": [True] * 5,
        }
    ).write_csv(tmp_path / "sample.csv")
    first = ["11110", "11110", "11211", "11212", "OOS"]
    second = ["11110", "11121", "11211", "11212", "UNK"]
    for p, labels in ((1, first), (2, second)):
        for i, k in enumerate(labels, 1):
            row = {"order": str(i), "barcode": f"b{i}", "label": k, "unsure": str(i == 2)}
            store.append_label(p, row | {"seconds": "1", "labelled_utc": "x"}, tmp_path)
    run_agreement(tmp_path, tmp_path, n_boot=20)
    a = pl.read_csv(tmp_path / "rq4_agreement.csv")
    assert a["n"].to_list() == [4, 4, 4]
    assert a.filter(pl.col("level") == "item")["agree"][0] == 0.75
    assert a.filter(pl.col("level") == "group")["agree"][0] == 1.0  # 11110 and 11121 share 111
    meta = json.loads((tmp_path / "rq4_meta.json").read_text())
    assert (meta["n_pairs"], meta["n_unk_either"], meta["n_unsure_either"]) == (5, 1, 1)
    assert meta["agree_if_unsure"] == 0.0
    by = pl.read_csv(tmp_path / "rq4_by_class.csv", schema_overrides={"class": pl.Utf8})
    g = by.filter(pl.col("level") == "group").sort("class")
    assert g["class"].to_list() == ["111", "112"] and g["disagree"].to_list() == [0.5, 0.0]
