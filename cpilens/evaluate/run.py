"""Evaluation of every method on the reference set (held-out official specifications):
scores with stratified-bootstrap intervals, coverage frontiers, paired comparisons, and the
dev-chosen cascades.

Methods are read from the prediction files written by ``cpilens models``. A cascade's
threshold is chosen on the development set (``results/cascade_dev.csv``) without
reference-set scores: the threshold with the highest Thai weak-validation weighted
accuracy, ties broken by lower cost. The reference set then scores that one threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.evaluate import frontier as fr
from cpilens.evaluate.gold import N_BOOT, Scorer, reference_table, sub_item
from cpilens.evaluate.metrics import group_of
from cpilens.paths import RESULTS

PREDS = RESULTS / "preds"
LLMS = ("claude-haiku-5-5", "claude-opus-5-5")
CASCADE_CHEAP = (
    "e5-large__mlp__nonthai+thaiweak+mocall",
    "bge-m3__mlp__nonthai+thaiweak+mocall",
    "xlmr__finetune__nonthai+thaiweak+mocall",
)
STATS = ("acc", "acc_w", "macro_f1", "within_group_errors")


def is_headline(name: str, meta: dict[str, object]) -> bool:
    """Methods that get frontiers, monthly series and full decompositions."""
    return meta["family"] in ("llm", "cascade") or name in CASCADE_CHEAP


def choose_threshold(dev: pl.DataFrame, cheap: str, llm: str) -> float:
    d = dev.filter(
        (pl.col("cheap") == cheap)
        & (pl.col("llm") == llm)
        & (pl.col("eval_set") == "thai_val_weak")
    )
    best = d.sort(["acc_w", "usd_per_million"], descending=[True, False]).row(0, named=True)
    return float(best["threshold"])


def methods(
    results: Path = RESULTS, eval_set: str = "moc_test", ref: pl.DataFrame | None = None
) -> dict[str, tuple[pl.DataFrame, dict[str, object]]]:
    """Method name -> (reference-set predictions: id, pred, conf; metadata incl. USD per
    million). A cascade's routed share, and so its cost, is design-weighted (1/pi over the
    reference table ``ref``): a share of the product universe, not of the sample."""
    inv = (reference_table() if ref is None else ref).select(
        "id", (1 / pl.col("incl_prob")).alias("inv"), (pl.col("label") == "OOS").alias("oos")
    )
    grid = pl.read_csv(results / "rq1_dev_grid.csv")
    costs = pl.read_csv(results / "llm_costs.csv")
    embed_s = json.loads((results / "embed_timing.json").read_text())
    dev = pl.read_csv(results / "cascade_dev.csv")
    preds = results / "preds"

    def gold(p: Path) -> pl.DataFrame:
        return (
            pl.read_parquet(p).filter(pl.col("eval_set") == eval_set).select("id", "pred", "conf")
        )

    out: dict[str, tuple[pl.DataFrame, dict[str, object]]] = {}
    cheap_cost: dict[str, float] = {}
    for row in grid.iter_rows(named=True):
        cheap_cost[row["run"]] = fr.cheap_cost_per_item(row, embed_s)
        meta = {k: row[k] for k in ("emb", "clf", "scenario", "text_variant")}
        family = "finetune" if row["clf"] == "finetune" else "embedding"
        out[row["run"]] = (
            gold(preds / "embed" / f"{row['run']}.parquet"),
            {**meta, "family": family, "usd_per_million": 1e6 * cheap_cost[row["run"]]},
        )
    llm_cost = {m: fr.api_cost_per_item(costs, m) for m in LLMS}
    for m in LLMS:
        out[m] = (
            gold(preds / "llm" / f"{m}__low.parquet"),
            {"family": "llm", "usd_per_million": 1e6 * llm_cost[m]},
        )
    for m in LLMS:
        lp = out[m][0].rename({"pred": "llm_pred", "conf": "llm_conf"})
        for c in CASCADE_CHEAP:
            t = choose_threshold(dev, c, m)
            d = out[c][0].join(lp, on="id", how="inner").join(inv, on="id", how="inner")
            pred, routed = fr.cascade(
                d["pred"].to_list(),
                d["conf"].to_list(),
                d["llm_pred"].to_list(),
                t,
                d["inv"].to_numpy(),
            )
            # a cascade's confidence is that of whichever model answered
            route = d["conf"] < t
            # the routed share, and so the cost, depends on the product mix
            by_scope = {
                f"routed_{k}": float(np.average(route.filter(sel), weights=d["inv"].filter(sel)))
                for k, sel in (("in_scope", ~d["oos"]), ("oos", d["oos"]))
            }
            conf = pl.select(pl.when(route).then(d["llm_conf"]).otherwise(d["conf"])).to_series()
            out[f"cascade__{c}__{m}"] = (
                pl.DataFrame({"id": d["id"], "pred": pred, "conf": conf}),
                {
                    "family": "cascade",
                    "cheap": c,
                    "llm": m,
                    "threshold": t,
                    "routed": routed,
                    **by_scope,
                    "usd_per_million": 1e6 * fr.cascade_cost(routed, cheap_cost[c], llm_cost[m]),
                },
            )
    return out


def paired_list() -> list[tuple[str, str]]:
    e5 = "e5-large__mlp__nonthai+thaiweak+mocall"
    pairs = [("claude-opus-5-5", "claude-haiku-5-5")]
    pairs += [("claude-haiku-5-5", c) for c in CASCADE_CHEAP]
    pairs += [(f"cascade__{c}__{m}", m) for m in LLMS for c in CASCADE_CHEAP]
    pairs += [(e5, f"{e5}-{v}") for v in ("nobrand", "noqty", "nobrand-noqty")]
    pairs += [(e5, "e5-large__mlp__nonthai"), (e5, "e5-large__mlp__nonthai+thaiweak")]
    return pairs


def label_recall(sc: Scorer, name: str, p: pl.DataFrame) -> pl.DataFrame:
    """Per true label: Horvitz-Thompson recall, universe share and CPI weight."""
    pred, _ = sc.align(p)
    inv = 1 / sc.pi
    d = pl.DataFrame(
        {"label": sc.y, "inv": inv, "hit": [a == b for a, b in zip(sc.y, pred, strict=True)]}
    )
    return (
        d.group_by("label")
        .agg(
            pl.len().alias("n"),
            ((pl.col("inv") * pl.col("hit")).sum() / pl.col("inv").sum()).alias("recall"),
            pl.col("inv").sum().alias("_inv"),
        )
        .with_columns(
            (pl.col("_inv") / inv.sum()).alias("universe_share"),
            pl.col("label")
            .replace_strict(sc.lw, default=0.0, return_dtype=pl.Float64)
            .alias("weight"),
            pl.lit(name).alias("method"),
        )
        .drop("_inv")
        .sort("label")
    )


def group_confusion(sc: Scorer, name: str, p: pl.DataFrame) -> pl.DataFrame:
    """Row-normalised (Horvitz-Thompson) confusion between CPI groups; ABSTAIN is a column."""
    pred, _ = sc.align(p)
    d = pl.DataFrame(
        {
            "true": [group_of(k) for k in sc.y],
            "pred": ["ABSTAIN" if b is None else group_of(b) for b in pred],
            "inv": 1 / sc.pi,
        }
    )
    return (
        d.group_by("true", "pred")
        .agg(pl.col("inv").sum(), pl.len().alias("n"))
        .with_columns(
            (pl.col("inv") / pl.col("inv").sum().over("true")).alias("share"),
            pl.lit(name).alias("method"),
        )
        .drop("inv")
        .sort("true", "pred")
    )


def cascade_sweep(
    sc: Scorer, ms: dict[str, tuple[pl.DataFrame, dict[str, object]]]
) -> pl.DataFrame:
    """Every threshold of every cascade on gold (descriptive; the reported cascade uses the
    dev-chosen threshold)."""
    out = []
    for m in LLMS:
        for c in CASCADE_CHEAP:
            cp, cmeta = ms[c]
            lp, lmeta = ms[m]
            d = (
                sc.g.select("id")
                .join(cp, on="id", how="left")
                .join(lp.rename({"pred": "llm_pred", "conf": "llm_conf"}), on="id", how="left")
            )
            for t in fr.THRESHOLDS:
                pred, routed = fr.cascade(
                    d["pred"].to_list(),
                    d["conf"].fill_null(0.0).to_list(),
                    d["llm_pred"].to_list(),
                    t,
                    sc.inv,
                )
                s = sc.score(pl.DataFrame({"id": d["id"], "pred": pred, "conf": [1.0] * d.height}))
                acc = s["acc_w"]
                assert isinstance(acc, dict)
                cost = float(cmeta["usd_per_million"]) + routed * float(lmeta["usd_per_million"])  # type: ignore[arg-type]
                out.append(
                    {
                        "cheap": c,
                        "llm": m,
                        "threshold": t,
                        "routed": routed,
                        "usd_per_million": cost,
                        "acc_w": acc["est"],
                        "acc_w_lo": acc["lo"],
                        "acc_w_hi": acc["hi"],
                        "chosen": t == ms[f"cascade__{c}__{m}"][1]["threshold"],
                    }
                )
    return pl.DataFrame(out)


def subitems(g: pl.DataFrame) -> pl.DataFrame:
    """Specification id -> its official sub-item (7-digit code, read off the id)."""
    return g.select(
        "id", pl.col("id").map_elements(sub_item, return_dtype=pl.Utf8).alias("item_code")
    )


def error_subitems(g: pl.DataFrame, name: str, p: pl.DataFrame, sub: pl.DataFrame) -> pl.DataFrame:
    """In-scope reference errors of one method, counted by official sub-item and prediction.
    In-scope rows all have inclusion probability 1, so counts need no design weight."""
    return (
        g.filter(pl.col("label") != "OOS")
        .join(p.select("id", "pred"), on="id", how="left")
        .filter(pl.col("pred").is_null() | (pl.col("pred") != pl.col("label")))
        .join(sub, on="id", how="left")
        .group_by("item_code", "label", "pred")
        .agg(pl.len().alias("n"))
        .with_columns(pl.lit(name).alias("method"))
        .sort("n", descending=True)
    )


def run_eval(
    ref: pl.DataFrame | None = None,
    results: Path = RESULTS,
    out: Path = RESULTS,
    n_boot: int = N_BOOT,
    eval_set: str = "moc_test",
) -> None:
    """Score every method on the reference set (``ref``: a table like ``reference_table``)."""
    g = reference_table() if ref is None else ref
    ms = methods(results, eval_set, g)
    summary = {
        "n_ref_specs": g.height,
        "n_ref_items": g.filter(pl.col("label") != "OOS")["label"].n_unique(),
        "n_ref_oos": int((g["label"] == "OOS").sum()),
        "n_boot": n_boot,
    }
    (out / "gold_summary.json").write_text(json.dumps(summary, indent=2))
    rows, fronts, pairs, recalls, confs, errs = [], [], [], [], [], []
    sub = subitems(g)
    for weights in ("equal", "nnls"):
        sc = Scorer(g, weights, n_boot)
        for name, (p, meta) in ms.items():
            s = sc.score(p)
            row: dict[str, object] = {"method": name, "weights": weights, "n": s["n"], **meta}
            for k in STATS:
                v = s[k]
                assert isinstance(v, dict)
                row |= {k: v["est"], f"{k}_lo": v["lo"], f"{k}_hi": v["hi"]}
            rows.append(row)
            if weights == "equal" and is_headline(name, meta):
                fronts.append(sc.frontier(p).with_columns(pl.lit(name).alias("method")))
                recalls.append(label_recall(sc, name, p))
                confs.append(group_confusion(sc, name, p))
                errs.append(error_subitems(g, name, p, sub))
        if weights == "equal":
            cascade_sweep(sc, ms).write_csv(out / "gold_cascade_sweep.csv")
        for a, b in paired_list():
            if a in ms and b in ms:
                d = sc.paired(ms[a][0], ms[b][0])
                pairs.append({"a": a, "b": b, "weights": weights, "stat": "acc_w", **d})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out / "gold_scores.csv")
    pl.concat(fronts).write_csv(out / "gold_frontier.csv")
    pl.DataFrame(pairs).write_csv(out / "gold_paired.csv")
    pl.concat(recalls).write_csv(out / "gold_label_recall.csv")
    pl.concat(confs).write_csv(out / "gold_confusion_group.csv")
    pl.concat(errs).write_csv(out / "gold_error_subitems.csv")
