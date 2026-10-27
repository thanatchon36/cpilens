"""Gold evaluation of every method: scores with stratified-bootstrap intervals, coverage
frontiers, paired comparisons, and the dev-chosen cascades.

Methods are read from the prediction files written by ``cpilens models``. A cascade's
threshold is chosen on the dev sets (``results/cascade_dev.csv``) before gold is touched:
the threshold with the highest Thai weak-validation weighted accuracy (the dev set closest
to the gold population), ties broken by lower cost. Gold then scores that one threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cpilens.evaluate import frontier as fr
from cpilens.evaluate.gold import N_BOOT, Scorer, gold_table
from cpilens.label import store
from cpilens.paths import RESULTS

PREDS = RESULTS / "preds"
LLMS = ("claude-haiku-5-5", "claude-opus-5-5")
CASCADE_CHEAP = (
    "e5-large__mlp__nonthai+thaiweak+mocall",
    "bge-m3__mlp__nonthai+thaiweak+mocall",
    "xlmr__finetune__nonthai+thaiweak+mocall",
)
STATS = ("acc", "acc_w", "macro_f1", "within_group_errors")


def choose_threshold(dev: pl.DataFrame, cheap: str, llm: str) -> float:
    d = dev.filter(
        (pl.col("cheap") == cheap)
        & (pl.col("llm") == llm)
        & (pl.col("eval_set") == "thai_val_weak")
    )
    best = d.sort(["acc_w", "usd_per_million"], descending=[True, False]).row(0, named=True)
    return float(best["threshold"])


def methods(results: Path = RESULTS) -> dict[str, tuple[pl.DataFrame, dict[str, object]]]:
    """Method name -> (gold predictions: id, pred, conf; metadata incl. USD per million)."""
    grid = pl.read_csv(results / "rq1_dev_grid.csv")
    costs = pl.read_csv(results / "llm_costs.csv")
    embed_s = json.loads((results / "embed_timing.json").read_text())
    dev = pl.read_csv(results / "cascade_dev.csv")
    preds = results / "preds"

    def gold(p: Path) -> pl.DataFrame:
        return pl.read_parquet(p).filter(pl.col("eval_set") == "gold").select("id", "pred", "conf")

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
            d = out[c][0].join(lp, on="id", how="inner")
            pred, routed = fr.cascade(
                d["pred"].to_list(), d["conf"].to_list(), d["llm_pred"].to_list(), t
            )
            # a cascade's confidence is that of whichever model answered
            route = d["conf"] < t
            conf = pl.select(pl.when(route).then(d["llm_conf"]).otherwise(d["conf"])).to_series()
            out[f"cascade__{c}__{m}"] = (
                pl.DataFrame({"id": d["id"], "pred": pred, "conf": conf}),
                {
                    "family": "cascade",
                    "cheap": c,
                    "llm": m,
                    "threshold": t,
                    "routed": routed,
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


def run_gold(
    gold: Path = store.GOLD, results: Path = RESULTS, out: Path = RESULTS, n_boot: int = N_BOOT
) -> None:
    g = gold_table(gold)
    if g.height == 0:
        raise SystemExit("no gold labels yet: label pass 1 first (cpilens label --pass 1)")
    ms = methods(results)
    summary = {
        "n_labelled": g.height,
        "n_unk": int((g["label"] == "UNK").sum()),
        "n_unsure": int(g["unsure"].sum()),
        "n_scored": int((g["label"] != "UNK").sum()),
    }
    (out / "gold_summary.json").write_text(json.dumps(summary, indent=2))
    rows, fronts, pairs = [], [], []
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
            if weights == "equal" and (
                meta["family"] in ("llm", "cascade") or name in CASCADE_CHEAP
            ):
                fronts.append(sc.frontier(p).with_columns(pl.lit(name).alias("method")))
        for a, b in paired_list():
            if a in ms and b in ms:
                d = sc.paired(ms[a][0], ms[b][0])
                pairs.append({"a": a, "b": b, "weights": weights, "stat": "acc_w", **d})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out / "gold_scores.csv")
    pl.concat(fronts).write_csv(out / "gold_frontier.csv")
    pl.DataFrame(pairs).write_csv(out / "gold_paired.csv")
