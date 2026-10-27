"""RQ1 grid: embedding model x classifier x training scenario, predictions for every eval set.

Scenarios (training pools; every pool uses only ``train``/``val`` rows):
- ``nonthai``: weak labels from non-Thai OFF/OBF products (Asia and Europe): zero-shot
  cross-lingual transfer to Thai;
- ``nonthai+thaiweak``: plus weak labels of Thai OFF/OBF products;
- ``nonthai+thaiweak+moc{n}`` and ``moc{n}``: plus / only ``n`` Thai official labels (MOC
  specifications). The budget ``n`` counts every official label used: they are taken in a
  fixed hash order from MOC train+val and split 80/20 into training and validation, so a
  run with ``n`` labels never looks at more than ``n`` official labels. ``all`` uses the
  full MOC train and val splits.

Hyper-parameters (k for k-NN, early stopping) and the temperature are fitted on the
scenario's own validation rows. Predictions are written for the MOC test split (the
reference set) and the Thai OFF validation split (weak labels; development only).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from functools import partial

import numpy as np
import polars as pl

from cpilens.classify import embedclf as ec
from cpilens.classify.config import BUDGETS, MLP_HIDDEN
from cpilens.classify.data import bucket, corpus
from cpilens.embed.encode import MODELS, embed
from cpilens.evaluate import metrics as m
from cpilens.paths import RESULTS, TAXONOMY

PREDS = RESULTS / "preds" / "embed"
CLASSIFIERS = ("knn", "linear", "mlp")


def classes() -> list[str]:
    return pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})[
        "label"
    ].to_list()


def moc_budget(c: pl.DataFrame, n: int | str) -> pl.DataFrame:
    moc = c.filter((pl.col("source") == "moc") & pl.col("split").is_in(["train", "val"]))
    if n == "all":
        return moc
    moc = moc.with_columns(
        pl.col("id")
        .map_elements(lambda k: bucket("budget:" + k, 10**9), return_dtype=pl.Int64)
        .alias("_h")
    )
    take = moc.sort("_h").head(int(n))
    n_val = max(1, round(0.2 * int(n)))
    return take.with_columns(
        pl.Series("split", ["val"] * n_val + ["train"] * (take.height - n_val))
    ).drop("_h")


def scenarios(c: pl.DataFrame) -> Iterator[tuple[str, pl.DataFrame]]:
    off = c.filter(pl.col("label_kind") == "weak")
    nonthai = off.filter(pl.col("region") != "thai")
    weak = off
    yield "nonthai", nonthai
    yield "nonthai+thaiweak", weak
    for n in BUDGETS:
        yield f"nonthai+thaiweak+moc{n}", pl.concat([weak, moc_budget(c, n)])
    for n in BUDGETS:
        yield f"moc{n}", moc_budget(c, n)


def eval_sets(c: pl.DataFrame) -> pl.DataFrame:
    return c.filter(
        ((pl.col("source") == "moc") & (pl.col("split") == "test"))
        | (
            (pl.col("region") == "thai")
            & (pl.col("label_kind") == "weak")
            & (pl.col("split") == "val")
        )
    ).with_columns(
        pl.when(pl.col("source") == "moc")
        .then(pl.lit("moc_test"))
        .otherwise(pl.lit("thai_val_weak"))
        .alias("eval_set")
    )


def run(
    emb_models: tuple[str, ...] = tuple(MODELS),
    clfs: tuple[str, ...] = CLASSIFIERS,
    only: set[str] | None = None,
    with_brand: bool = True,
    with_quantity: bool = True,
) -> pl.DataFrame:
    """The grid. ``with_brand`` / ``with_quantity`` = False is the text ablation; it changes
    OFF texts only (MOC specifications are single free-text strings)."""
    c = corpus(with_brand, with_quantity)
    variant = ("" if with_brand else "-nobrand") + ("" if with_quantity else "-noqty")
    cls = classes()
    ev = eval_sets(c)
    PREDS.mkdir(parents=True, exist_ok=True)
    rows = []
    for emb in emb_models:
        x_all = embed(c["text"].to_list(), emb)
        row_of = {k: i for i, k in enumerate(c["id"])}
        xev = x_all[[row_of[k] for k in ev["id"]]]
        for name, pool in scenarios(c):
            if only and name not in only:
                continue
            tr, va = pool.filter(pl.col("split") == "train"), pool.filter(pl.col("split") == "val")
            xtr, xva = (x_all[[row_of[k] for k in d["id"]]] for d in (tr, va))
            ytr, yva = (ec.encode_labels(d["label"].to_list(), cls) for d in (tr, va))
            for clf in clfs:
                t0 = time.time()
                info: dict[str, object] = {}
                logits: Callable[[np.ndarray], np.ndarray]
                if clf == "knn":
                    k = ec.knn_tune(xtr, ytr, xva, yva, len(cls))
                    info["k"] = k
                    logits = partial(ec.knn_logits, xtr, ytr, k=k, n_classes=len(cls))
                else:
                    cfg = ec.TrainCfg(hidden=0 if clf == "linear" else MLP_HIDDEN)
                    logits = partial(ec.net_logits, ec.train_net(xtr, ytr, xva, yva, len(cls), cfg))
                lva = logits(xva)
                temp = ec.fit_temperature(lva, yva)
                fit_s = time.time() - t0
                t1 = time.time()
                prob = ec.softmax(logits(xev), temp)
                pred_s = (time.time() - t1) / len(xev)
                run_id = f"{emb}__{clf}__{name}{variant}"
                out = ev.select("id", "eval_set", "label").with_columns(
                    pl.Series("pred", [cls[i] for i in prob.argmax(1)]),
                    pl.Series("conf", prob.max(1)),
                )
                out.write_parquet(PREDS / f"{run_id}.parquet")
                pva = ec.softmax(lva, temp)
                rows.append(
                    {
                        "run": run_id,
                        "emb": emb,
                        "clf": clf,
                        "scenario": name,
                        "text_variant": variant.lstrip("-") or "full",
                        "n_train": tr.height,
                        "n_val": va.height,
                        "n_official": int((pool["label_kind"] == "official").sum()),
                        "temperature": temp,
                        "ece_val_raw": ec.ece(ec.softmax(lva), yva),
                        "ece_val": ec.ece(pva, yva),
                        "fit_s": fit_s,
                        "pred_s_per_item": pred_s,
                        **info,
                        **dev_metrics(out),
                    }
                )
                print(
                    json.dumps(
                        {k: v for k, v in rows[-1].items() if k not in ("emb", "clf", "scenario")},
                        default=str,
                    ),
                    flush=True,
                )
    return pl.DataFrame(rows)


def dev_metrics(out: pl.DataFrame) -> dict[str, float]:
    """Development metrics on labelled non-gold eval sets (gold is scored separately)."""
    lw = m.cpi_label_weights()
    res = {}
    for s in ("moc_test", "thai_val_weak"):
        d = out.filter(pl.col("eval_set") == s)
        y, p = d["label"].to_list(), d["pred"].to_list()
        res[f"{s}_acc"] = m.accuracy(y, p)
        res[f"{s}_acc_w"] = m.accuracy(y, p, m.item_weights(y, lw))
        res[f"{s}_macro_f1"] = m.macro_f1(y, p)
    return res
