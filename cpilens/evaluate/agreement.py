"""RQ4, the label ceiling: how often the gold labeller disagrees with themself.

The ``double`` subset is labelled twice, blind, at least 48 hours apart (``label.store``).
Pass 1 plays the role of the truth and pass 2 the role of a classifier, so ``acc_w`` here is
on the same scale as every method's ``acc_w`` on gold. Agreement is reported at the item,
group and division level, with Cohen's kappa and stratified bootstrap intervals. Pairs where
either pass is ``UNK`` are excluded and counted.

Outputs: ``rq4_agreement.csv`` (by hierarchy level), ``rq4_by_class.csv`` (item-level
disagreement by division and by group of the pass-1 label) and ``rq4_meta.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.evaluate import metrics as m
from cpilens.evaluate.gold import N_BOOT, SEED
from cpilens.label import store
from cpilens.paths import RESULTS


def division_of(label: str) -> str:
    return label[0] if label[:1].isdigit() else label


LEVELS: dict[str, Callable[[str], str]] = {
    "item": lambda k: k,
    "group": m.group_of,
    "division": division_of,
}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n**2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pairs(gold: Path = store.GOLD) -> pl.DataFrame:
    """Products labelled in both passes: pass-1 and pass-2 label, unsure flags, design."""
    s = pl.read_csv(
        gold / store.SAMPLE.name, schema_overrides={"barcode": pl.Utf8, "order": pl.Utf8}
    )
    p1 = store.read_labels(1, gold).select(
        "order", pl.col("label").alias("label1"), (pl.col("unsure") == "True").alias("unsure1")
    )
    p2 = store.read_labels(2, gold).select(
        "order", pl.col("label").alias("label2"), (pl.col("unsure") == "True").alias("unsure2")
    )
    return s.join(p1, on="order").join(p2, on="order")


def _stats(a: list[str], b: list[str], w: np.ndarray) -> dict[str, float]:
    return {
        "agree": m.accuracy(a, b),
        "acc_w": m.accuracy(a, b, w),
        "kappa": cohen_kappa(a, b),
    }


def run_agreement(
    gold: Path = store.GOLD, out: Path = RESULTS, n_boot: int = N_BOOT, weights: str = "equal"
) -> None:
    d = pairs(gold)
    if d.height == 0:
        raise SystemExit("no double labels yet: label pass 2 first (cpilens label --pass 2)")
    unk = d.filter((pl.col("label1") == "UNK") | (pl.col("label2") == "UNK"))
    d = d.filter((pl.col("label1") != "UNK") & (pl.col("label2") != "UNK"))
    lw = m.cpi_label_weights(weights)
    pi = d["incl_prob"].to_numpy()
    y1, y2 = d["label1"].to_list(), d["label2"].to_list()
    boot = m.bootstrap_indices(d["stratum"].to_list(), n_boot, SEED)

    def weights_of(idx: np.ndarray) -> np.ndarray:
        return m.item_weights([y1[i] for i in idx], lw, pi[idx])

    rows = []
    for level, f in LEVELS.items():
        a, b = [f(k) for k in y1], [f(k) for k in y2]
        point = _stats(a, b, weights_of(np.arange(len(a))))
        draws = [_stats([a[i] for i in ix], [b[i] for i in ix], weights_of(ix)) for ix in boot]
        row: dict[str, object] = {"level": level, "n": len(a)}
        for k, v in point.items():
            lo, hi = m.ci(np.array([x[k] for x in draws]))
            row |= {k: v, f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
    pl.DataFrame(rows).write_csv(out / "rq4_agreement.csv")

    # item-level disagreement by the pass-1 label's division and group (unweighted share)
    rng = np.random.default_rng(SEED)
    by = []
    for level in ("division", "group"):
        cls = np.array([LEVELS[level](k) for k in y1])
        for c in sorted(set(cls)):
            ix = np.flatnonzero(cls == c)
            miss = np.array([y1[i] != y2[i] for i in ix], float)
            bd = [miss[rng.integers(0, len(ix), len(ix))].mean() for _ in range(n_boot)]
            lo, hi = m.ci(np.array(bd))
            by.append(
                {
                    "level": level,
                    "class": c,
                    "n": len(ix),
                    "disagree": float(miss.mean()),
                    "disagree_lo": lo,
                    "disagree_hi": hi,
                }
            )
    pl.DataFrame(by).write_csv(out / "rq4_by_class.csv")

    unsure = (d["unsure1"] | d["unsure2"]).to_numpy()
    agree = np.array([x == y for x, y in zip(y1, y2, strict=True)], float)
    meta = {
        "n_pairs": d.height + unk.height,
        "n_unk_either": unk.height,
        "n_unsure_either": int(unsure.sum()),
        "agree_if_unsure": float(agree[unsure].mean()) if unsure.any() else None,
        "agree_if_sure": float(agree[~unsure].mean()) if (~unsure).any() else None,
        "n_boot": n_boot,
        "weights": weights,
    }
    (out / "rq4_meta.json").write_text(json.dumps(meta, indent=2))
