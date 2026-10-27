"""Economist-first classification metrics, a stratified bootstrap, and confusion tables.

Item weights. A gold product *i* with true CPI label *k* gets

    omega_i = w_k * (1/pi_i) / sum_{j: y_j = k} (1/pi_j),

where ``w_k`` is the label's CPI weight and ``pi_i`` its inclusion probability in the gold
sample. Then sum_i omega_i 1[correct_i] / sum_i omega_i = sum_k w_k r_k / sum_k w_k, with
``r_k`` the Horvitz-Thompson recall of label *k* over the Thai product universe: per-item
recall weighted by CPI weight, the expenditure-weighted accuracy of the mission. Labels
with no CPI weight (``OOS``) get omega = 0; they enter unweighted accuracy and macro-F1.

Abstention is ``None`` in the predictions; an abstained product counts as wrong in
full-coverage accuracy and as uncovered in the coverage frontier. Weighted coverage is the
share of the (gold-present) CPI weight whose products are classified, not abstained.

Every statistic is a function of row indices, so the bootstrap recomputes the weights in
each resample (the label normalisation depends on the resample).
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from cpilens.paths import TAXONOMY

Labels = Sequence[str | None]


def cpi_label_weights(method: str = "equal") -> dict[str, float]:
    """CPI weight (price-updated, % of headline) of each in-scope label. Leaves inside a
    residual weight cell share it ``equal``-ly (primary) or by ``nnls`` (sensitivity)."""
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    lw = pl.read_csv(TAXONOMY / "thcpi_leaf_weights_2566.csv", schema_overrides={"leaf": pl.Utf8})
    w = dict(zip(lw["leaf"], lw[f"w_{method}"], strict=True))
    return {k: float(w[k]) for k in scope.filter(pl.col("in_scope"))["label"]}


def item_weights(
    y: Labels, label_w: dict[str, float], incl_prob: Sequence[float] | None = None
) -> np.ndarray:
    y = list(y)
    inv = np.ones(len(y)) if incl_prob is None else 1.0 / np.asarray(incl_prob, dtype=float)
    tot: dict[str, float] = {}
    for k, v in zip(y, inv, strict=True):
        if k in label_w:
            tot[k] = tot.get(k, 0.0) + v
    return np.array(
        [label_w[k] * v / tot[k] if k in label_w else 0.0 for k, v in zip(y, inv, strict=True)]
    )


def accuracy(y: Labels, yhat: Labels, w: np.ndarray | None = None) -> float:
    hit = np.array([a == b and b is not None for a, b in zip(y, yhat, strict=True)], float)
    w = np.ones(len(hit)) if w is None else w
    return float((w * hit).sum() / w.sum()) if w.sum() > 0 else float("nan")


def macro_f1(y: Labels, yhat: Labels, labels: Sequence[str] | None = None) -> float:
    """Mean F1 over ``labels`` (default: labels present in ``y``); abstentions are misses."""
    ks = sorted({k for k in y if k is not None}) if labels is None else list(labels)
    tp: Counter[str | None] = Counter()
    n_true: Counter[str | None] = Counter()
    n_pred: Counter[str | None] = Counter()
    for a, b in zip(y, yhat, strict=True):
        n_true[a] += 1
        n_pred[b] += 1
        tp[a] += a == b
    # 2tp + fp + fn = (tp + fn) + (tp + fp) = n_true + n_pred
    f1 = [2 * tp[k] / (n_true[k] + n_pred[k]) if n_true[k] + n_pred[k] else 0.0 for k in ks]
    return float(np.mean(f1)) if f1 else float("nan")


def coverage_curve(
    y: Labels,
    yhat: Labels,
    conf: Sequence[float] | np.ndarray,
    w: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Selective accuracy against weighted coverage when abstaining below each threshold."""
    conf_a = np.asarray(conf, dtype=float)
    hit = np.array([a == b and b is not None for a, b in zip(y, yhat, strict=True)], float)
    answered = np.array([b is not None for b in yhat])
    ts = np.unique(conf_a[answered]) if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for t in ts:
        keep = answered & (conf_a >= t)
        cw = (w * keep).sum()
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(keep.mean()),
                "coverage_w": float(cw / w.sum()),
                "acc": float(hit[keep].mean()) if keep.any() else float("nan"),
                "acc_w": float((w * hit * keep).sum() / cw) if cw > 0 else float("nan"),
            }
        )
    return pl.DataFrame(rows)


def bootstrap_indices(
    strata: Sequence[object],
    n: int = 1000,
    seed: int = 0,
    clusters: Sequence[object] | None = None,
) -> list[np.ndarray]:
    """``n`` resamples of row indices (seeded). Within each stratum, rows are drawn with
    replacement; with ``clusters``, whole clusters are drawn instead (as many as the stratum
    has) and all their rows kept. A stratum with a single cluster falls back to its rows,
    which understates its variance but does not set it to zero."""
    rng = np.random.default_rng(seed)
    s = np.asarray([str(x) for x in strata])
    c = (
        np.arange(len(s)).astype(str)
        if clusters is None
        else np.asarray([str(x) for x in clusters])
    )
    groups = []
    for g in np.unique(s):
        rows = np.flatnonzero(s == g)
        ks = np.unique(c[rows])
        groups.append([rows[c[rows] == k] for k in ks] if len(ks) > 1 else [[r] for r in rows])
    out = []
    for _ in range(n):
        parts: list[np.ndarray] = []
        for cl in groups:
            parts.extend(cl[i] for i in rng.integers(0, len(cl), len(cl)))
        out.append(np.concatenate(parts))
    return out


def bootstrap(
    stat: Callable[[np.ndarray], float | np.ndarray],
    strata: Sequence[object],
    n: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Draws of ``stat(idx)`` under the stratified resampling of :func:`bootstrap_indices`."""
    return np.asarray([stat(idx) for idx in bootstrap_indices(strata, n, seed)])


def ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    with warnings.catch_warnings():  # e.g. no errors in a resample -> NaN error share
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(draws, [a, 1 - a], axis=0)
    return float(lo), float(hi)


def confusion(
    y: Labels, yhat: Labels, labels: Sequence[str], level: Callable[[str], str] | None = None
) -> pl.DataFrame:
    """Counts true x predicted (abstentions as ``ABSTAIN``), optionally at a coarser level."""
    f = level or (lambda k: k)
    t = [f(str(a)) for a in y]
    p = ["ABSTAIN" if b is None else f(b) for b in yhat]
    order = list(dict.fromkeys(f(k) for k in labels))
    df = pl.DataFrame({"true": t, "pred": p}).group_by("true", "pred").len()
    return (
        df.pivot(on="pred", index="true", values="len")
        .fill_null(0)
        .select("true", *[c for c in [*order, "ABSTAIN"] if c in df["pred"].unique().to_list()])
    )


def group_of(label: str) -> str:
    """CPI group: the first three digits of the item code (``OOS`` stays ``OOS``)."""
    return label if not label[:1].isdigit() else label[:3]


def within_group_error_share(y: Labels, yhat: Labels) -> float:
    errs = [(a, b) for a, b in zip(y, yhat, strict=True) if b is not None and a != b]
    if not errs:
        return float("nan")
    return float(np.mean([group_of(str(a)) == group_of(str(b)) for a, b in errs]))
