"""Scoring on the gold sample: headline metrics with stratified-bootstrap intervals, the
expenditure-weighted accuracy-coverage frontier, and paired method comparisons.

Gold = pass-1 labels (latest per product). ``UNK`` products are excluded from accuracy and
counted separately. The bootstrap resamples gold products within the sampling strata, and a
paired comparison uses the same resampled indices for both methods.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.evaluate import metrics as m
from cpilens.label import store

N_BOOT = 1000
SEED = 20261027
THRESHOLDS = tuple(np.round(np.arange(0, 1.0001, 0.05), 2))


def gold_table(gold: Path = store.GOLD) -> pl.DataFrame:
    """One row per labelled gold product: ``id`` (corpus id), label, sampling design."""
    s = pl.read_csv(
        gold / store.SAMPLE.name, schema_overrides={"barcode": pl.Utf8, "order": pl.Utf8}
    )
    lab = store.read_labels(1, gold).select("order", "label", "unsure")
    return s.join(lab, on="order", how="inner").with_columns(
        ("off:" + pl.col("barcode")).alias("id"), (pl.col("unsure") == "True").alias("unsure")
    )


def _stats(y: list[str], p: list[str | None], w: np.ndarray) -> dict[str, float]:
    return {
        "acc": m.accuracy(y, p),
        "acc_w": m.accuracy(y, p, w),
        "macro_f1": m.macro_f1(y, p),
        "within_group_errors": m.within_group_error_share(y, p),
    }


class Scorer:
    def __init__(self, g: pl.DataFrame, method: str = "equal") -> None:
        g = g.filter(pl.col("label") != "UNK")
        self.g = g
        self.y = g["label"].to_list()
        self.pi = g["incl_prob"].to_numpy()
        self.strata = g["stratum"].to_list()
        self.lw = m.cpi_label_weights(method)
        # the same resamples for every method make comparisons paired
        self.boot_idx = m.bootstrap_indices(self.strata, N_BOOT, SEED)

    def align(self, preds: pl.DataFrame) -> tuple[list[str | None], np.ndarray]:
        d = self.g.select("id").join(preds.select("id", "pred", "conf"), on="id", how="left")
        return d["pred"].to_list(), d["conf"].fill_null(0.0).to_numpy()

    def weights(self, idx: np.ndarray) -> np.ndarray:
        return m.item_weights([self.y[i] for i in idx], self.lw, self.pi[idx])

    def score(self, preds: pl.DataFrame) -> dict[str, object]:
        p, _ = self.align(preds)
        full = np.arange(len(self.y))
        point = _stats(self.y, p, self.weights(full))
        draws: dict[str, list[float]] = {k: [] for k in point}
        for idx in self.boot_idx:
            st = _stats([self.y[i] for i in idx], [p[i] for i in idx], self.weights(idx))
            for k, v in st.items():
                draws[k].append(v)
        out: dict[str, object] = {"n": len(self.y)}
        for k, v in point.items():
            lo, hi = m.ci(np.asarray(draws[k]))
            out[k] = {"est": v, "lo": lo, "hi": hi}
        return out

    def frontier(self, preds: pl.DataFrame) -> pl.DataFrame:
        """Weighted selective accuracy vs weighted coverage, with bootstrap bands."""
        p, conf = self.align(preds)
        full = np.arange(len(self.y))
        cc = m.coverage_curve(self.y, p, conf, self.weights(full), THRESHOLDS)
        bands = []
        for idx in self.boot_idx:
            b = m.coverage_curve(
                [self.y[i] for i in idx],
                [p[i] for i in idx],
                conf[idx],
                self.weights(idx),
                THRESHOLDS,
            )
            bands.append(b.select("coverage_w", "acc_w").to_numpy())
        arr = np.stack(bands)
        with warnings.catch_warnings():  # thresholds where nothing is covered are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanquantile(arr, [0.025, 0.975], axis=0)
        return cc.with_columns(
            pl.Series("coverage_w_lo", q[0, :, 0]),
            pl.Series("coverage_w_hi", q[1, :, 0]),
            pl.Series("acc_w_lo", q[0, :, 1]),
            pl.Series("acc_w_hi", q[1, :, 1]),
        )

    def paired(self, a: pl.DataFrame, b: pl.DataFrame, stat: str = "acc_w") -> dict[str, float]:
        """Difference a - b on the same resamples; ``p_le_0`` = share of draws with a <= b."""
        pa, _ = self.align(a)
        pb, _ = self.align(b)
        full = np.arange(len(self.y))
        est = (
            _stats(self.y, pa, self.weights(full))[stat]
            - _stats(self.y, pb, self.weights(full))[stat]
        )
        d = []
        for idx in self.boot_idx:
            w = self.weights(idx)
            yy = [self.y[i] for i in idx]
            d.append(
                _stats(yy, [pa[i] for i in idx], w)[stat]
                - _stats(yy, [pb[i] for i in idx], w)[stat]
            )
        arr = np.asarray(d)
        lo, hi = m.ci(arr)
        return {"est": est, "lo": lo, "hi": hi, "p_le_0": float((arr <= 0).mean())}
