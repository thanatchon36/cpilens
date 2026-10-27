"""Gold-label storage and the double-labelling protocol.

Pass 1 labels every sampled product in the sample's random ``order``. Pass 2 relabels the
``double`` subset blind: in a different random order, never showing pass-1 labels, and only
for products whose pass-1 label is at least 48 hours old (MISSION §3.3). Every click is
appended to ``labels_pass{n}.csv``; the latest row per product is the label, earlier rows
are the audit trail.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.label.sampling import SEED
from cpilens.paths import DATA

GOLD = DATA / "gold"
SAMPLE = GOLD / "sample.csv"
MIN_GAP = timedelta(hours=48)
FIELDS = ["order", "barcode", "label", "unsure", "seconds", "labelled_utc"]
SPECIAL = {"OOS": "out of scope", "UNK": "cannot tell from the information shown"}


def labels_file(pass_no: int, gold: Path = GOLD) -> Path:
    return gold / f"labels_pass{pass_no}.csv"


def read_labels(pass_no: int, gold: Path = GOLD) -> pl.DataFrame:
    """Latest label per product for a pass (empty frame if nothing labelled yet)."""
    f = labels_file(pass_no, gold)
    if not f.exists():
        return pl.DataFrame(schema={c: pl.Utf8 for c in FIELDS})
    df = pl.read_csv(f, schema_overrides={c: pl.Utf8 for c in FIELDS})
    return df.unique("order", keep="last", maintain_order=True)


def append_label(pass_no: int, row: dict[str, object], gold: Path = GOLD) -> None:
    f = labels_file(pass_no, gold)
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def queue(pass_no: int, now: datetime | None = None, gold: Path = GOLD) -> pl.DataFrame:
    """Products to show in this pass, in presentation order."""
    s = pl.read_csv(gold / SAMPLE.name, schema_overrides={"barcode": pl.Utf8, "order": pl.Utf8})
    if pass_no == 1:
        return s.sort(pl.col("order").cast(pl.Int64))
    if pass_no != 2:
        raise ValueError(pass_no)
    now = now or datetime.now(UTC)
    first = read_labels(1, gold).select("order", "labelled_utc")
    ready = {o for o, t in first.rows() if now - datetime.fromisoformat(t) >= MIN_GAP}
    d = s.filter(pl.col("double")).sort(pl.col("order").cast(pl.Int64))
    perm = np.random.default_rng(SEED + 2).permutation(d.height)
    d = d[perm]
    return d.filter(pl.col("order").is_in(list(ready)))
