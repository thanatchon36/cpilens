"""The modelling corpus: one row per text, with its label, provenance and split.

Sources and label kinds:
- OFF/OBF products with a crosswalk label (``weak``), grouped as ``thai``, ``asia``, ``europe``;
- MOC price-collection specifications with labels from their official codes (``official``);
- the gold sample (``gold``; labels joined later from ``data/gold``), evaluation only.

Leakage guards: every gold barcode is removed from the training pool, and so is any
training row whose normalised text equals a gold text. MOC splits are by 7-digit item code,
so specifications of the same item never straddle train and test. Splits are deterministic
hashes with a fixed salt.
"""

from __future__ import annotations

import hashlib

import polars as pl

from cpilens.embed.text import clean, product_text
from cpilens.paths import DATA, INTERIM

SALT = "cpilens-20261027"
ASIA = {
    "japan",
    "india",
    "singapore",
    "philippines",
    "indonesia",
    "malaysia",
    "hong-kong",
    "taiwan",
    "south-korea",
    "china",
    "vietnam",
    "cambodia",
}


def bucket(key: str, n: int = 100) -> int:
    return int(hashlib.sha1(f"{SALT}:{key}".encode()).hexdigest(), 16) % n


def _split(b: int, cuts: tuple[int, int]) -> str:
    return "train" if b < cuts[0] else "val" if b < cuts[1] else "test"


def off_rows(with_brand: bool = True, with_quantity: bool = True) -> pl.DataFrame:
    p = pl.read_parquet(INTERIM / "off_products.parquet")
    p = p.unique("barcode", keep="first", maintain_order=True)
    texts = [
        product_text(th or main, b, q, with_brand, with_quantity)
        for th, main, b, q in p.select("name_th", "name_main", "brands", "quantity").rows()
    ]
    region = [
        ("thai" if m == "thailand" else "asia" if m in ASIA else "europe") for m in p["market"]
    ]
    return pl.DataFrame(
        {
            "id": "off:" + p["barcode"],
            "barcode": p["barcode"],
            "source": p["source"],
            "region": region,
            "text": texts,
            "label": p["silver_label"],
            "label_kind": "weak",
        }
    ).with_columns(
        pl.col("id")
        .map_elements(lambda k: _split(bucket(k), (90, 100)), return_dtype=pl.Utf8)
        .alias("split")
    )


def moc_rows() -> pl.DataFrame:
    m = pl.read_parquet(INTERIM / "moc_specs.parquet")
    m = m.with_columns(pl.col("text").map_elements(clean, return_dtype=pl.Utf8))
    m = m.unique("text", keep="first", maintain_order=True)
    return pl.DataFrame(
        {
            "id": "moc:" + m["spec_id"],
            "barcode": None,
            "source": "moc",
            "region": "thai",
            "text": m["text"],
            "label": m["label"],
            "label_kind": "official",
            "split": [_split(bucket(c), (60, 80)) for c in m["item_code"]],
        },
        schema_overrides={"barcode": pl.Utf8},
    )


def corpus(with_brand: bool = True, with_quantity: bool = True) -> pl.DataFrame:
    gold = pl.read_csv(DATA / "gold" / "sample.csv", schema_overrides={"barcode": pl.Utf8})
    off = off_rows(with_brand, with_quantity)
    gold_ids = set(gold["barcode"])
    is_gold = off["barcode"].is_in(list(gold_ids))
    g = off.filter(is_gold).with_columns(
        pl.lit("gold").alias("split"),
        pl.lit("gold").alias("label_kind"),
        pl.lit(None, pl.Utf8).alias("label"),
    )
    gold_texts = set(g["text"])
    train_pool = pl.concat([off.filter(~is_gold & pl.col("label").is_not_null()), moc_rows()])
    train_pool = train_pool.filter(~pl.col("text").is_in(list(gold_texts)))
    return pl.concat([train_pool, g]).sort("id")
