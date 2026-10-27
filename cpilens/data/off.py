"""Product extracts from the Open Food Facts / Open Beauty Facts Parquet dumps (ODbL).

One row per product: barcode, source, market, names (Thai-script, English, main language), brand,
quantity and the category tags used for silver labels. Thai products are kept whenever they
have a name (they are the gold-sampling frame); other markets need a name and a category.
Europe is a fixed random sample, used only as a high-resource source for transfer (RQ1).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from cpilens.paths import INTERIM, RAW

OFF_DIR = RAW / "off"
OUT = INTERIM / "off_products.parquet"
THAI = "en:thailand"
ASIA = [
    "en:japan",
    "en:india",
    "en:singapore",
    "en:philippines",
    "en:indonesia",
    "en:malaysia",
    "en:hong-kong",
    "en:taiwan",
    "en:south-korea",
    "en:china",
    "en:vietnam",
    "en:cambodia",
]
EUROPE = ["en:france", "en:germany", "en:united-kingdom", "en:italy", "en:spain"]
EUROPE_SAMPLE = 40_000
SEED = 20261027

_THAI_SCRIPT = "regexp_matches(x.text, '[\u0e00-\u0e7f]')"  # Thai block


def _name(lang_expr: str) -> str:
    return (
        f"list_filter(product_name, lambda x: x.lang = {lang_expr} "
        "AND length(trim(x.text)) > 0)[1].text"
    )


def _select(path: Path, source: str) -> str:
    markets = [THAI, *ASIA, *EUROPE]
    case = " ".join(f"WHEN list_contains(countries_tags, '{m}') THEN '{m[3:]}'" for m in markets)
    return f"""
        SELECT code AS barcode, '{source}' AS source, CASE {case} END AS market,
               coalesce(
                   list_filter(product_name, lambda x: x.lang = 'th' AND {_THAI_SCRIPT})[1].text,
                   list_filter(product_name, lambda x: {_THAI_SCRIPT})[1].text) AS name_th,
               {_name("'en'")} AS name_en,
               coalesce({_name("lang")},
                        list_filter(product_name, lambda x: length(trim(x.text)) > 0)[1].text)
                   AS name_main,
               lang AS main_lang, brands, quantity, categories_tags
        FROM read_parquet('{path}')
        WHERE len(list_filter(product_name, lambda x: length(trim(x.text)) > 0)) > 0
    """


def extract(off_dir: Path = OFF_DIR, out: Path = OUT) -> Path:
    con = duckdb.connect()
    parts = " UNION ALL ".join(
        _select(off_dir / f, src) for f, src in [("food.parquet", "off"), ("beauty.parquet", "obf")]
    )
    con.sql(f"CREATE TABLE p AS SELECT * FROM ({parts}) WHERE market IS NOT NULL")
    europe = ", ".join(f"'{m[3:]}'" for m in EUROPE)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"""
        COPY (
          SELECT * FROM p WHERE market = 'thailand'
          UNION ALL
          SELECT * FROM p WHERE market NOT IN ('thailand', {europe}) AND len(categories_tags) > 0
          UNION ALL
          SELECT * FROM (SELECT * FROM p WHERE market IN ({europe}) AND len(categories_tags) > 0
                         ORDER BY hash(barcode || '{SEED}') LIMIT {EUROPE_SAMPLE})
          ORDER BY market, source, barcode
        ) TO '{out}' (FORMAT parquet)
    """)
    return out
