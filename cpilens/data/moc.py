"""TPSO price-collection specifications from the Ministry of Commerce open-data portal.

Each record is ``{"id": 16-digit code, "name": "<id> - <item heading> : <specification>"}``.
The first five digits are the CPI index code, the first seven the item. Records whose last
nine digits are zero are item headings, not specifications. The item heading is TPSO's own
item name, so it is kept as metadata and never used as model input (it would leak the label).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from cpilens.paths import RAW

MOC_FILE = RAW / "moc" / "getCPIProducts.json"
UNIT = re.compile(r"\(([^()]*)\)\s*$")


def parse_record(rec: dict[str, str]) -> dict[str, object]:
    code = rec["id"]
    body = rec["name"].removeprefix(code).lstrip(" -")  # a few records lack " - " or " : "
    item, sep, spec = body.partition(" : ")
    if not sep and body.startswith(":"):
        item, spec = "", body[1:]
    spec = " ".join(spec.replace("\\", " / ").split()).strip(" -/")
    unit = UNIT.search(spec)
    return {
        "spec_id": code,
        "code5": code[:5],
        "item_code": code[:7],
        "is_heading": code[7:] == "0" * 9,
        "item_th": item.strip(),
        "text": spec,
        "unit": unit[1].strip() if unit else None,
    }


def load_specs(path: Path = MOC_FILE) -> pl.DataFrame:
    """Specifications only (headings dropped), one row per spec, with duplicates removed."""
    df = pl.DataFrame([parse_record(r) for r in json.loads(path.read_text())])
    return (
        df.filter(~pl.col("is_heading") & (pl.col("text").str.len_chars() > 0))
        .drop("is_heading")
        .unique("spec_id")
        .sort("spec_id")
    )


def label_specs(
    specs: pl.DataFrame, scope: pl.DataFrame, current_leaves: list[str]
) -> pl.DataFrame:
    """Gold label for each spec from its index code: in-scope leaf, ``OOS`` or dropped.

    Specs under codes that are not current leaves (earlier bases) are dropped; specs under
    current leaves outside the scope are ``OOS``.
    """
    in_scope = scope.filter(pl.col("in_scope"))["label"].to_list()
    return specs.filter(pl.col("code5").is_in(current_leaves)).with_columns(
        pl.when(pl.col("code5").is_in(in_scope))
        .then(pl.col("code5"))
        .otherwise(pl.lit("OOS"))
        .alias("label")
    )
