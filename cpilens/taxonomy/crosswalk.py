"""Silver labels for Open Food Facts products from a hand-curated category crosswalk.

``taxonomy/off_categories_to_thcpi.csv`` maps OFF category tags to a CPI leaf, ``OOS`` (out
of scope) or ``?`` (ambiguous: blocks inheritance from more general tags). A product's tags
are expanded with their taxonomy ancestors; among the mapped tags, the highest ``priority``
wins, then the deepest. Priority 1 marks product-type groups (snacks, bakery, sweets, ready
meals, supplements) that TPSO classifies by type, not by main ingredient. If the winning tags
disagree, or the decision is ``?``, the product has no silver label.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import polars as pl

from cpilens.paths import RAW, TAXONOMY

CROSSWALK = TAXONOMY / "off_categories_to_thcpi.csv"
OFF_TAXONOMY = RAW / "off" / "categories.json"
AMBIGUOUS = "?"


class Crosswalk:
    def __init__(
        self,
        mapping: dict[str, str],
        parents: dict[str, list[str]],
        priority: dict[str, int] | None = None,
    ) -> None:
        self.mapping = mapping
        self.priority = priority or {}
        self.parents = parents
        self.depth = cache(self._depth)

    @classmethod
    def load(cls, crosswalk: Path = CROSSWALK, taxonomy: Path = OFF_TAXONOMY) -> Crosswalk:
        m = pl.read_csv(crosswalk, schema_overrides={"label": pl.Utf8})
        tax = json.loads(taxonomy.read_text())
        parents = {k: v.get("parents", []) for k, v in tax.items()}
        return cls(
            dict(zip(m["tag"], m["label"], strict=True)),
            parents,
            dict(zip(m["tag"], m["priority"], strict=True)),
        )

    def _depth(self, tag: str) -> int:
        ps = [p for p in self.parents.get(tag, []) if p != tag]
        return 1 + max((self.depth(p) for p in ps), default=0)

    def expand(self, tags: list[str]) -> set[str]:
        out, todo = set(), list(tags)
        while todo:
            t = todo.pop()
            if t not in out:
                out.add(t)
                todo += self.parents.get(t, [])
        return out

    def label(self, tags: list[str] | None) -> str | None:
        mapped = [t for t in self.expand(tags or []) if t in self.mapping]
        if not mapped:
            return None

        def key(t: str) -> tuple[int, int]:
            return self.priority.get(t, 0), self.depth(t)

        top = max(key(t) for t in mapped)
        labels = {self.mapping[t] for t in mapped if key(t) == top}
        if len(labels) != 1 or AMBIGUOUS in labels:
            return None
        return labels.pop()

    def apply(self, products: pl.DataFrame) -> pl.DataFrame:
        return products.with_columns(
            pl.col("categories_tags")
            .map_elements(lambda t: self.label(list(t)), return_dtype=pl.Utf8)
            .alias("silver_label")
        )
