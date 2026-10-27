"""Current-structure Thai CPI hierarchy (base 2566) from the public index codes.

Codes are five digits; a code's parent is the nearest code obtained by zeroing trailing
digits. Divisions are 10000..70000 under 00000. Codes starting 8 or 9 are analytic
aggregates (non-food, core, fresh food, energy) and are not part of the tree. A code is
"current" if it has an index in the latest month; older codes belong to the back-series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

ROOT = "00000"


def parent_code(code: str, pool: set[str]) -> str | None:
    if code == ROOT:
        return None
    s = code.rstrip("0")
    while s:
        s = s[:-1]
        p = s.ljust(5, "0")
        if p in pool:
            return p
    return ROOT


@dataclass(frozen=True)
class Hierarchy:
    names: dict[str, str]
    parent: dict[str, str | None]
    children: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_index(cls, index: pl.DataFrame) -> Hierarchy:
        latest = index["date"].max()
        cur = index.filter(
            (pl.col("date") == latest) & pl.col("code").str.slice(0, 1).is_in(list("01234567"))
        )
        names = dict(zip(cur["code"], cur["name_th"], strict=True))
        pool = set(names)
        parent = {c: parent_code(c, pool) for c in sorted(pool)}
        children: dict[str, list[str]] = {}
        for c, p in parent.items():
            if p is not None:
                children.setdefault(p, []).append(c)
        return cls(names=names, parent=parent, children=children)

    @property
    def leaves(self) -> list[str]:
        return sorted(c for c in self.names if c not in self.children)

    def ancestors(self, code: str) -> list[str]:
        out, p = [], self.parent[code]
        while p is not None:
            out.append(p)
            p = self.parent[p]
        return out

    def division(self, code: str) -> str:
        return code[0] + "0000"
