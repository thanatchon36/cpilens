"""One text normalisation for every model: NFC, HTML entities, invisible characters, spacing.

Product text is built from separate fields so the brand/quantity ablation can switch them
off: ``name | brand | quantity``. Brands are de-duplicated (OFF often repeats a brand in
Latin and Thai script with different case).
"""

from __future__ import annotations

import html
import re
import unicodedata

INVISIBLE = re.compile(r"[​-‏⁠﻿­]")
SPACE = re.compile(r"\s+")


def clean(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", html.unescape(s))
    s = INVISIBLE.sub("", s)
    return SPACE.sub(" ", s).strip(" ,;/|")


def brands(s: str | None) -> str:
    seen: dict[str, str] = {}
    for b in clean(s).split(","):
        b = b.strip()
        if b and b.casefold() not in seen:
            seen[b.casefold()] = b
    return ", ".join(seen.values())


def product_text(
    name: str | None,
    brand: str | None = None,
    quantity: str | None = None,
    with_brand: bool = True,
    with_quantity: bool = True,
) -> str:
    parts = [clean(name)]
    if with_brand and (b := brands(brand)):
        parts.append(b)
    if with_quantity and (q := clean(quantity)):
        parts.append(q)
    return " | ".join(p for p in parts if p)
