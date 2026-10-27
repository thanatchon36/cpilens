"""Parsers and the silver-label crosswalk on small synthetic inputs."""

from __future__ import annotations

import polars as pl

from cpilens.data.moc import label_specs, parse_record
from cpilens.label.sampling import allocate
from cpilens.taxonomy.crosswalk import Crosswalk


def test_parse_spec_record() -> None:
    r = parse_record(
        {
            "id": "1152205111050201",
            "name": "1152205111050201 - ซอสพริก : ซอสพริก \\บรรจุขวด 310 กรัม \\ตราไฮนซ์ (310 กรัม)",
        }
    )
    assert r["code5"] == "11522" and not r["is_heading"]
    assert r["item_th"] == "ซอสพริก"
    assert r["text"] == "ซอสพริก / บรรจุขวด 310 กรัม / ตราไฮนซ์ (310 กรัม)"
    assert r["unit"] == "310 กรัม"


def test_parse_heading_and_odd_separators() -> None:
    assert parse_record({"id": "1111001000000000", "name": "1111001000000000 - ข้าวสารเจ้า : -"})[
        "is_heading"
    ]
    r = parse_record({"id": "4111016311020280", "name": "4111016311020280 : ยาวิตามินซี (1 ขวด)"})
    assert r["text"] == "ยาวิตามินซี (1 ขวด)" and r["item_th"] == ""


def test_label_specs_scope_and_old_codes() -> None:
    specs = pl.DataFrame({"code5": ["11522", "41110", "11510"], "text": ["a", "b", "c"]})
    scope = pl.DataFrame({"label": ["11522", "OOS"], "in_scope": [True, False]})
    out = label_specs(specs, scope, ["11522", "41110"])
    assert out["label"].to_list() == ["11522", "OOS"]


def crosswalk() -> Crosswalk:
    parents = {
        "en:beverages": [],
        "en:alcoholic-beverages": ["en:beverages"],
        "en:beers": ["en:alcoholic-beverages"],
        "en:snacks": [],
        "en:salty-snacks": ["en:snacks"],
        "en:nuts": ["en:plant"],
        "en:plant": [],
        "en:salted-nuts": ["en:nuts", "en:salty-snacks"],
        "en:ciders": ["en:alcoholic-beverages"],
    }
    mapping = {
        "en:beverages": "11600",
        "en:alcoholic-beverages": "?",
        "en:beers": "72100",
        "en:salty-snacks": "11122",
        "en:nuts": "11412",
        "en:plant": "11412",
    }
    return Crosswalk(mapping, parents, {"en:salty-snacks": 1})


def test_crosswalk_deepest_wins_and_ambiguity_blocks() -> None:
    cw = crosswalk()
    assert cw.label(["en:beers"]) == "72100"
    assert cw.label(["en:ciders"]) is None  # blocked by "?" on alcoholic-beverages
    assert cw.label(["en:beverages"]) == "11600"
    assert cw.label([]) is None and cw.label(None) is None


def test_crosswalk_priority_beats_depth() -> None:
    assert crosswalk().label(["en:salted-nuts"]) == "11122"


def test_allocation_sums_and_respects_caps() -> None:
    sizes = {"a": 1000, "b": 50, "c": 5, "none": 3000}
    a = allocate(sizes, {"a": 10.0, "b": 30.0, "c": 1.0}, n=600, floor=10)
    assert sum(a.values()) == 600
    assert a["c"] == 5 and a["b"] <= 50
    assert all(v >= min(10, sizes[h]) for h, v in a.items())
