"""Paper numbers: formatting and macro generation."""

from __future__ import annotations

import json

import polars as pl

from cpilens.report import numbers


def test_formatting():
    assert numbers.num(76269) == "76{,}269"
    assert numbers.pct(0.7392) == "73.9"
    assert numbers.signed(-12.6) == "\\textminus{}13"
    assert numbers.signed(-0.4) == "0"


def test_macros_from_results(tmp_path):
    (tmp_path / "data_summary.json").write_text(
        json.dumps(
            {"gold_sample": 600, "in_scope_weight_equal": 40.46, "in_scope_weight_nnls": 43.5}
        )
    )
    (tmp_path / "gold_summary.json").write_text(
        json.dumps({"n_labelled": 600, "n_unk": 3, "n_unsure": 9, "n_scored": 597, "n_boot": 1000})
    )
    row = {
        "method": "claude-haiku-5-5",
        "weights": "equal",
        "usd_per_million": 79.1,
        "routed": None,
        "threshold": None,
    }
    for s in ("acc", "acc_w", "macro_f1"):
        row |= {s: 0.5, f"{s}_lo": 0.4, f"{s}_hi": 0.6}
    pl.DataFrame([row]).write_csv(tmp_path / "gold_scores.csv")
    m = numbers.macros(tmp_path)
    assert m["NGoldSample"] == "600" and m["InScopeWeightEqual"] == "40.5"
    assert (m["AccWHaiku"], m["AccWHaikuLo"], m["CostHaiku"]) == ("50.0", "40.0", "79.10")
    assert m["NGoldUnk"] == "3"
    assert "RoutedHaiku" not in m
    out = tmp_path / "numbers.tex"
    numbers.write(tmp_path, out)
    assert "\\newcommand{\\AccWHaiku}{50.0}" in out.read_text()
