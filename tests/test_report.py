"""Paper numbers: formatting and macro generation."""

from __future__ import annotations

import json

import polars as pl
import pytest

from cpilens.report import numbers


def test_formatting():
    assert numbers.num(76269) == "76{,}269"
    assert numbers.pct(0.7392) == "73.9"
    assert numbers.signed(-12.6) == "\\textminus{}13"
    assert numbers.signed(-0.4) == "0"
    assert numbers.sci(3e-5) == "$3\\times 10^{-5}$" and numbers.sci(1e-4) == "$10^{-4}$"


def test_macros_from_results(tmp_path):
    pytest.importorskip("torch")  # the Cfg macros are read from the model code (ml group)
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


def test_costs_table_sums_by_model_and_stage(tmp_path):
    from cpilens.report import numbers

    (tmp_path / "llm_costs.csv").write_text(
        "utc,model,purpose,key,input,cache_write,cache_read,output,usd\n"
        "t,m1,eval:m1,a,10,0,90,5,0.5\n"
        "t,m1,eval:m1,b,10,0,90,5,0.25\n"
        "t,m1,smoke,c,1,0,0,1,0.01\n"
    )
    rows = numbers.costs_table(tmp_path).split("\n")
    assert rows[0] == "m1 & eval & 2 & 200 & 10 & 0.75 \\\\"
    assert rows[-1] == "\\midrule Total & & 3 & 201 & 11 & 0.76 \\\\"


def test_manifest_reads_hf_refs_and_cached_llm_models(tmp_path):
    from cpilens.report import manifest

    ref = tmp_path / "hub" / "models--org--m" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc123\n")
    assert manifest.hf_revision("org/m", tmp_path / "hub") == "abc123"
    assert manifest.hf_revision("org/none", tmp_path / "hub") is None
    cache = tmp_path / "llm"
    cache.mkdir()
    for i, m in enumerate(["b", "a", "a"]):
        (cache / f"{i}.json").write_text(f'{{"model": "{m}"}}')
    assert manifest.llm_models_returned(cache) == {"a": 2, "b": 1}
