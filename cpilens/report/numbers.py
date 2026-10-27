"""Every number the paper, brief and README print: results tables -> ``paper/numbers.tex``.

``describe`` writes ``results/data_summary.json`` (corpus and sample sizes). ``macros`` turns
the results that exist into LaTeX macros; a result that does not exist yet has no macro, so a
paper that cites it fails to build instead of printing a stale or typed-in number.

Macro names are letters only: ``\\AccW<Method>`` (+``Lo``/``Hi``), ``\\Bp<Kind><Stat><Agg>
<Method>``, ``\\Ceil...`` for RQ4, ``\\N...`` for counts. Percentages are printed without the
sign (the text writes ``\\AccWHaiku\\%``).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.paths import INTERIM, RESULTS, ROOT, TAXONOMY

OUT = ROOT / "paper" / "numbers.tex"

ALIAS = {
    "claude-haiku-5-5": "Haiku",
    "claude-opus-5-5": "Opus",
    "e5-large__mlp__nonthai+thaiweak+mocall": "EfiveMLP",
    "bge-m3__mlp__nonthai+thaiweak+mocall": "BgeMLP",
    "xlmr__finetune__nonthai+thaiweak+mocall": "XlmrFT",
    # learning-curve end points
    "e5-large__mlp__nonthai": "EfiveMLPZeroShot",
    "e5-large__mlp__nonthai+thaiweak": "EfiveMLPThaiWeak",
    "e5-large__mlp__mocall": "EfiveMLPMocOnly",
}
for _c in CASCADE_CHEAP:
    for _l in LLMS:
        ALIAS[f"cascade__{_c}__{_l}"] = f"Cas{ALIAS[_c]}{ALIAS[_l]}"

AGG = {"headline": "Headline", "food": "Food", "core": "Core"}
BP_STATS = {"rmse_bp": "Rmse", "max_abs_bp": "MaxAbs", "mean_bp": "Mean"}
VARIANT = {"full": "", "noise_s0.05": "NoiseFive", "noise_s0.10": "NoiseTen"}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def num(x: float, d: int = 0) -> str:
    """Thousands separated the LaTeX way (``76{,}269``)."""
    return f"{x:,.{d}f}".replace(",", "{,}")


def signed(x: float, d: int = 0) -> str:
    """A typographic minus, and no ``-0`` from rounding."""
    s = f"{x:.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"
    return s.replace("-", "\\textminus{}")


def describe(out: Path = RESULTS) -> dict[str, object]:
    from cpilens.classify.data import corpus

    o = pl.read_parquet(INTERIM / "off_products.parquet")
    thai = o.filter(pl.col("market") == "thailand")
    c = corpus()
    moc = c.filter(pl.col("source") == "moc")
    weak = c.filter(pl.col("label_kind") == "weak")
    s = pl.read_csv(ROOT / "data" / "gold" / "sample.csv", schema_overrides={"barcode": pl.Utf8})
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    cov = pl.read_csv(RESULTS / "coverage.csv").filter(pl.col("label") != "OOS")
    d: dict[str, object] = {
        "off_products": o.height,
        "off_markets": o["market"].n_unique(),
        "off_thailand": thai.height,
        "off_thailand_thai_script": int(thai["name_th"].is_not_null().sum()),
        "off_thailand_silver": int(thai["silver_label"].is_not_null().sum()),
        "corpus_texts": c.height,
        "weak_nonthai": weak.filter(pl.col("region") != "thai").height,
        "weak_thai": weak.filter(pl.col("region") == "thai").height,
        "moc_specs": moc.height,
        "moc_test": moc.filter(pl.col("split") == "test").height,
        "moc_trainval": moc.filter(pl.col("split") != "test").height,
        "gold_sample": s.height,
        "gold_double": int(s["double"].sum()),
        "gold_strata": s["stratum"].n_unique(),
        "labels_in_scope": int(scope["in_scope"].sum()),
        "in_scope_weight_equal": float(cov["w_equal"].sum()),
        "in_scope_weight_nnls": float(cov["w_nnls"].sum()),
    }
    (out / "data_summary.json").write_text(json.dumps(d, indent=2))
    return d


def _data(r: Path) -> dict[str, str]:
    f = r / "data_summary.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    m = {f"N{_camel(k)}": num(v) for k, v in d.items() if isinstance(v, int)}
    m["InScopeWeightEqual"] = f"{d['in_scope_weight_equal']:.1f}"
    m["InScopeWeightNnls"] = f"{d['in_scope_weight_nnls']:.1f}"
    return m


def _costs(r: Path) -> dict[str, str]:
    f = r / "llm_costs.csv"
    if not f.exists():
        return {}
    c = pl.read_csv(f)
    return {"LLMSpendUSD": f"{c['usd'].sum():.2f}", "NLLMCalls": num(c.height)}


def _gold(r: Path) -> dict[str, str]:
    f = r / "gold_scores.csv"
    if not f.exists():
        return {}
    m: dict[str, str] = {}
    meta = json.loads((r / "gold_summary.json").read_text())
    for k in ("n_labelled", "n_unk", "n_unsure", "n_scored", "n_boot"):
        m[f"NGold{_camel(k.removeprefix('n_'))}"] = num(meta[k])
    s = pl.read_csv(f).filter(pl.col("weights") == "equal")
    for row in s.iter_rows(named=True):
        a = ALIAS.get(row["method"])
        if a is None:
            continue
        for stat, tag in (("acc_w", "AccW"), ("acc", "Acc"), ("macro_f1", "MacroF")):
            m[f"{tag}{a}"] = pct(row[stat])
            m[f"{tag}{a}Lo"] = pct(row[f"{stat}_lo"])
            m[f"{tag}{a}Hi"] = pct(row[f"{stat}_hi"])
        m[f"Cost{a}"] = f"{row['usd_per_million']:,.2f}"
        if row["routed"] is not None:
            m[f"Routed{a}"] = pct(row["routed"], 0)
            m[f"Threshold{a}"] = f"{row['threshold']:.2f}"
    return m


def _rq3(r: Path) -> dict[str, str]:
    f = r / "rq3_errors.csv"
    if not f.exists():
        return {}
    e = pl.read_csv(f).filter(pl.col("weights") == "equal")
    m: dict[str, str] = {}
    for row in e.iter_rows(named=True):
        a = "Perfect" if row["variant"] == "coverage_only" else ALIAS.get(row["method"])
        if a is None or row["variant"] not in {*VARIANT, "coverage_only"}:
            continue
        v = VARIANT.get(row["variant"], "")
        for stat, tag in BP_STATS.items():
            name = f"Bp{row['kind'].title()}{tag}{AGG[row['aggregate']]}{a}{v}"
            m[name] = signed(row[stat])
            if row.get(f"{stat}_lo") is not None:
                m[f"{name}Lo"] = signed(row[f"{stat}_lo"])
                m[f"{name}Hi"] = signed(row[f"{stat}_hi"])
    return m


def _rq4(r: Path) -> dict[str, str]:
    f = r / "rq4_agreement.csv"
    if not f.exists():
        return {}
    m: dict[str, str] = {}
    for row in pl.read_csv(f).iter_rows(named=True):
        lv = row["level"].title()
        # disagreement = 1 - agreement, so the interval bounds swap
        m[f"Ceil{lv}Disagree"] = pct(1 - row["agree"])
        m[f"Ceil{lv}DisagreeLo"] = pct(1 - row["agree_hi"])
        m[f"Ceil{lv}DisagreeHi"] = pct(1 - row["agree_lo"])
        m[f"Ceil{lv}AccW"] = pct(row["acc_w"])
        m[f"Ceil{lv}Kappa"] = f"{row['kappa']:.2f}"
        m[f"Ceil{lv}KappaLo"] = f"{row['kappa_lo']:.2f}"
        m[f"Ceil{lv}KappaHi"] = f"{row['kappa_hi']:.2f}"
    meta = json.loads((r / "rq4_meta.json").read_text())
    m["NCeilPairs"] = num(meta["n_pairs"])
    m["NCeilUnk"] = num(meta["n_unk_either"])
    m["NCeilUnsure"] = num(meta["n_unsure_either"])
    return m


def _camel(key: str) -> str:
    return "".join(p.title() for p in key.split("_"))


def macros(results: Path = RESULTS) -> dict[str, str]:
    m: dict[str, str] = {}
    for part in (_data, _costs, _gold, _rq3, _rq4):
        m |= part(results)
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    return m


def write(results: Path = RESULTS, out: Path = OUT) -> int:
    m = macros(results)
    lines = ["% Generated by `cpilens report` from results/. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return len(m)
