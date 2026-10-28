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
import re
from pathlib import Path

import polars as pl

from cpilens.evaluate.gold import reference_table
from cpilens.evaluate.run import CASCADE_CHEAP, LLMS
from cpilens.paths import INTERIM, RESULTS, ROOT, TAXONOMY

OUT = ROOT / "paper" / "numbers.tex"
GENERATED = ROOT / "paper" / "generated"

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
    "bge-m3__mlp__nonthai": "BgeMLPZeroShot",
    "bge-m3__mlp__mocall": "BgeMLPMocOnly",
    "xlmr__finetune__nonthai": "XlmrFTZeroShot",
    # a low-coverage run with a small basis-point error (results text)
    "e5-large__linear__moc100": "EfiveLinMocHundred",
}
BUDGET_WORD = {100: "Hundred", 250: "TwoFifty", 500: "FiveHundred", 1000: "Thousand"}
for _n, _w in BUDGET_WORD.items():  # the learning curve of the e5-large MLP
    ALIAS[f"e5-large__mlp__nonthai+thaiweak+moc{_n}"] = f"EfiveMLPPlus{_w}"
    ALIAS[f"e5-large__mlp__moc{_n}"] = f"EfiveMLPMoc{_w}"
for _v, _w in (("nobrand", "NoBrand"), ("noqty", "NoQty"), ("nobrand-noqty", "NoBrandNoQty")):
    ALIAS[f"e5-large__mlp__nonthai+thaiweak+mocall-{_v}"] = f"EfiveMLP{_w}"
ALIAS["bge-m3__mlp__nonthai+thaiweak+mocall-nobrand"] = "BgeMLPNoBrand"
for _c in CASCADE_CHEAP:
    for _l in LLMS:
        ALIAS[f"cascade__{_c}__{_l}"] = f"Cas{ALIAS[_c]}{ALIAS[_l]}"

AGG = {"headline": "Headline", "food": "Food", "core": "Core"}
BP_STATS = {"rmse_bp": "Rmse", "max_abs_bp": "MaxAbs", "mean_bp": "Mean"}
VARIANT = {
    "full": "",
    "coverage_only": "",
    "noise_s0.05": "NoiseFive",
    "noise_s0.10": "NoiseTen",
    "cluster": "Cluster",
}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def num(x: float, d: int = 0) -> str:
    """Thousands separated the LaTeX way (``76{,}269``)."""
    return f"{x:,.{d}f}".replace(",", "{,}")


def sci(x: float) -> str:
    """``3e-05`` -> ``$3\\times 10^{-5}$`` (powers of ten print as ``$10^{-4}$``)."""
    mant, exp = f"{x:e}".split("e")
    m, e = float(mant), int(exp)
    return f"$10^{{{e}}}$" if m == 1 else f"${m:g}\\times 10^{{{e}}}$"


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
    from cpilens.classify.data import moc_test_duplicates

    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    cov = pl.read_csv(RESULTS / "coverage.csv", schema_overrides={"label": pl.Utf8}).filter(
        pl.col("label") != "OOS"
    )
    test = moc.filter(pl.col("split") == "test")
    ref_items = sorted(set(test["label"]) - {"OOS"})
    with_specs = set(moc["label"])
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
        "moc_test": test.height,
        "moc_test_items": len(ref_items),
        "moc_test_oos": test.filter(pl.col("label") == "OOS").height,
        "moc_trainval": moc.filter(pl.col("split") != "test").height,
        "moc_dup_dropped": moc_test_duplicates(),
        "items_without_specs": int((~cov["label"].is_in(list(with_specs))).sum()),
        "labels_in_scope": int(scope["in_scope"].sum()),
        "in_scope_weight_equal": float(cov["w_equal"].sum()),
        "in_scope_weight_nnls": float(cov["w_nnls"].sum()),
        "ref_weight_equal": float(cov.filter(pl.col("label").is_in(ref_items))["w_equal"].sum()),
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
    m["RefWeightEqual"] = f"{d['ref_weight_equal']:.1f}"
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
    for k in ("n_ref_specs", "n_ref_items", "n_ref_oos", "n_boot"):
        m[f"N{_camel(k.removeprefix('n_'))}"] = num(meta[k])
    m["NRefInScope"] = num(meta["n_ref_specs"] - meta["n_ref_oos"])
    sub = (
        reference_table()
        .filter(pl.col("label") != "OOS")
        .group_by("label")
        .agg(pl.col("cluster").n_unique())
    )
    m["NRefSubitems"] = num(int(sub["cluster"].sum()))
    m["NRefItemsOneSubitem"] = num(int((sub["cluster"] == 1).sum()))
    allw = pl.read_csv(f)
    for row in allw.filter(pl.col("weights") == "nnls").iter_rows(named=True):
        if (a := ALIAS.get(row["method"])) is not None:
            m[f"AccW{a}Nnls"] = pct(row["acc_w"])
            m[f"AccW{a}NnlsLo"] = pct(row["acc_w_lo"])
            m[f"AccW{a}NnlsHi"] = pct(row["acc_w_hi"])
    s = allw.filter(pl.col("weights") == "equal")
    cost = dict(zip(s["method"], s["usd_per_million"], strict=True))
    opus, haiku = cost.get("claude-opus-5-5"), cost.get("claude-haiku-5-5")
    if opus and haiku:
        m["CostRatioOpusHaiku"] = f"{opus / haiku:.0f}"
        for k, v in cost.items():
            if k.startswith("cascade__") and k.endswith("claude-opus-5-5"):
                m[f"CostPct{ALIAS[k]}"] = f"{100 * v / opus:.0f}"
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
            if row.get("routed_in_scope") is not None:
                m[f"Routed{a}InScope"] = pct(row["routed_in_scope"], 0)
                m[f"Routed{a}Oos"] = pct(row["routed_oos"], 0)
            m[f"Threshold{a}"] = f"{row['threshold']:.2f}"
    return m


def _rq3(r: Path) -> dict[str, str]:
    f = r / "rq3_errors.csv"
    if not f.exists():
        return {}
    e = pl.read_csv(f)
    if "universe" not in e.columns:
        e = e.with_columns(pl.lit("cpi").alias("universe"))
    m: dict[str, str] = {}
    meta = r / "rq3_meta.json"
    if meta.exists() and "n_clusters" in (d := json.loads(meta.read_text())):
        m["NClusters"] = num(d["n_clusters"])
    for kind, tag in (("yoy", "Yoy"), ("mom", "Mom")):
        k = e.filter((pl.col("kind") == kind) & (pl.col("weights") == "equal"))
        if k.height:
            m[f"NMonths{tag}"] = str(k["n_months"].cast(pl.Int64).max())
    ser = r / "rq3_series.csv"
    if ser.exists():
        d = pl.read_csv(ser, try_parse_dates=True)
        for kind, tag in (("yoy", "Yoy"), ("mom", "Mom")):
            dates = sorted(d.filter(pl.col("kind") == kind)["date"].unique().to_list())
            if dates:
                m[f"SimFirst{tag}"] = dates[0].strftime("%B %Y")
                m[f"SimLast{tag}"] = dates[-1].strftime("%B %Y")
    # the primary universe (products in proportion to CPI weights) and weights (equal split)
    # have no suffix
    uni = {"cpi": "", "list": "List"}
    wts = {"equal": "", "nnls": "Nnls"}
    for row in e.filter(pl.col("universe").is_in(list(uni))).iter_rows(named=True):
        a = "Perfect" if row["method"] == "perfect" else ALIAS.get(row["method"])
        if a is None or row["variant"] not in VARIANT:
            continue
        v = VARIANT[row["variant"]] + uni[row["universe"]] + wts[row["weights"]]
        for stat, tag in BP_STATS.items():
            name = f"Bp{row['kind'].title()}{tag}{AGG[row['aggregate']]}{a}{v}"
            m[name] = signed(row[stat], 1)
            if row.get(f"{stat}_lo") is not None:
                m[f"{name}Lo"] = signed(row[f"{stat}_lo"], 1)
                m[f"{name}Hi"] = signed(row[f"{stat}_hi"], 1)
    return m


def _paired(r: Path) -> dict[str, str]:
    """Paired differences in weighted accuracy, percentage points: ``\\Diff<A><B>``."""
    f = r / "gold_paired.csv"
    if not f.exists():
        return {}
    m: dict[str, str] = {}
    for row in pl.read_csv(f).iter_rows(named=True):
        a, b = ALIAS.get(row["a"]), ALIAS.get(row["b"])
        if a is None or b is None:
            continue
        tag = f"Diff{a}{b}" + ("Nnls" if row["weights"] == "nnls" else "")
        for k, suffix in (("est", ""), ("lo", "Lo"), ("hi", "Hi")):
            m[tag + suffix] = signed(100 * row[k], 1)
    return m


def _errors(r: Path) -> dict[str, str]:
    """Where the in-scope errors fall: by official sub-item (``gold_error_subitems.csv``)."""
    f = r / "gold_error_subitems.csv"
    if not f.exists():
        return {}
    e = pl.read_csv(f, schema_overrides={"item_code": pl.Utf8, "label": pl.Utf8})
    m: dict[str, str] = {}
    for name, d in e.group_by("method"):
        a = ALIAS.get(str(name[0]))
        if a is None:
            continue
        by = d.group_by("item_code").agg(pl.col("n").sum()).sort("n", descending=True)
        top, total = int(by["n"].head(3).sum()), int(by["n"].sum())
        m[f"NErr{a}"] = num(total)
        m[f"NErrTopThree{a}"] = num(top)
        m[f"PctErrTopThree{a}"] = pct(top / total, 0)
        m[f"NErrSubitems{a}"] = num(by.height)
    rec = r / "gold_label_recall.csv"
    if rec.exists():  # the specification counts behind "misses whole items"
        z = pl.read_csv(rec, schema_overrides={"label": pl.Utf8}).filter(pl.col("recall") == 0)
        for name, d in z.group_by("method"):
            if (a := ALIAS.get(str(name[0]))) is not None:
                m[f"NZeroRecallMin{a}"] = num(int(d["n"].min()))  # type: ignore[arg-type]
                m[f"NZeroRecallMax{a}"] = num(int(d["n"].max()))  # type: ignore[arg-type]
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


def _config(_: Path) -> dict[str, str]:
    """Design constants, read from the code that uses them."""
    import inspect

    from cpilens.classify import config, llm
    from cpilens.econ import run as econ
    from cpilens.embed import projection
    from cpilens.evaluate import frontier, gold, metrics

    def default(f: object, arg: str) -> object:
        return inspect.signature(f).parameters[arg].default  # type: ignore[arg-type]

    cfg = config.TrainCfg()
    budgets = [b for b in config.BUDGETS if b != "all"]
    return {
        "CfgBoot": num(gold.N_BOOT),
        "CfgOosN": num(gold.OOS_N),
        "CfgDraws": num(econ.N_DRAWS),
        "CfgSeed": str(gold.SEED),
        "CfgAlpha": f"{econ.ALPHA:g}",
        "CfgSigmaLo": pct(econ.SIGMAS[0], 0),
        "CfgSigmaHi": pct(econ.SIGMAS[1], 0),
        "CfgMinFlowBp": f"{econ.MIN_FLOW_BP:g}",
        "CfgCapUSD": f"{llm.CAP_USD:.0f}",
        "CfgGpuUSDHour": f"{frontier.GPU_USD_PER_HOUR:.2f}",
        "CfgMLPHidden": num(config.MLP_HIDDEN),
        "CfgDropout": f"{cfg.dropout:g}",
        "CfgLR": sci(cfg.lr),
        "CfgWeightDecay": sci(cfg.weight_decay),
        "CfgBatch": num(cfg.batch),
        "CfgPatience": str(cfg.patience),
        "CfgKs": ", ".join(str(k) for k in config.KNN_KS),
        "CfgFtEpochs": str(config.FT_EPOCHS),
        "CfgFtBatch": str(config.FT_BATCH),
        "CfgFtLR": sci(config.FT_LR),
        "CfgFtMaxLen": str(config.FT_MAX_LEN),
        "CfgFewShot": str(default(llm.few_shot, "per_label")),
        "CfgMaxTokens": num(llm.MAX_TOKENS),
        "CfgBudgets": ", ".join(num(b) for b in budgets),  # type: ignore[arg-type]
        "CfgBudgetMax": num(max(int(b) for b in budgets)),
        "CfgCI": pct(default(metrics.ci, "level"), 0),  # type: ignore[arg-type]
        "CfgAlignK": str(projection.K),
        "CfgThresholds": ", ".join(f"{t:g}" for t in frontier.THRESHOLDS),
    }


def _names(_: Path) -> dict[str, str]:
    """Model names (they contain digits, so they too come from code)."""
    from cpilens.embed.encode import MODELS
    from cpilens.viz import names

    m = {f"Name{ALIAS[k]}": v for k, v in names.LLM.items()}
    m |= {
        "NameEfive": names.EMB["e5-large"],
        "NameBge": names.EMB["bge-m3"],
        "HfEfive": MODELS["e5-large"]["hf"],
        "HfBge": MODELS["bge-m3"]["hf"],
    }
    from cpilens.classify.config import FT_HF

    return m | {"HfXlmr": FT_HF}


def _camel(key: str) -> str:
    return "".join(p.title() for p in key.split("_"))


def _dev(r: Path) -> dict[str, str]:
    """The LLMs as the development set scored them: the sweep rows that route every product."""
    f = r / "cascade_dev.csv"
    if not f.exists():
        return {}
    d = pl.read_csv(f).filter((pl.col("threshold") > 1) & (pl.col("cheap") == CASCADE_CHEAP[0]))
    return {f"DevAccW{ALIAS[row['llm']]}": pct(row["acc_w"]) for row in d.iter_rows(named=True)}


def macros(results: Path = RESULTS) -> dict[str, str]:
    m: dict[str, str] = {}
    for part in (_config, _names, _data, _costs, _gold, _dev, _paired, _errors, _rq3, _rq4):
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


# ---------------------------------------------------------------- appendix tables


def _tex(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def items_table(results: Path = RESULTS) -> str:
    """Rows of the in-scope item table: code, name, COICOP 2018 class, both weight splits and
    the number of reference specifications (0 for an item the office has none for)."""
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    cov = pl.read_csv(results / "coverage.csv", schema_overrides={"label": pl.Utf8})
    t = scope.filter(pl.col("in_scope")).join(cov.select("label", "w_equal", "w_nnls"), on="label")
    n_ref: dict[str, int] = {}
    f = results / "gold_label_recall.csv"
    if f.exists():
        rec = pl.read_csv(f, schema_overrides={"label": pl.Utf8})
        first = rec.filter(pl.col("method") == rec["method"][0])
        n_ref = dict(zip(first["label"], first["n"], strict=True))
    return "\n".join(
        f"{r['label']} & {_tex(r['name_en'])} & {r['coicop2018'].replace(';', ', ')} & "
        f"{r['w_equal']:.2f} & {r['w_nnls']:.2f} & {num(n_ref.get(r['label'], 0))} \\\\"
        for r in t.iter_rows(named=True)
    )


def costs_table(results: Path = RESULTS) -> str:
    """Rows of the LLM cost ledger: every billed call, by model and purpose."""
    c = pl.read_csv(results / "llm_costs.csv")
    g = (
        c.with_columns(pl.col("purpose").str.split(":").list.first().alias("stage"))
        .group_by("model", "stage")
        .agg(
            pl.len().alias("calls"),
            (pl.col("input") + pl.col("cache_write") + pl.col("cache_read")).sum().alias("inp"),
            pl.col("output").sum().alias("out"),
            pl.col("usd").sum(),
        )
        .sort("model", "stage")
    )
    rows = [
        f"{_tex(r['model'])} & {r['stage']} & {num(r['calls'])} & {num(r['inp'])} & "
        f"{num(r['out'])} & {r['usd']:.2f} \\\\"
        for r in g.iter_rows(named=True)
    ]
    rows.append(
        f"\\midrule Total & & {num(c.height)} & "
        f"{num(int((c['input'] + c['cache_write'] + c['cache_read']).sum()))} & "
        f"{num(int(c['output'].sum()))} & {c['usd'].sum():.2f} \\\\"
    )
    return "\n".join(rows)


def prompt_text() -> str:
    """The instruction part of the LLM system prompt (the label list and examples follow)."""
    from cpilens.classify import llm

    none = pl.DataFrame(schema={"text": pl.Utf8, "label": pl.Utf8})
    head = llm.system_prompt(none)
    return head.split("Labels by group")[0].rstrip() + "\n"


def write_tables(results: Path = RESULTS, out: Path = GENERATED) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, f in (("items.tex", items_table), ("costs.tex", costs_table)):
        try:
            text = f(results)
        except FileNotFoundError:
            continue
        (out / name).write_text(f"% Generated by `cpilens report`. Do not edit.\n{text}\n")
        made.append(name)
    (out / "prompt.txt").write_text(prompt_text())
    return [*made, "prompt.txt"]


# ---------------------------------------------------------------- README

README = ROOT / "README.md"
README_TEMPLATE = ROOT / "docs" / "readme_results.md"
START, END = "<!-- results:start (generated by `cpilens report`) -->", "<!-- results:end -->"


def _plain(v: str) -> str:
    """A macro value as Markdown text: ``1{,}000`` -> ``1,000``, ``\\textminus{}`` -> ``−``."""
    return v.replace("{,}", ",").replace("\\textminus{}", "−")


def render_readme(results: Path = RESULTS, template: Path | None = None) -> str:
    """The README results block: ``{Macro}`` placeholders in the template filled from
    ``macros``. An unknown placeholder raises, as an undefined macro fails the paper build."""
    m = macros(results)

    def fill(match: re.Match[str]) -> str:
        return _plain(m[match.group(1)])

    text = (template or README_TEMPLATE).read_text()
    return re.sub(r"\{([A-Za-z]+)\}", fill, text).strip()


def write_readme(results: Path = RESULTS, readme: Path = README) -> bool:
    """Replace the block between the markers in README.md; False if the markers are missing."""
    text = readme.read_text()
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    tail = rest.split(END, 1)[1]
    readme.write_text(f"{head}{START}\n{render_readme(results)}\n{END}{tail}")
    return True
