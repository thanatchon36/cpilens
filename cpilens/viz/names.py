"""Display names for methods, CPI labels and groups (figures and tables)."""

from __future__ import annotations

import polars as pl

from cpilens.paths import TAXONOMY

EMB = {"e5-large": "e5-large", "bge-m3": "bge-m3", "xlmr": "XLM-R"}
CLF = {"knn": "k-NN", "linear": "logistic", "mlp": "MLP", "finetune": "fine-tuned"}
LLM = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-opus-5-5": "Claude Opus 5.5"}


def method(name: str) -> str:
    if name in LLM:
        return LLM[name]
    if name.startswith("cascade__"):
        _, emb, clf, _scenario, llm = name.split("__")
        return f"Cascade: {EMB[emb]} {CLF[clf]} → {LLM[llm].removeprefix('Claude ')}"
    emb, clf, scenario = name.split("__")
    base = f"{EMB[emb]} {CLF[clf]}"
    for v, txt in (
        ("-nobrand-noqty", "no brand, no qty"),
        ("-nobrand", "no brand"),
        ("-noqty", "no qty"),
    ):
        if scenario.endswith(v):
            return f"{base} ({txt})"
    return base


def labels() -> dict[str, str]:
    """CPI label -> English name (``taxonomy/thcpi_scope.csv``), shortened at ';' / '('."""
    s = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    short = [n.split(" (")[0].split(";")[0] for n in s["name_en"]]
    return {**dict(zip(s["label"], short, strict=True)), "OOS": "Out of scope"}


def groups() -> dict[str, str]:
    """CPI group (first three digits) -> English name, from the published weights table."""
    w = pl.read_csv(TAXONOMY / "thcpi_weights_2566.csv", schema_overrides={"code": pl.Utf8})
    out = {
        c[:3]: n.split(" (")[0]
        for c, n in zip(w["code"], w["name_en"], strict=True)
        if c.endswith("00")
    }
    # groups the weights table does not list (only their divisions)
    extra = {"421": "Personal care products", "721": "Alcoholic beverages"}
    return {**extra, **out, "OOS": "Out of scope", "ABSTAIN": "Abstain"}
