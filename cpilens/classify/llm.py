"""LLM classification through the Anthropic Messages API, with a disk cache and a cost log.

One product per request. The system prompt (instructions, the label list grouped by CPI
division, and few-shot examples drawn from the MOC *training* split only) is identical for
every request, so it is prompt-cached. The answer is constrained by a JSON schema whose
``group`` and ``label`` fields are enums: the model names the group first, then the label
(hierarchical prompting within one call), plus a verbalised confidence 0-100 used for the
coverage frontier.

Every response is cached in ``.cache/llm/<sha256 of the request>.json``; a cached request
costs nothing and is not logged. Every billed call appends one row to
``results/llm_costs.csv``. Before each billed call the logged total plus a worst-case
estimate for that call is checked against ``CAP_USD``; :class:`BudgetExceeded` stops the run.

No server-side refusal fallback is enabled: a fallback would let a different model answer
and blur the per-model cost-accuracy comparison. A refusal is recorded as an abstention.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cpilens.paths import CACHE, RESULTS, ROOT, TAXONOMY

# USD per million tokens, platform.claude.com/docs/en/about-claude/pricing (2026-10-27):
# input, 5-minute cache write, cache read, output. Haiku 5.5 rows are for prompts <= 100k.
PRICES = {
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 2.50, 0.10, 10.00),
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
}
CAP_USD = 60.0
MAX_TOKENS = 2048
LLM_CACHE = CACHE / "llm"
COSTS = RESULTS / "llm_costs.csv"
COST_FIELDS = [
    "utc",
    "model",
    "purpose",
    "key",
    "input",
    "cache_write",
    "cache_read",
    "output",
    "usd",
]
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def usd(model: str, inp: int, write: int, read: int, out: int) -> float:
    p = PRICES[model]
    return (inp * p[0] + write * p[1] + read * p[2] + out * p[3]) / 1e6


def spent(costs: Path = COSTS) -> float:
    if not costs.exists():
        return 0.0
    return float(pl.read_csv(costs)["usd"].sum())


def _env() -> dict[str, str]:
    env = {}
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.startswith("#"):
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("ANTHROPIC_")})
    return env


def client() -> Any:
    import anthropic

    env = _env()
    return anthropic.Anthropic(
        api_key=env["ANTHROPIC_API_KEY"],
        default_headers={"anthropic-workspace-id": env["ANTHROPIC_WORKSPACE_ID"]},
        max_retries=5,
    )


# ---------------------------------------------------------------- prompt


def labels() -> pl.DataFrame:
    return pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})


def few_shot(train: pl.DataFrame, per_label: int = 2) -> pl.DataFrame:
    """``per_label`` official MOC training examples per label, chosen by a fixed hash."""
    from cpilens.classify.data import bucket

    m = train.filter((pl.col("source") == "moc") & (pl.col("split") == "train"))
    m = m.with_columns(
        pl.col("id").map_elements(lambda k: bucket(k, 10**9), return_dtype=pl.Int64).alias("_h")
    )
    return m.sort("_h").group_by("label", maintain_order=True).head(per_label).sort("label")


def system_prompt(examples: pl.DataFrame) -> str:
    lab = labels()
    lines = [
        "You classify products sold in Thai shops into the item categories of the Thai "
        "consumer price index (CPI, base year 2023 = BE 2566), compiled by the Trade Policy "
        "and Strategy Office (TPSO). Each product is given as `name | brand | quantity`; "
        "names are usually Thai, sometimes English or other languages.",
        "",
        "Pick exactly one label. First choose the group, then a label inside that group. "
        "Use OOS when the product belongs to none of the labels (for example pet food, "
        "medicines, kitchenware, stationery, electronics). Classify what the product is, "
        "not its flavour: a shrimp-flavoured cracker is a snack (11122), not seafood. "
        "Confidence is your probability, 0-100, that the label is correct.",
        "",
        "Labels by group (code: Thai name / English name: what it includes):",
    ]
    for (group,), rows in lab.group_by("scope_group", maintain_order=True):
        lines.append(f"\n[{group}]")
        for r in rows.iter_rows(named=True):
            lines.append(f"{r['label']}: {r['name_th'] or '-'} / {r['name_en']}: {r['includes']}")
    lines += ["", "Examples (TPSO price-collection specifications with their official labels):"]
    for t, y in examples.select("text", "label").rows():
        lines.append(f"{t} -> {y}")
    return "\n".join(lines)


def schema() -> dict[str, Any]:
    lab = labels()
    return {
        "type": "object",
        "properties": {
            "group": {
                "type": "string",
                "enum": lab["scope_group"].unique(maintain_order=True).to_list(),
            },
            "label": {"type": "string", "enum": lab["label"].to_list()},
            "confidence": {"type": "integer"},
        },
        "required": ["group", "label", "confidence"],
        "additionalProperties": False,
    }


def request(model: str, system: str, text: str, effort: str = "low") -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": text}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema()}},
    }


# ---------------------------------------------------------------- calls


def _key(req: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _parse(resp: dict[str, Any]) -> dict[str, Any]:
    if resp.get("stop_reason") != "end_turn":
        return {
            "label": None,
            "group": None,
            "confidence": None,
            "stop_reason": resp.get("stop_reason"),
        }
    text = "".join(b["text"] for b in resp["content"] if b["type"] == "text")
    out = json.loads(text)
    out["confidence"] = min(max(int(out["confidence"]), 0), 100)
    out["stop_reason"] = "end_turn"
    return out


def call(
    api: Any,
    req: dict[str, Any],
    purpose: str,
    cache: Path = LLM_CACHE,
    costs: Path = COSTS,
    cap: float = CAP_USD,
) -> dict[str, Any]:
    """Classify one request; cached responses are free and not logged."""
    key = _key(req)
    f = cache / f"{key}.json"
    if f.exists():
        return _parse(json.loads(f.read_text()))
    worst = usd(req["model"], 0, 4 * len(json.dumps(req, ensure_ascii=False)), 0, req["max_tokens"])
    with _lock:
        if spent(costs) + worst > cap:
            raise BudgetExceeded(f"logged spend {spent(costs):.2f} + {worst:.3f} > {cap}")
    resp = api.messages.create(**req).to_dict()
    u = resp["usage"]
    row = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": req["model"],
        "purpose": purpose,
        "key": key,
        "input": u["input_tokens"],
        "cache_write": u.get("cache_creation_input_tokens") or 0,
        "cache_read": u.get("cache_read_input_tokens") or 0,
        "output": u["output_tokens"],
    }
    row["usd"] = round(
        usd(req["model"], row["input"], row["cache_write"], row["cache_read"], row["output"]), 6
    )
    with _lock:
        cache.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(resp, ensure_ascii=False))
        new = not costs.exists()
        costs.parent.mkdir(parents=True, exist_ok=True)
        with costs.open("a", newline="") as fh:
            w = csv.DictWriter(fh, COST_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return _parse(resp)


def classify(
    texts: list[str],
    model: str,
    system: str,
    purpose: str,
    effort: str = "low",
    workers: int = 8,
    max_usd: float | None = None,
) -> pl.DataFrame:
    """Labels for ``texts`` in order. Stops early (rows left null) if this run alone would
    exceed ``max_usd`` or the project cap is reached."""
    api = client()
    start = spent()
    reqs = [request(model, system, t, effort) for t in texts]
    # warm the prompt cache with one call before fanning out
    out: list[dict[str, Any] | None] = [None] * len(reqs)
    if reqs:
        out[0] = call(api, reqs[0], purpose)

    def one(i: int) -> None:
        if max_usd is not None and spent() - start > max_usd:
            return
        try:
            out[i] = call(api, reqs[i], purpose)
        except BudgetExceeded:
            return

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(1, len(reqs))))
    empty = {"label": None, "group": None, "confidence": None, "stop_reason": "not_run"}
    return pl.DataFrame(
        [o or empty for o in out],
        schema={"label": pl.Utf8, "group": pl.Utf8, "confidence": pl.Int64, "stop_reason": pl.Utf8},
    )


def run_eval(model: str, effort: str = "low", max_usd: float | None = None) -> pl.DataFrame:
    """LLM predictions on the Thai weak validation split and the reference set (the rows of
    ``reference_table``, not the whole MOC test split: its out-of-scope part is sampled)."""
    from cpilens.classify import experiments as ex
    from cpilens.classify.data import corpus
    from cpilens.evaluate.gold import reference_table

    c = corpus()
    ev = ex.eval_sets(c)
    ref_ids = reference_table(c)["id"].to_list()
    ev = ev.filter((pl.col("eval_set") != "moc_test") | pl.col("id").is_in(ref_ids))
    sp = system_prompt(few_shot(c))
    t0 = datetime.now(UTC)
    r = classify(ev["text"].to_list(), model, sp, f"eval:{model}", effort, max_usd=max_usd)
    secs = (datetime.now(UTC) - t0).total_seconds()
    out = ev.select("id", "eval_set", "label").with_columns(
        r["label"].alias("pred"), (r["confidence"] / 100).alias("conf"), r["stop_reason"]
    )
    d = RESULTS / "preds" / "llm"
    d.mkdir(parents=True, exist_ok=True)
    out.write_parquet(d / f"{model}__{effort}.parquet")
    print(f"{model}: {out.height} items, {secs:.0f}s wall, spent so far {spent():.2f} USD")
    return out
