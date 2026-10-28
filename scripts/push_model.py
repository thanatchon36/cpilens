"""Stage the fine-tuned XLM-R classifier for the Hugging Face Hub, and optionally upload it.

``make models`` saves each fine-tuned run under ``.cache/ft/xlmr__<scenario>/`` with generic
``LABEL_0`` names. This writes the real CPI item codes into ``config.json``, adds the label
table, a model card whose every number comes from ``paper/numbers.tex``, and the licence
notices the training data require, then uploads the folder if a token is available.

Usage::

    python scripts/push_model.py --repo <user>/<name>                 # stage only, print the plan
    python scripts/push_model.py --repo <user>/<name> --push          # stage and upload
    python scripts/push_model.py --repo <user>/<name> --scenario nonthai --push

The token comes from ``HF_TOKEN`` in the environment or in ``.env`` (gitignored); nothing is
uploaded without ``--push``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import polars as pl

from cpilens.classify import experiments as ex
from cpilens.classify.config import FT_HF
from cpilens.paths import CACHE, ROOT, TAXONOMY

STAGE = CACHE / "hf"
REPO_URL = "https://github.com/thanatchon36/cpilens"
PAPER_URL = "https://thanatchon36.github.io/cpilens/paper/"
DOI = "10.5281/zenodo.23686886"
# the paper's name for each training pool, and the macro suffix of its scores
SCENARIOS = {
    "nonthai+thaiweak+mocall": ("all official labels", "XlmrFT"),
    "nonthai": ("no Thai label (zero-shot cross-lingual transfer)", "XlmrFTZeroShot"),
}


def macros() -> dict[str, str]:
    text = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2).replace("{,}", ",").replace("\\%", "%")
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", text, re.M)
    }


def token() -> str | None:
    if tok := os.environ.get("HF_TOKEN"):
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def card(scenario: str, repo: str, licence: str) -> str:
    v, (pool, tag) = macros(), SCENARIOS[scenario]
    bp = "year-on-year inflation error (RMSE, basis points)"
    rows = "\n".join(
        f"| {agg.capitalize()} {bp} | {v[f'BpYoyRmse{agg.capitalize()}XlmrFT']} "
        f"| {v[f'BpYoyRmse{agg.capitalize()}Opus']} ({v['NameOpus']}) |"
        for agg in ("headline", "food")
    )
    return f"""---
language:
- th
- en
license: {licence}
base_model: {FT_HF}
datasets:
- openfoodfacts/product-database
pipeline_tag: text-classification
library_name: transformers
tags:
- economics
- official-statistics
- consumer-price-index
- thai
widget:
- text: "นมสดรสจืด | โฟร์โมสต์ | 200 มล."
- text: "ข้าวหอมมะลิ 100% | ตราฉัตร | 5 กก."
---

# cpilens: Thai CPI item classifier (XLM-R, fine-tuned)

Assigns a product text to one of the {v["NLabelsInScope"]} in-scope items of the **Thai
consumer price index** (base 2566 / 2023) or to `OOS` (out of scope). The input is the
product's name, brand and quantity joined as `name | brand | quantity`; names are usually in
Thai, sometimes in English.

This is the fine-tuned encoder of the paper
[*How much does product classification cost in basis points?*]({PAPER_URL}), trained on
{pool}. The paper's purpose is to measure what a classifier's errors do to **measured
inflation**, so the numbers below include that, not accuracy alone.

## Scores, and why you should read the second column

On {v["NRefSpecs"]} held-out official price-collection specifications published by Thailand's
Ministry of Commerce ({v["NRefInScope"]} of them in scope), scored with CPI expenditure
weights:

| Metric | This model | For comparison |
|---|---|---|
| Expenditure-weighted accuracy | {v[f"AccW{tag}"]}% | {v["AccWOpus"]}% ({v["NameOpus"]} prompted) |
| Accuracy over the whole held-out list | {v[f"Acc{tag}"]}% | — |
| Macro-F1 | {v[f"MacroF{tag}"]}% | — |
{rows}

**A frontier LLM is far better at this task.** The paper's finding is that prompting
{v["NameOpus"]} reaches {v["AccWOpus"]}% of the evaluated CPI weight against
{v["AccWXlmrFT"]}% here, and that this model would understate Thai food inflation in every
simulated month. This encoder is published for reproducibility and as a cheap local baseline
(about USD {v["CostXlmrFT"]} per million products against USD {v["CostOpus"]} for the LLM),
not as the recommended classifier.

It also misses whole items: every held-out specification of fresh meat, freshwater fish,
processed fish and tobacco is misclassified (each a small item, with
{v["NZeroRecallMinXlmrFT"]} to {v["NZeroRecallMaxXlmrFT"]} held-out specifications), and most
personal-care specifications go to `OOS`.

## Use

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{repo}")
clf("นมสดรสจืด | โฟร์โมสต์ | 200 มล.")  # -> [{{'label': '11320', 'score': ...}}]
```

Labels are **TPSO item codes** (`11320` is milk and dairy products) plus `OOS`. `labels.csv`
in this repository gives each code's English name and its COICOP 2018 class. The scores from
`pipeline` are raw softmax; the paper calibrates them with one temperature fitted on a
validation split, which `cpilens` applies ({REPO_URL}).

## Training

Fine-tuned from [`{FT_HF}`](https://huggingface.co/{FT_HF}) with a classification head on
{pool}: at most {v["CfgFtEpochs"]} epochs, batch {v["CfgFtBatch"]}, AdamW at {v["CfgFtLR"]}
with linear warm-up and decay, inputs truncated to {v["CfgFtMaxLen"]} tokens, class weights
proportional to the inverse square root of class frequency, early stopping on validation loss.
The official-label pool holds {v["NMocTrainval"]} specifications; whole sub-items are held out,
so no test specification's sub-item is seen in training. One training run, one seed
({v["CfgSeed"]}).

## Intended use, and what this is not for

For research on the measurement error that product classification adds to price statistics,
and as a baseline for Thai or other non-English product text. **Not** for producing official
statistics without validation on your own data, and not for any decision about a person or a
firm. The reference text is the statistical office's own short specifications; shop and
database listings are longer and noisier, and accuracy on them will be lower. The
specifications are public and may have been seen in pre-training by other models the paper
compares against.

## Training data, licences and attribution

- Product text and weak labels: **Open Food Facts** and Open Beauty Facts, © Open Food Facts
  contributors, under the **Open Database License (ODbL) 1.0**. A model trained on an ODbL
  database is a Produced Work: this notice is the attribution ODbL asks for. If you extract a
  derived *database* from this work, ODbL's share-alike terms apply to it.
- Official labels: price-collection specifications published as open data by the **Ministry of
  Commerce**, Thailand, compiled by the Trade Policy and Strategy Office (TPSO); CPI indices
  and weights from TPSO.
- Base model `{FT_HF}`: MIT.

Every source, with its URL, retrieval date, licence and checksum, is listed in
[`data/SOURCES.md`]({REPO_URL}/blob/main/data/SOURCES.md).

**No training corpus is published here.** The Thai government source pages state no licence, so
this project does not redistribute their files; the repository publishes the held-out labels as
item codes without the official text (`data/reference_set.csv` in the repository), and
`make data` rebuilds the full corpus from the sources with checksums.

## Citation

```bibtex
@misc{{chongmankhong2026cpilens,
  author = {{Chongmankhong, Thanatchon}},
  title  = {{How much does product classification cost in basis points?
            Multilingual classification of Thai product text for consumer price measurement}},
  year   = {{2026}},
  doi    = {{{DOI}}},
  url    = {{{PAPER_URL}}}
}}
```

Code, data and the paper: {REPO_URL}. Every number above is generated by code from
`results/`; nothing is typed by hand.

*The views expressed are the author's own and do not necessarily reflect those of the Bank of
Thailand. This work uses only publicly available data. This is independent work, not a BIS,
ECB or Bank of Thailand publication, and its numbers are not directly comparable with the BIS
Innovation Hub's Project Spectrum.*
"""


def stage(scenario: str, repo: str, licence: str) -> Path:
    src = CACHE / "ft" / f"xlmr__{scenario}"
    if not src.exists():
        raise SystemExit(f"no checkpoint at {src}; run `make models` first")
    out = STAGE / scenario
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(src / name, out / name)

    cls = ex.classes()
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if len(cfg["id2label"]) != len(cls):
        raise SystemExit(f"checkpoint has {len(cfg['id2label'])} labels, taxonomy has {len(cls)}")
    cfg["id2label"] = {str(i): c for i, c in enumerate(cls)}
    cfg["label2id"] = {c: i for i, c in enumerate(cls)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    keep = [c for c in ("label", "name_en", "coicop2018", "group") if c in scope.columns]
    scope.select(keep).write_csv(out / "labels.csv")
    (out / "README.md").write_text(card(scenario, repo, licence), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Hub repo id, e.g. user/cpilens-xlmr-thcpi")
    p.add_argument("--scenario", default="nonthai+thaiweak+mocall", choices=list(SCENARIOS))
    p.add_argument("--license", default="mit", dest="licence")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--push", action="store_true", help="upload; without it, only stage")
    a = p.parse_args()

    out = stage(a.scenario, a.repo, a.licence)
    files = sorted(f.name for f in out.iterdir())
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"staged {out.relative_to(ROOT)} ({size:.0f} MB): {', '.join(files)}")
    if not a.push:
        print(f"review {out.relative_to(ROOT) / 'README.md'}, then re-run with --push")
        return 0

    tok = token()
    if tok is None:
        raise SystemExit("no HF_TOKEN in the environment or .env; add one with write access")
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(
        repo_id=a.repo, folder_path=str(out), commit_message="cpilens fine-tuned XLM-R"
    )
    print(f"pushed https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
