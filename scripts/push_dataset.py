"""Stage the cpilens reference set and taxonomy tables for the Hugging Face Hub, and upload them.

The CSV files are copied byte for byte from a clean commit, under the paths they have in the code
repository. Each also gets a typed Parquet file, which the Hub configs load: CSV type inference
would turn the code ``00000`` into ``0``. The official specification text is not among them: the
Thai government sources state no licence, so only ids, item codes and derived numbers are
published (data/SOURCES.md).

Usage::

    python scripts/push_dataset.py --repo <user>/<name>           # stage only, print the plan
    python scripts/push_dataset.py --repo <user>/<name> --push    # stage and upload

The token comes from ``HF_TOKEN`` in the environment or ``.env`` (gitignored), else from
``hf auth login``; nothing is uploaded without ``--push``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import polars as pl
import yaml
from push_model import DOI, PAPER_URL, REPO_URL, macros, token

from cpilens.paths import CACHE, ROOT

STAGE = CACHE / "hf" / "dataset"
MODEL = "thanatchon36/cpilens-xlmr-thcpi"
ODBL = "ODbL 1.0"
MIT = "MIT"
UNSTATED = "Attribution, non-commercial research use (no upstream licence stated)"
TPSO_MIT = "MIT for the derivation; the official figures are TPSO's (no upstream licence stated)"


class Table(NamedTuple):
    config: str
    path: str
    split: str
    licence: str
    about: str
    columns: dict[str, tuple[str, str]]  # name -> (dtype, meaning), in file order


TABLES = (
    Table(
        "reference_set",
        "data/reference_set.csv",
        "test",
        UNSTATED,
        "The held-out evaluation rows of the paper: ids, labels and the sampling design.",
        {
            "id": ("string", "`moc:` plus the official 16-digit specification code"),
            "label": ("string", "TPSO item code of the in-scope item, or `OOS`"),
            "incl_prob": ("float64", "Inclusion probability; weight a row by its inverse"),
            "stratum": ("string", "Bootstrap stratum: the label"),
            "cluster": ("string", "Bootstrap cluster: the official 7-digit sub-item"),
            "unsure": ("bool", "Always false here; kept for the hand-labelling format"),
        },
    ),
    Table(
        "gold_sample",
        "data/gold/sample.csv",
        "train",
        ODBL,
        "Thai-market Open Food Facts products drawn for hand labelling. **It has no labels and "
        "was not used in this version.**",
        {
            "order": ("int32", "Labelling order"),
            "barcode": ("string", "Product barcode"),
            "source": ("string", "`off` (Open Food Facts) or `obf` (Open Beauty Facts)"),
            "stratum": ("string", "CPI weight cell of the weak label, `none` or `OOS`"),
            "incl_prob": ("float64", "Inclusion probability within the stratum"),
            "double": ("bool", "Selected for a blind second labelling pass"),
            "name_th": ("string", "Product name in Thai"),
            "name_en": ("string", "Product name in English, if any"),
            "brands": ("string", "Brands"),
            "quantity": ("string", "Quantity as entered"),
            "image_url": ("string", "Link to the front image (CC BY-SA; not copied here)"),
        },
    ),
    Table(
        "item_scope",
        "taxonomy/thcpi_scope.csv",
        "train",
        MIT,
        "The classifier's label set: the in-scope CPI items and their COICOP 2018 class.",
        {
            "label": ("string", "TPSO item code"),
            "name_th": ("string", "Item name in Thai"),
            "name_en": ("string", "Item name in English"),
            "in_scope": ("bool", "Whether the item is in scope"),
            "scope_group": ("string", "Group of the item within the scope"),
            "coicop2018": ("string", "COICOP 2018 class"),
            "coicop_pages": ("string", "Pages of the UN classification that give the class"),
            "note": ("string", "Note on the mapping, if any"),
            "includes": ("string", "Short description of what the item covers"),
        },
    ),
    Table(
        "category_crosswalk",
        "taxonomy/off_categories_to_thcpi.csv",
        "train",
        ODBL,
        "Crosswalk from Open Food Facts category tags to CPI items, which gives the weak labels.",
        {
            "tag": ("string", "Open Food Facts category tag"),
            "label": ("string", "TPSO item code, or `OOS`"),
            "priority": ("int32", "Priority when a product has several matching tags"),
            "basis": ("string", "Reason for the mapping"),
        },
    ),
    Table(
        "aggregate_weights",
        "taxonomy/thcpi_weights_2566.csv",
        "train",
        TPSO_MIT,
        "Official aggregate weights recovered from TPSO's monthly releases, base 2566 (2023).",
        {
            "code": ("string", "TPSO aggregate code"),
            "name_en": ("string", "Aggregate name in English"),
            "kind": ("string", "Level of the aggregate"),
            "w0": ("float64", "Base weight, the mean across the releases"),
            "w0_spread": ("float64", "Largest minus smallest `w0` across the releases"),
            "w_pr": ("float64", "Price-updated weight share, as used in the paper"),
            "release_months_be": ("string", "Releases used, in Buddhist Era years"),
            "price_ref_be": ("string", "Price reference month, in Buddhist Era years"),
        },
    ),
    Table(
        "item_weights",
        "taxonomy/thcpi_leaf_weights_2566.csv",
        "train",
        TPSO_MIT,
        "Item weights. **Item weights inside an official cell are estimates, not official "
        "figures.**",
        {
            "leaf": ("string", "TPSO item code"),
            "name_th": ("string", "Item name in Thai"),
            "cell": ("string", "Official node whose weight the item shares"),
            "residual_cell": ("bool", "Whether official nodes are nested below that node"),
            "cell_weight": ("float64", "Weight of the cell"),
            "w_equal": ("float64", "Estimated item weight: equal split of the cell"),
            "w_nnls": ("float64", "Estimated item weight: non-negative least squares"),
        },
    ),
    Table(
        "aggregate_codes",
        "taxonomy/thcpi_weight_aggregates.csv",
        "train",
        TPSO_MIT,
        "The aggregates in TPSO's weight table, in the order of the table.",
        {
            "row": ("int32", "Row in the published table"),
            "code": ("string", "TPSO aggregate code"),
            "name_en": ("string", "Aggregate name in English"),
            "kind": ("string", "Level of the aggregate"),
        },
    ),
    Table(
        "source_manifest",
        "data/manifest.csv",
        "train",
        MIT,
        "URL and SHA-256 of every raw source file, which `make data` checks on download.",
        {
            "path": ("string", "Path under `data/raw/`"),
            "url": ("string", "Source URL"),
            "sha256": ("string", "SHA-256 of the file"),
        },
    ),
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def commit() -> str:
    """The commit the files are copied from; refuses files that differ from it."""
    paths = [t.path for t in TABLES]
    changed = git("-c", "core.filemode=false", "status", "--porcelain", "--", *paths)
    if changed:
        raise SystemExit(f"commit these data files first:\n{changed}")
    return git("rev-parse", "HEAD")


def check_columns(t: Table) -> int:
    """Every column must be declared, so a new column (official text, say) cannot slip out."""
    with (ROOT / t.path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if header != list(t.columns):
        raise SystemExit(f"{t.path}: columns {header} are not the declared {list(t.columns)}")
    return rows


def front_matter(sizes: dict[str, int]) -> str:
    meta = {
        "license": "other",
        "license_name": "per-file",
        "license_link": f"{REPO_URL}/blob/main/data/SOURCES.md",
        "language": ["th", "en"],
        "task_categories": ["text-classification"],
        "tags": ["economics", "official-statistics", "consumer-price-index", "thai"],
        "pretty_name": "cpilens reference set and Thai CPI taxonomy",
        "size_categories": ["1K<n<10K"] if max(sizes.values()) >= 1000 else ["n<1K"],
        "configs": [
            {
                "config_name": t.config,
                "data_files": [{"split": t.split, "path": f"parquet/{t.config}.parquet"}],
                **({"default": True} if t.config == "reference_set" else {}),
            }
            for t in TABLES
        ],
    }
    return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)


def card(repo: str, revision: str, sizes: dict[str, int]) -> str:
    v = macros()
    ref = pl.read_csv(ROOT / "data" / "reference_set.csv", infer_schema_length=0).filter(
        pl.col("label") != "OOS"
    )
    n_in, n_items = ref.height, ref["label"].n_unique()
    n_out = sizes["reference_set"] - n_in
    n_scope = int(v["NLabelsInScope"])
    if f"{sizes['reference_set']:,}" != v["NRefSpecs"]:
        raise SystemExit(f"reference set has {sizes['reference_set']}, paper says {v['NRefSpecs']}")

    tables = "\n".join(
        f"| `{t.config}` | `{t.path}` | {sizes[t.config]:,} | {t.about} | {t.licence} |"
        for t in TABLES
    )
    fields = "\n\n".join(
        f"### `{t.config}`\n\n| Column | Type | Meaning |\n|---|---|---|\n"
        + "\n".join(f"| `{c}` | {d} | {m} |" for c, (d, m) in t.columns.items())
        for t in TABLES
    )
    return f"""---
{front_matter(sizes)}---

# cpilens: reference set and Thai CPI taxonomy

The evaluation labels and taxonomy tables of the paper
[*How much does product classification cost in basis points?*]({PAPER_URL}). The paper classifies
Thai product text into the items of the Thai consumer price index (base 2566 / 2023), then
measures what the classifier's errors do to measured inflation.

📄 [Paper]({PAPER_URL}) · 💻 [Code]({REPO_URL}) ·
🤗 [Model](https://huggingface.co/{MODEL}) · 🔖 DOI [{DOI}](https://doi.org/{DOI})

**The official specification text is not in this dataset.** The reference set holds ids, item
codes and the sampling design. The Thai government source pages state no licence, so the project
does not redistribute their files. See [Getting the text](#getting-the-text).

**No training corpus is published here.** `make data` in the code repository rebuilds it from the
public sources, with checksums.

## Contents

Each CSV file is copied unchanged from commit
[`{revision[:7]}`]({REPO_URL}/tree/{revision}) of the code repository, at the same path. The
configs load `parquet/<config>.parquet`, which holds the same rows with the column types listed
under [Fields](#fields), so codes such as `00000` keep their leading zeros.

| Config | File | Rows | Content | Licence |
|---|---|---|---|---|
{tables}

## Use

```python
from datasets import load_dataset

ref = load_dataset("{repo}", split="test")  # the reference set, the default config
scope = load_dataset("{repo}", "item_scope", split="train")
```

## The reference set

- **Source:** the price-collection specifications the Ministry of Commerce publishes as open
  data, each under the official item code TPSO uses.
- **Rows:** {n_in:,} held-out specifications of {n_items} in-scope items, all kept, and {n_out:,}
  held-out out-of-scope specifications in a fixed hash-ordered sample.
- **Weighting:** weight each row by `1 / incl_prob`, so estimates refer to the whole held-out
  list.
- **Split:** whole sub-items are held out inside every item. A fifth of an item's sub-items, at
  least one, are test.
- **Coverage:** {n_scope - n_items} of the {n_scope} in-scope items have no specifications in the
  published file and cannot be evaluated. Items with few specifications have wide intervals.
- **Not sealed:** cascade thresholds and classifier settings were chosen on the training and
  validation splits only, but the development grid logged reference scores for every run. They
  were visible during development.

For intervals, resample whole `cluster` values within each `stratum`. The errors of a method
cluster within sub-items.

## Getting the text

Scoring a saved prediction needs only this dataset. To run a classifier on the specifications,
build them from the source:

```bash
git clone {REPO_URL} && cd cpilens
uv sync && make data
```

`data/interim/moc_specs.parquet` then holds the text. A row's `id` here is `moc:` plus that file's
`spec_id`. If an upstream file has changed since this release, the build stops at the checksum
check.

## Limitations

- Specifications are the statistical office's own short descriptions, not shop or web listings.
- Open Food Facts is crowd-sourced, and its Thai products served as a development set only.
- There are no prices. The data support classification, not index compilation.
- This is not an official classification. Do not use it as, or to infer, any statistical office's
  or central bank's internal data or methods.

## Licences and attribution

The licence is set per file, as in the table above. The paper's
[data sources]({REPO_URL}/blob/main/data/SOURCES.md) give every source's URL, retrieval date,
licence and checksum.

- **Open Food Facts and Open Beauty Facts:** © Open Food Facts contributors,
  <https://world.openfoodfacts.org>, under the
  [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). A database derived
  from these files must carry the same licence.
- **Thai government sources:** Ministry of Commerce open data, and Trade Policy and Strategy Office
  (TPSO) publications. Their pages state no licence, so this project treats them under
  attribution and non-commercial research use.
- **COICOP 2018:** © United Nations. Class codes and titles are cited.

## Fields

{fields}

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

*The views expressed are the author's own and do not necessarily reflect those of the Bank of
Thailand. This work uses only publicly available data. This is independent work, not a BIS, ECB or
Bank of Thailand publication.*
"""


POLARS = {"string": pl.Utf8, "float64": pl.Float64, "int32": pl.Int32, "bool": pl.Boolean}


def to_parquet(t: Table, dest: Path) -> None:
    """Write the CSV with its declared types, and check that no value changed on the way."""
    text = pl.read_csv(ROOT / t.path, infer_schema_length=0)
    typed = pl.read_csv(ROOT / t.path, schema={c: POLARS[d] for c, (d, _) in t.columns.items()})
    for c, (d, _) in t.columns.items():
        if d == "string" and not typed[c].equals(text[c], null_equal=True):
            raise SystemExit(f"{t.path}: column {c} changed when typed")
        if typed[c].null_count() != text[c].null_count():
            raise SystemExit(f"{t.path}: column {c} lost values when typed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    typed.write_parquet(dest)


def stage(repo: str) -> Path:
    revision = commit()
    sizes = {t.config: check_columns(t) for t in TABLES}
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for t in TABLES:
        dest = STAGE / t.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / t.path, dest)
        to_parquet(t, STAGE / "parquet" / f"{t.config}.parquet")
    (STAGE / "README.md").write_text(card(repo, revision, sizes), encoding="utf-8")
    return STAGE


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Hub repo id, e.g. user/cpilens-reference-set")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--push", action="store_true", help="upload; without it, only stage")
    a = p.parse_args()

    out = stage(a.repo)
    files = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())
    print(f"staged {out.relative_to(ROOT)}: {', '.join(files)}")
    if not a.push:
        print(f"review {out.relative_to(ROOT) / 'README.md'}, then re-run with --push")
        return 0

    tok = token()
    if tok is None:
        raise SystemExit("no HF_TOKEN in the environment or .env; add one with write access")
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(
        repo_id=a.repo,
        repo_type="dataset",
        folder_path=str(out),
        commit_message="cpilens reference set and taxonomy tables",
    )
    print(f"pushed https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
