"""``results/run_manifest.json``: what produced the results (code, packages, model revisions).

Embedding and encoder revisions are the commits the Hugging Face cache resolved; LLM
revisions are the model names the API returned in the cached responses.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from cpilens.paths import RESULTS, ROOT

PACKAGES = (
    "anthropic", "duckdb", "matplotlib", "numpy", "polars", "pyarrow", "scikit-learn",
    "scipy", "sentence-transformers", "torch", "transformers",
)  # fmt: skip


def hf_revision(repo: str, hub: Path | None = None) -> str | None:
    hub = hub or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def llm_models_returned(cache: Path) -> dict[str, int]:
    """Model name returned by the API -> number of cached responses."""
    out: dict[str, int] = {}
    for f in cache.glob("*.json") if cache.exists() else []:
        m = json.loads(f.read_text()).get("model")
        if m:
            out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items()))


def _git() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip()


def build() -> dict[str, object]:
    from cpilens.classify import finetune
    from cpilens.classify.llm import LLM_CACHE
    from cpilens.embed.encode import MODELS
    from cpilens.evaluate import gold
    from cpilens.evaluate.run import LLMS

    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            continue
    hf = [m["hf"] for m in MODELS.values()] + [finetune.HF]
    return {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "hf_models": {r: hf_revision(r) for r in hf},
        "llm_models_requested": list(LLMS),
        "llm_models_returned": llm_models_returned(LLM_CACHE),
        "seed": gold.SEED,
        "data_manifest": "data/manifest.csv",
    }


def write(out: Path = RESULTS) -> Path:
    f = out / "run_manifest.json"
    f.write_text(json.dumps(build(), indent=2) + "\n")
    return f
