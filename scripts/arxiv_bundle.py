"""Build the arXiv source bundle, ``paper/arxiv.tar.gz``, from the paper as last built.

arXiv compiles the paper itself, from the root of the bundle, with XeLaTeX (declared in
``00README.json``). This stages ``main.tex`` with the figures moved inside the bundle, the
section files, ``numbers.tex``, the generated tables and prompt, ``refs.bib`` and a
precompiled ``main.bbl``; compiles the staged copy once with tectonic to prove it is
self-contained; then tars it. It also prints the title, the plain-text abstract (for the
arXiv form) and a comments line. Run ``make paper`` first.

Usage: ``python scripts/arxiv_bundle.py``.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = PAPER / "arxiv"
OUT = PAPER / "arxiv.tar.gz"
FIGURES = ROOT / "figures" / "out"
INTERMEDIATES = ("main.aux", "main.log", "main.out", "main.blg", "main.pdf")


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "\\graphicspath{{../figures/out/}}" not in main_tex:
        raise SystemExit("main.tex: expected \\graphicspath{{../figures/out/}}")
    (STAGE / "main.tex").write_text(
        main_tex.replace("\\graphicspath{{../figures/out/}}", "\\graphicspath{{figures/}}"),
        encoding="utf-8",
    )
    for name in ("numbers.tex", "refs.bib"):
        shutil.copyfile(PAPER / name, STAGE / name)
    for sub in ("sections", "generated"):
        shutil.copytree(PAPER / sub, STAGE / sub)
    sections = "".join(p.read_text(encoding="utf-8") for p in (PAPER / "sections").glob("*.tex"))
    figures = sorted(
        set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([\w-]+\.pdf)\}", sections))
    )
    for name in figures:
        shutil.copyfile(FIGURES / name, STAGE / "figures" / name)
    (STAGE / "00README.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "process": {"compiler": "xelatex"},
                "sources": [{"filename": "main.tex", "usage": "toplevel"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # the staged copy must compile on its own; this also writes main.bbl
    subprocess.run(
        ["tectonic", "-X", "compile", "--keep-intermediates", "--keep-logs", "main.tex"],
        cwd=STAGE,
        check=True,
        capture_output=True,
    )
    log = (STAGE / "main.log").read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on main\.(?:xdv|pdf) \((\d+) pages", log)
    if pages is None or not (STAGE / "main.bbl").exists():
        raise SystemExit("staged compile left no page count or no main.bbl")
    for name in INTERMEDIATES:
        (STAGE / name).unlink(missing_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(STAGE)))
    files = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} kB, {len(files)} files)")
    print("\n".join(f"  {f}" for f in files))

    # metadata for the submission form, from the web build (numbers already filled in)
    page = (ROOT / "docs" / "paper" / "index.html").read_text(encoding="utf-8")
    title = re.search(r"<h1>(.*?)</h1>", page, re.S)
    subtitle = re.search(r'<p class="subtitle">(.*?)</p>', page, re.S)
    abstract = re.search(r'<div class="abstract-body">(.*?)</div>', page, re.S)
    if title is None or subtitle is None or abstract is None:
        raise SystemExit("docs/paper/index.html lacks the title, subtitle or abstract")

    def text(fragment: str) -> str:
        plain = html.unescape(" ".join(re.sub(r"<[^>]+>", "", fragment).split()))
        return plain.translate(str.maketrans("\u2018\u2019\u201c\u201d\u2013", "''\"\"-"))

    print(f"\nTitle:\n{text(title.group(1))} {text(subtitle.group(1))}")
    body = text(abstract.group(1))
    print(f"\nAbstract ({len(body)} characters; arXiv allows 1,920):\n{body}")
    print(
        f"\nComments:\n{pages.group(1)} pages, {len(figures)} figures. Code, data and the "
        "policy brief: https://github.com/thanatchon36/cpilens "
        "(archived at https://doi.org/10.5281/zenodo.23686886). "
        "JEL C55, C81, E31."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
