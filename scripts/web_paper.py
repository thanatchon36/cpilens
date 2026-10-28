"""Build the web version of the paper, ``docs/paper/``, from ``paper/main.tex``.

Same text as the PDF: the section files, generated tables and ``numbers.tex`` macros are
inlined and pandoc renders them with ``paper/reader.html`` (masthead, abstract, a results strip,
contents, numbered sections, author-year citations). The results strip is filled from
``numbers.tex`` too, so the page types no number by hand. The PDFs and the PNG figures are copied
next to the page; GitHub Pages serves the folder at ``/paper/``. Requires pandoc on PATH.

Usage: ``python scripts/web_paper.py``.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "docs" / "paper"
FIGURES = ROOT / "figures" / "out"
REPO_URL = "https://github.com/thanatchon36/cpilens"
PAPER_URL = "https://thanatchon36.github.io/cpilens/paper/"
DOI = "10.5281/zenodo.23686886"
DISCLAIMER = (
    "The views expressed are the author's own and do not necessarily reflect those of the "
    "Bank of Thailand. This work uses only publicly available data."
)

# (research question, finding, detail), with \macro placeholders from numbers.tex
RESULTS = [
    (
        "RQ1",
        r"Transfer from non-Thai products alone: \AccWEfiveMLPZeroShot% of CPI weight correct",
        r"\NameEfive with an MLP; all \NMocTrainval official labels raise it to \AccWEfiveMLP%",
    ),
    (
        "RQ2",
        r"\NameOpus \AccWOpus%, \NameHaiku \AccWHaiku%",
        r"weighted by CPI expenditure on \NRefSpecs held-out official specifications",
    ),
    (
        "RQ2",
        r"Cascade \AccWCasEfiveMLPOpus% at \CostPctCasEfiveMLPOpus% of the LLM's cost "
        "on the held-out list",
        r"\NameEfive keeps the products it is sure of and sends the rest to \NameOpus",
    ),
    (
        "RQ3",
        r"Headline inflation off by \BpYoyRmseHeadlineOpus basis points with the LLM",
        r"year-on-year RMSE; at most \BpYoyRmseHeadlineOpusClusterHi if errors cluster; "
        r"\BpYoyRmseHeadlineBgeMLP to \BpYoyRmseHeadlineEfiveMLP basis points "
        "for cheaper classifiers",
    ),
]


def macros() -> dict[str, str]:
    text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2).replace("{,}", ",").replace("\\%", "%")
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", text, re.M)
    }


def expand(text: str, values: dict[str, str]) -> str:
    return re.sub(r"\\([A-Za-z]+)", lambda m: values[m.group(1)], text)  # a missing macro fails


def flatten(tex: str) -> str:
    """Inline sections, generated rows and numbers.tex macros; turn LaTeX-only constructs into
    ones pandoc reads (pandoc's own macro expansion drops digits inside table cells)."""
    tex = re.sub(
        r"\\input\{(sections/[\w-]+\.tex)\}",
        lambda m: (PAPER / m.group(1)).read_text(encoding="utf-8"),
        tex,
    )
    tex = re.sub(
        r"\\inputrows\{([\w/.-]+)\}",
        lambda m: (PAPER / m.group(1)).read_text(encoding="utf-8"),
        tex,
    )
    tex = re.sub(
        r"\\VerbatimInput(?:\[[^\]]*\])?\{([\w/.-]+)\}",
        lambda m: (
            "\\begin{verbatim}\n"
            + (PAPER / m.group(1)).read_text(encoding="utf-8").rstrip()
            + "\n\\end{verbatim}"
        ),
        tex,
    )
    tex = re.sub(
        r"\\includegraphics(\[[^\]]*\])?\{([\w-]+)\.pdf\}",
        r"\\includegraphics{figures/\2.png}",
        tex,
    )
    raw = dict(
        re.findall(
            r"^\\newcommand\{\\(\w+)\}\{(.*)\}$",
            (PAPER / "numbers.tex").read_text(encoding="utf-8"),
            re.M,
        )
    )
    tex = re.sub(r"\\([A-Za-z]+)(?:\{\})?", lambda m: raw.get(m.group(1), m.group(0)), tex)
    # pandoc drops \textminus and keeps {,} as a span; enumitem options and \path are unknown to it
    tex = (
        tex.replace("\\textminus{}", "\u2212").replace("\\textminus", "\u2212").replace("{,}", ",")
    )
    tex = re.sub(r"\\begin\{description\}\[[^\]]*\]", r"\\begin{description}", tex)
    tex = re.sub(
        r"\\path\{([^}]*)\}", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_") + "}", tex
    )
    # pandoc drops the first token of a cell whose column has a >{...} prefix
    tex = tex.replace(">{\\raggedright\\arraybackslash}", "")
    # \paragraph would become a numbered heading; in the PDF it is a run-in bold lead
    return re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1}", tex)


def pandoc(tex: str, args: list[str]) -> str:
    tmp = PAPER / ".web.tex"
    tmp.write_text(tex, encoding="utf-8")
    try:
        return subprocess.run(
            ["pandoc", tmp.name, "--from=latex", "--to=html5", "--mathml", "--wrap=none", *args],
            cwd=PAPER,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        tmp.unlink(missing_ok=True)


def number_like_pdf(page: str) -> str:
    """Give the HTML the PDF's numbering: appendix letters, equation numbers, caption labels."""
    refs_at = page.index('<div id="refs"')
    appendix = sorted({int(n) for n in re.findall(r'<h1 data-number="(\d+)"', page[refs_at:])})
    letter = {n: chr(ord("A") + i) for i, n in enumerate(appendix)}
    in_appendix = set(re.findall(r'<h[1-3] data-number="[\d.]+" id="([^"]+)"', page[refs_at:]))

    def renumber(m: re.Match[str]) -> str:
        head, rest = int(m.group(2)), m.group(3) or ""
        return f"{m.group(1)}{letter.get(head, head)}{rest}"

    page = re.sub(
        r'((?:header|toc)-section-number">|data-number=")(\d+)((?:\.\d+)*)', renumber, page
    )
    page = re.sub(
        r'(data-reference-type="ref" data-reference="([^"]+)">)(\d+)((?:\.\d+)*)</a>',
        lambda m: (
            f"{m.group(1)}{letter[int(m.group(3))] if m.group(2) in in_appendix else m.group(3)}"
            f"{m.group(4)}</a>"
        ),
        page,
    )
    # display equations with a \label get a number on the right and an anchor
    eqs: dict[str, int] = {}

    def tag(m: re.Match[str]) -> str:
        label = re.search(r"\\label\{([^}]+)\}", m.group(0))
        if label is None:
            return m.group(0)
        eqs[label.group(1)] = n = len(eqs) + 1
        eqno = f'<span class="eqno">({n})</span>'
        return f'<span class="eq" id="{label.group(1)}">{m.group(0)}{eqno}</span>'

    page = re.sub(r'<math display="block".*?</math>', tag, page, flags=re.S)
    page = re.sub(
        r'(data-reference="(eq:[^"]+)">)\[[^\]]*\]</a>',
        lambda m: f"{m.group(1)}({eqs[m.group(2)]})</a>",
        page,
    )
    counter = iter(range(1, 1000))
    page = re.sub(r"<figcaption>", lambda _: f"<figcaption><b>Figure {next(counter)}.</b> ", page)
    counter = iter(range(1, 1000))
    return re.sub(r"<caption>", lambda _: f"<caption><b>Table {next(counter)}.</b> ", page)


def main() -> int:
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    values = macros()
    title_m = re.search(r"\\title\{(.*?)\\\\\[[^\]]*\]\s*\\large\s*(.*?)\}\n", main_tex, re.S)
    if title_m is None:
        raise SystemExit("no \\title{...\\\\[..]\\large ...} in paper/main.tex")
    title, subtitle = (" ".join(s.split()) for s in title_m.groups())
    body = main_tex.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    jel = re.search(
        r"\\noindent\\textbf\{JEL codes:\}\s*(.*?)\\quad.*?Keywords:\}\s*(.*?)\n\n", body, re.S
    )
    if abstract is None or jel is None:
        raise SystemExit("no abstract or JEL/keywords line in paper/main.tex")
    body = body.replace(abstract.group(0), "").replace(jel.group(0), "")
    body = body.replace("\\maketitle", "")
    # citeproc appends the references at the end; in the PDF they come before the appendices
    body = re.sub(
        r"\\bibliographystyle\{\w+\}\s*\\bibliography\{\w+\}",
        "\\\\section*{References}\n\nREFS-HERE\n",
        body,
    )
    doc = "\\begin{document}\n"
    abstract_html = pandoc(doc + flatten(abstract.group(1)) + "\n\\end{document}\n", [])
    page = pandoc(
        doc + flatten(body) + "\n\\end{document}\n",
        [
            f"--template={PAPER / 'reader.html'}",
            "--standalone",
            "--citeproc",
            "--bibliography=refs.bib",
            "--number-sections",
            "--toc",
            "--toc-depth=1",
            f"--metadata=title:{title}",
            f"--metadata=subtitle:{subtitle}",
            "--metadata=author:Thanatchon Chongmankhong",
            "--metadata=pagetitle:cpilens: paper",
            f"--metadata=description:{subtitle}",
        ],
    )
    refs = re.search(r'<div id="refs".*?(?=</article>)', page, re.S)
    if refs is None or "<p>REFS-HERE</p>" not in page:
        raise SystemExit("could not move the references before the appendices")
    page = page.replace(refs.group(0), "").replace("<p>REFS-HERE</p>", refs.group(0))
    page = number_like_pdf(page)

    items = "\n".join(
        f'<li class="r r-ref"><span class="r-id">{rq}</span>'
        f'<span class="r-text">{html.escape(expand(text, values))}'
        f"<small>{html.escape(expand(detail, values))}</small></span></li>"
        for rq, text, detail in RESULTS
    )
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    codes, keywords = (html.escape(" ".join(s.split()).rstrip(".")) for s in jel.groups())
    meta = f"<b>cpilens v{version}</b> · JEL {codes} · {keywords}<br>{DISCLAIMER}"
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        '<li><a href="brief.pdf">Policy brief (PDF)</a></li>'
        f'<li><a href="{REPO_URL}">Source code</a></li>'
        f'<li><a href="{REPO_URL}/blob/main/data/SOURCES.md">Data sources</a></li>'
        f'<li><a href="https://doi.org/{DOI}">Zenodo archive</a></li>'
    )
    bib = (
        "@misc{chongmankhong2026cpilens,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title} {subtitle}}},\n"
        "  year   = {2026},\n"
        f"  note   = {{cpilens v{version}}},\n"
        f"  doi    = {{{DOI}}},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    colophon = (
        f"<p>{DISCLAIMER} This is independent work, not a BIS, ECB or Bank of Thailand "
        "publication.</p>"
        "<p>Rendered from <code>paper/main.tex</code> by <code>scripts/web_paper.py</code>. Every "
        "number comes from <code>results/</code> through <code>paper/numbers.tex</code>; the PDF "
        "is authoritative.</p>"
    )
    for marker, value in (
        ("<!--ABSTRACT-->", abstract_html),
        ("<!--META-->", meta),
        ("<!--ACTIONS-->", actions),
        (
            "<!--RESULTS-NOTE-->",
            f"Headline numbers, generated from the results files. Total LLM spend: USD "
            f"{values['LLMSpendUSD']}.",
        ),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise SystemExit(f"reader.html lacks {marker}")
        page = page.replace(marker, value)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(PAPER / "main.pdf", OUT / "paper.pdf")
    shutil.copyfile(PAPER / "brief.pdf", OUT / "brief.pdf")
    (OUT / "figures").mkdir(exist_ok=True)
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(FIGURES / name, OUT / "figures" / name)
    print(f"wrote {OUT.relative_to(ROOT) / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
