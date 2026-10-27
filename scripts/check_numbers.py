"""Fail if the paper types a number by hand or uses a macro ``numbers.tex`` does not define.

Every result in ``paper/*.tex`` must come from ``paper/numbers.tex`` (written by ``cpilens
report``). This lists each numeric literal left in the prose of the paper sources, after
removing what is not a typed result: comments, macros, math, citation and reference
arguments, file names, lengths and years. A line may opt out with a ``% numbers-ok: <reason>``
comment (for example JEL codes). Generated files (``numbers.tex``, ``generated/``) are not
scanned. Exit status 1 if any literal or undefined macro remains.

Usage: ``python scripts/check_numbers.py [paper_dir]``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRAGMA = "% numbers-ok:"

# commands whose arguments are not prose (optional and mandatory arguments both dropped)
ARG_CMDS = (
    r"cite\w*|ref|eqref|autoref|label|url|href|texttt|includegraphics|input|include|"
    r"VerbatimInput|bibliography\w*|graphicspath|newcommand|renewcommand|usepackage|"
    r"documentclass|begin|end|hspace|vspace|setlength|addtolength|thanks"
)
STRIP = [
    re.compile(r"(?<!\\)%.*"),  # comments
    re.compile(r"\$\$.*?\$\$|(?<!\\)\$.*?(?<!\\)\$", re.S),  # inline / display math
    re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.S),
    re.compile(
        r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}", re.S
    ),  # math environments
    re.compile(rf"\\(?:{ARG_CMDS})\*?(?:\s*\[[^\]]*\])*(?:\s*\{{[^{{}}]*\}})*"),
    re.compile(r"\\\\\[[^\]]*\]"),  # line break with a length
    re.compile(r"\\[A-Za-z]+"),  # remaining macros (their names hold no digits)
]
LENGTH = re.compile(r"\d+(?:\.\d+)?\s*(?:pt|em|ex|cm|mm|in|bp)\b")
YEAR = re.compile(r"^(?:19|20|25)\d\d$")  # Gregorian and Buddhist-era years
NUMBER = re.compile(r"(?<![A-Za-z0-9.])(?<![A-Za-z]-)\d+(?:[.,]\d+)*(?![A-Za-z0-9])")
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z]*)")
# LaTeX's own capitalised commands that the paper may use
LATEX = {
    "LaTeX", "TeX", "Large", "LARGE", "Huge", "VerbatimInput", "Cref", "Roman",
    "Alph", "Delta", "Sigma", "Phi", "Pr", "Re", "S",
}  # fmt: skip


def sources(paper: Path) -> list[Path]:
    return sorted(p for p in [*paper.glob("*.tex"), *paper.glob("sections/*.tex")]
                  if p.name not in {"numbers.tex"})  # fmt: skip


def body(text: str) -> str:
    """Only the document body counts: the preamble holds lengths and package options."""
    return text.split("\\begin{document}", 1)[1] if "\\begin{document}" in text else text


def literals(text: str) -> list[tuple[int, str]]:
    """(line number, literal) for every hand-typed number in ``text``."""
    pre = len(text) - len(body(text))
    first_line = text[:pre].count("\n")
    lines = body(text).split("\n")
    keep = "\n".join("" if PRAGMA in ln else ln for ln in lines)
    for pat in STRIP:
        # keep line structure: replace with as many newlines as the match spans
        keep = pat.sub(lambda m: "\n" * m.group(0).count("\n") + " ", keep)
    out = []
    for i, ln in enumerate(keep.split("\n")):
        ln = LENGTH.sub(" ", ln)
        for m in NUMBER.finditer(ln):
            if not YEAR.match(m.group(0)):
                out.append((first_line + i + 1, m.group(0)))
    return out


def defined(numbers_tex: Path) -> set[str]:
    if not numbers_tex.exists():
        return set()
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers_tex.read_text()))


def undefined(text: str, known: set[str]) -> set[str]:
    used = set(MACRO_USE.findall(re.sub(r"(?<!\\)%.*", "", body(text))))
    return used - known - LATEX


def main(paper: Path = ROOT / "paper") -> int:
    known = defined(paper / "numbers.tex")
    bad = 0
    for f in sources(paper):
        text = f.read_text()
        rel = f.relative_to(paper.parent)
        for line, lit in literals(text):
            print(f"{rel}:{line}: typed number {lit!r}")
            bad += 1
        for name in sorted(undefined(text, known)):
            print(f"{rel}: macro \\{name} is not defined in numbers.tex")
            bad += 1
    print(f"check_numbers: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper"))
