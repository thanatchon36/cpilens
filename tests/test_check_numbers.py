import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_path = ROOT / "scripts" / "check_numbers.py"
_spec = importlib.util.spec_from_file_location("check_numbers", _path)
assert _spec and _spec.loader
cn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cn)


def lits(text: str) -> list[str]:
    return [v for _, v in cn.literals(text)]


def test_typed_numbers_are_found_with_line_numbers():
    assert cn.literals("Intro.\nAccuracy was 87.5\\% on 412 products.\n") == [
        (2, "87.5"),
        (2, "412"),
    ]


def test_macros_math_citations_years_and_names_are_allowed():
    text = (
        "In 2566 (2023) we got \\AccWHaiku\\% \\citep[p.~22]{bis2026spectrum}, see "
        "Table~\\ref{tab:1}, $\\alpha = 1$, \\texttt{e5-large}, RQ1, SHA-256, macro-F1.\n"
        "\\begin{equation} x = 2 \\end{equation}\n"
        "\\includegraphics[width=0.5\\textwidth]{fig01.pdf}\\\\[0.3em] % 42 in a comment\n"
    )
    assert lits(text) == []


def test_pragma_exempts_a_line_and_preamble_is_skipped():
    text = "\\geometry{margin=2.5cm}\n\\begin{document}\nJEL: C55, E31, 99. % numbers-ok: JEL\n"
    assert lits(text) == []
    assert lits("\\begin{document}\nWe used 12 models.\n") == ["12"]


def test_undefined_macros_are_reported(tmp_path):
    (tmp_path / "numbers.tex").write_text("\\newcommand{\\AccWHaiku}{90.1}\n")
    known = cn.defined(tmp_path / "numbers.tex")
    assert known == {"AccWHaiku"}
    assert cn.undefined("\\AccWHaiku and \\AccWOpus, \\LaTeX % \\Hidden", known) == {"AccWOpus"}


def test_the_paper_passes():
    assert cn.main(ROOT / "paper") == 0
