"""Official base-year weights backed out from the TPSO monthly CPI release.

The national release (``Cpig{MM}{YYYY}_tg.pdf``) prints, for 29 aggregates, the month's
price-updated weight share s_t ("สัดส่วนน้ำหนัก") next to the index I_t. Under a fixed-base
Laspeyres index the base-year weight is

    w0_i = s_i,t * I_total,t / I_i,t

which is invariant across months up to rounding; we use that invariance as a check. These w0
match the 2566 expenditure weights in TPSO's rebasing note (e.g. fresh food 17.43), but they do
not add up the hierarchy exactly, because the index is a Lowe index whose price reference is
the link month (Dec 2567): weights are price-updated item by item before aggregation. The
shares that do add up, and that reproduce every published index, are the price-updated ones

    w_pr_i = w0_i * I_i,L / I_total,L,   L = Dec 2567,
    I_parent,t / I_parent,L = sum_i (w_pr_i / w_pr_parent) * I_i,t / I_i,L.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from pypdf import PdfReader

ROW = re.compile(
    r"^(?P<label>.*?)\s+(?P<share>\d+\.\d\d)\s+(?P<index>\d+\.\d\d)\s+(?P<prev>\d+\.\d\d)\s"
)
FIRST_LABEL = "ดัชนีราคาผู้บริโภคทั่วไป 100.00"
PRICE_REF = (2567, 12)  # (year_be, month): link month of the 2566-based series


def release_table(pdf: Path) -> pl.DataFrame:
    """Rows of the weights table, in printed order: label, share (%), index this month."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
    start = text.index(FIRST_LABEL)
    rows: list[dict[str, object]] = []
    for line in text[start:].splitlines():
        m = ROW.match(line.strip())
        if m is None:
            if rows:
                break
            continue
        rows.append(
            {
                "row": len(rows) + 1,
                "label_pdf": m["label"].strip().lstrip("-").strip(),
                "share": float(m["share"]),
                "index": float(m["index"]),
            }
        )
    return pl.DataFrame(rows)


def base_weights(pdf: Path) -> pl.DataFrame:
    """Backed-out base weights (% of the all-items basket) for every row of the table."""
    t = release_table(pdf)
    total = t.filter(pl.col("row") == 1)["index"][0]
    return t.with_columns((pl.col("share") * total / pl.col("index")).alias("w0"))


def build_weights(
    pdfs: dict[tuple[int, int], Path], index: pl.DataFrame, aggregates: pl.DataFrame
) -> pl.DataFrame:
    """Base weights per aggregate code from several monthly releases, with checks.

    ``pdfs`` maps (year_be, month) to a release PDF. Each printed index must equal the API
    index of the mapped code for that month; otherwise the row mapping is wrong and we stop.
    """
    cols = []
    for (y, m), pdf in sorted(pdfs.items()):
        w = base_weights(pdf).join(aggregates.select("row", "code"), on="row", how="inner")
        api = index.filter((pl.col("year_be") == y) & (pl.col("month") == m)).select(
            "code", pl.col("index").alias("api_index")
        )
        chk = w.join(api, on="code", how="left")
        bad = chk.filter(pl.col("api_index").is_null() | (pl.col("api_index") != pl.col("index")))
        if bad.height or w.height != aggregates.height:
            raise ValueError(f"release {pdf.name}: row/code mapping does not match the API\n{bad}")
        cols.append(w.select("code", pl.col("w0").alias(f"w0_{y}_{m:02d}")))
    out = aggregates
    for c in cols:
        out = out.join(c, on="code", how="left")
    wcols = [c for c in out.columns if c.startswith("w0_")]
    ref = index.filter((pl.col("year_be") == PRICE_REF[0]) & (pl.col("month") == PRICE_REF[1]))
    i_ref = dict(zip(ref["code"], ref["index"], strict=True))
    return (
        out.with_columns(
            pl.mean_horizontal(wcols).round(3).alias("w0"),
            (pl.max_horizontal(wcols) - pl.min_horizontal(wcols)).round(3).alias("w0_spread"),
        )
        .with_columns(
            (
                pl.col("w0")
                * pl.col("code").replace_strict(i_ref, return_dtype=pl.Float64)
                / i_ref["00000"]
            )
            .round(3)
            .alias("w_pr"),
        )
        .sort("row")
    )


def build_from_dir(
    release_dir: Path, index: pl.DataFrame, aggregates: pl.DataFrame, min_months: int = 5
) -> tuple[pl.DataFrame, list[tuple[int, int]]]:
    """Use every release whose table parses and verifies against the API.

    Releases with a different table layout (several 2568 issues) are skipped, not guessed.
    """
    used: dict[tuple[int, int], Path] = {}
    for pdf in sorted(release_dir.glob("Cpig*_tg.pdf")):
        ym = (int(pdf.name[6:10]), int(pdf.name[4:6]))
        try:
            build_weights({ym: pdf}, index, aggregates)
        except ValueError:
            continue
        used[ym] = pdf
    if len(used) < min_months:
        raise ValueError(f"only {len(used)} releases verified; need {min_months}")
    return build_weights(used, index, aggregates), sorted(used)
