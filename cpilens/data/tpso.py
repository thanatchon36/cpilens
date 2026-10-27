"""Thai CPI indices from the TPSO Open API (documented endpoints only).

API documentation: https://index.tpso.go.th/document/cpig
- GET  /OpenApi/Cpig/Month/MasterData  -> regions, index codes, available periods
- POST /OpenApi/Cpig/Month             -> one month, all codes, for one region type

Years are Buddhist Era (BE = CE + 543), as served by the API.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import polars as pl

from cpilens.paths import RAW

API = "https://index-api.tpso.go.th"
USER_AGENT = "cpilens-research/0.1 (public open-data API user)"
API_DIR = RAW / "tpso" / "api"
BASE_YEAR_BE = 2566


def _request(url: str, body: dict[str, object] | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    headers = {"User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        out: bytes = r.read()
        return out


def month_file(year_be: int, month: int, region: str = "TG", out_dir: Path = API_DIR) -> Path:
    return out_dir / f"cpig_{region}_{year_be}_{month:02d}.json"


def fetch_months(
    start: tuple[int, int], end: tuple[int, int], region: str = "TG", out_dir: Path = API_DIR
) -> list[Path]:
    """Download monthly files (inclusive range, BE years); existing files are kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "cpig_month_masterdata.json"
    if not master.exists():
        master.write_bytes(_request(f"{API}/OpenApi/Cpig/Month/MasterData"))
    paths = []
    y, m = start
    while (y, m) <= end:
        f = month_file(y, m, region, out_dir)
        if not f.exists() or f.stat().st_size < 1000:
            body = {
                "yearBase": BASE_YEAR_BE,
                "year": y,
                "month": m,
                "type": region,
                "commodities": [],
            }
            f.write_bytes(_request(f"{API}/OpenApi/Cpig/Month", body))
            time.sleep(0.3)
        paths.append(f)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return paths


def load_index(region: str = "TG", in_dir: Path = API_DIR) -> pl.DataFrame:
    """Tidy monthly index table: one row per (code, month); null indices dropped."""
    rows = []
    for f in sorted(in_dir.glob(f"cpig_{region}_*.json")):
        for r in json.loads(f.read_text()):
            if r["index"] is None:
                continue
            rows.append(
                {
                    "code": r["commodityCode"],
                    "name_th": " ".join(r["commodityNameTH"].split()),
                    "level": r["level"],
                    "year_be": r["year"],
                    "month": r["month"],
                    "index": float(r["index"]),
                }
            )
    df = pl.DataFrame(rows)
    return df.with_columns(pl.date(pl.col("year_be") - 543, pl.col("month"), 1).alias("date")).sort(
        ["code", "date"]
    )
