"""Fetch the raw public files listed in ``data/manifest.csv`` and verify their checksums.

Existing files are only verified. A mismatch means the upstream file changed (MOC open data,
the OFF taxonomy and TPSO releases are not versioned); the run stops rather than silently
using different data. TPSO API indices are fetched separately (``cpilens fetch-tpso``).
"""

from __future__ import annotations

import csv
import hashlib
import urllib.request
from pathlib import Path

from cpilens.data.tpso import USER_AGENT
from cpilens.paths import DATA, RAW

MANIFEST = DATA / "manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_all(manifest: Path = MANIFEST, raw: Path = RAW) -> list[str]:
    """Download missing files; return the list of paths whose checksum does not match."""
    bad = []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            dest = raw / row["path"]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(row["url"], headers={"User-Agent": USER_AGENT})
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as out:
                    while block := r.read(1 << 20):
                        out.write(block)
                tmp.rename(dest)
            if sha256(dest) != row["sha256"]:
                bad.append(row["path"])
    return bad
