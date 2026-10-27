"""Repository paths. Everything is relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
TAXONOMY = ROOT / "taxonomy"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"
