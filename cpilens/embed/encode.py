"""Local multilingual sentence embeddings, cached on disk by model and text hash.

Vectors are L2-normalised float32, so cosine similarity is a dot product. The cache is one
``.npy`` matrix plus a Parquet index of SHA-1 text hashes per model; new texts are appended.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from cpilens.paths import CACHE

MODELS = {
    "bge-m3": {"hf": "BAAI/bge-m3", "prefix": ""},
    "e5-large": {"hf": "intfloat/multilingual-e5-large", "prefix": "query: "},
}
EMB_DIR = CACHE / "emb"


def text_hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def _paths(model: str, root: Path) -> tuple[Path, Path]:
    return root / f"{model}.npy", root / f"{model}.parquet"


def embed(texts: list[str], model: str, root: Path = EMB_DIR, batch_size: int = 128) -> np.ndarray:
    """Embeddings for ``texts`` (rows in the same order), computing only uncached ones."""
    root.mkdir(parents=True, exist_ok=True)
    mat_p, idx_p = _paths(model, root)
    hashes = [text_hash(t) for t in texts]
    if idx_p.exists():
        known = pl.read_parquet(idx_p)["hash"].to_list()
        mat = np.load(mat_p)
    else:
        known, mat = [], np.zeros((0, 0), dtype=np.float32)
    pos = {h: i for i, h in enumerate(known)}
    todo = list(dict.fromkeys(h for h in hashes if h not in pos))
    if todo:
        from sentence_transformers import SentenceTransformer

        spec = MODELS[model]
        text_of = dict(zip(hashes, texts, strict=True))
        st = SentenceTransformer(spec["hf"], device="cuda")
        new = st.encode(
            [spec["prefix"] + text_of[h] for h in todo],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        mat = new if mat.size == 0 else np.vstack([mat, new])
        known += todo
        np.save(mat_p, mat)
        pl.DataFrame({"hash": known}).write_parquet(idx_p)
        pos = {h: i for i, h in enumerate(known)}
    return mat[[pos[h] for h in hashes]]
