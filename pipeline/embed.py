"""Sentence embeddings for "Similar papers" and the For You ranking.

bge-small-en-v1.5 (384-d) over title + abstract, run on CPU via fastembed (ONNX). Each vector
is int8-quantized with its own scale (largest component -> ±127); cosine similarity ignores
per-vector scale, so the browser can compare the int8 rows directly. Shipped as:

    site/data/vectors.bin    N × DIM int8, row i belongs to ids[i]
    site/data/vectors.json   {"model": ..., "dim": DIM, "ids": [...]}

Vectors from the previous build are reused, so each run only embeds papers it hasn't seen.
"""

import json
import os
from pathlib import Path

import numpy as np

from .common import ROOT, client

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
CACHE_DIR = ROOT / ".cache" / "fastembed"


def text_of(p: dict) -> str:
    return f"{p['title']}. {(p.get('abstract') or '')[:1000]}"


def quantize(v: np.ndarray) -> np.ndarray:
    scale = np.abs(v).max(axis=1, keepdims=True)
    return np.round(v / np.where(scale > 0, scale, 1) * 127).astype(np.int8)


def load_previous(base: str | None) -> dict[str, np.ndarray]:
    """id -> int8 vector from the last build (`base` is its data dir, local or URL)."""
    if not base:
        return {}
    try:
        if base.startswith("http"):
            with client(timeout=60) as c:
                meta = c.get(f"{base.rstrip('/')}/vectors.json")
                if meta.status_code == 404:
                    return {}
                meta, raw = meta.raise_for_status().json(), c.get(f"{base.rstrip('/')}/vectors.bin").raise_for_status().content
        else:
            if not (Path(base) / "vectors.json").exists():
                return {}
            meta, raw = json.loads((Path(base) / "vectors.json").read_text()), (Path(base) / "vectors.bin").read_bytes()
    except Exception as ex:
        print(f"! couldn't load previous vectors: {ex}", flush=True)
        return {}
    if meta.get("model") != MODEL or meta.get("dim") != DIM:
        return {}  # model changed: re-embed everything
    rows = np.frombuffer(raw, dtype=np.int8).reshape(-1, DIM)
    return dict(zip(meta["ids"], rows))


def vectors(papers: list[dict], previous: dict[str, np.ndarray], log) -> dict[str, np.ndarray]:
    """id -> int8 vector for every paper, embedding only the ones `previous` doesn't have."""
    todo = [p for p in papers if p["id"] not in previous]
    vecs = dict(previous)
    if todo:
        from fastembed import TextEmbedding

        # One thread per usable core: ONNX Runtime otherwise sizes its pool to the whole machine,
        # which on a big shared host with a small CPU allotment is ~50x slower.
        model = TextEmbedding(MODEL, cache_dir=str(CACHE_DIR), threads=len(os.sched_getaffinity(0)))
        # Batches are padded to their longest text, so embedding in length order saves a lot of work.
        todo.sort(key=lambda p: len(text_of(p)))
        fresh = np.array(list(model.embed([text_of(p) for p in todo], batch_size=64)), dtype=np.float32)
        vecs.update(zip((p["id"] for p in todo), quantize(fresh)))
    log(f"  embed: {len(todo)} new, {len(papers) - len(todo)} reused")
    return vecs


def write(papers: list[dict], vecs: dict[str, np.ndarray], out_dir: Path) -> None:
    ids = [p["id"] for p in papers]
    (out_dir / "vectors.bin").write_bytes(np.stack([vecs[i] for i in ids]).tobytes() if ids else b"")
    (out_dir / "vectors.json").write_text(json.dumps({"model": MODEL, "dim": DIM, "ids": ids}, separators=(",", ":")))
