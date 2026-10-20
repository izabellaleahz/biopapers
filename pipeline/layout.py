"""2D paper map: a UMAP layout of the embeddings, plus each paper's nearest neighbours.

    site/data/map.json   {"ids": [...], "xy": [[x, y], ...], "nn": [[i, ...], ...]}

`ids` is in the same order as vectors.json; xy is scaled to ~[0, 1]; nn[i] lists the K most
similar papers (row indexes, most similar first), which the site draws as the similarity graph.
"""

import json
from pathlib import Path

import numpy as np

K = 8
SEED = 42


def neighbours(unit: np.ndarray, k: int = K, chunk: int = 2048) -> np.ndarray:
    """Exact cosine kNN, chunked so the similarity matrix never has to exist in full."""
    out = np.empty((len(unit), k), dtype=np.int32)
    for start in range(0, len(unit), chunk):
        sims = unit[start:start + chunk] @ unit.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf  # not your own neighbour
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-sims[rows[:, None], top], axis=1)
        out[start:start + chunk] = top[rows[:, None], order]
    return out


def run(out_dir: Path, log) -> None:
    meta = json.loads((out_dir / "vectors.json").read_text())
    ids = meta["ids"]
    if len(ids) <= K + 1:
        return
    vecs = np.frombuffer((out_dir / "vectors.bin").read_bytes(), dtype=np.int8).reshape(-1, meta["dim"]).astype(np.float32)
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-6)

    import umap

    xy = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=SEED).fit_transform(unit)
    # Scale to the central 99% so a few far-off stragglers don't squash the main map; they're
    # pinned just past the edge instead.
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    xy = np.clip((xy - lo) / (hi - lo).clip(1e-6), -0.03, 1.03)
    payload = {"ids": ids, "xy": np.round(xy.astype(np.float64), 4).tolist(), "nn": neighbours(unit).tolist()}
    (out_dir / "map.json").write_text(json.dumps(payload, separators=(",", ":")))
    log(f"  map: {len(ids)} papers laid out")
