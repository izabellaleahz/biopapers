"""2D paper map: a UMAP layout of the embeddings, each paper's nearest neighbours, and named regions.

    site/data/map.json   {"ids": [...], "xy": [[x, y], ...], "nn": [[i, ...], ...],
                          "c": [region index per paper], "regions": [{"label", "x", "y", "n"}, ...]}

`ids` is in the same order as vectors.json; nn[i] lists the K most similar papers (row indexes,
most similar first), which the site draws as the similarity graph.

The layout is kept stable from day to day: UMAP starts from yesterday's positions (new papers start
at the weighted centre of their nearest old neighbours), and the result is rotated / scaled back onto
yesterday's frame, so the map doesn't spin or flip. Regions are k-means clusters of the 2D layout,
also seeded from yesterday's; a region keeps its name while most of its papers stay, and only new
or changed regions are sent to the model for a name.
"""

import json
from pathlib import Path

import numpy as np

K = 8
SEED = 42
N_REGIONS = 30
KEEP_NAME = 0.5  # a region keeps yesterday's name if this share of its papers were in that region
NAME_TITLES = 12
STABLE_EPOCHS = 60  # vs UMAP's default 500 for a fresh layout of this size
STABLE_LR = 0.25    # vs 1.0: small steps from yesterday's positions

NAME_INSTRUCTIONS = """Each numbered group below is a cluster of research papers that sit together on a similarity map. Give each group a short, specific name (2-4 words) for its shared topic, as a scientist would label that region of the map: "T cell exhaustion", "Protein language models", "Gut microbiome", not "Biology" or "Various topics". Name every group exactly once, using its number as c."""

NAME_SCHEMA = {
    "type": "object",
    "properties": {"regions": {"type": "array", "items": {
        "type": "object",
        "properties": {"c": {"type": "integer"}, "label": {"type": "string"}},
        "required": ["c", "label"],
        "additionalProperties": False,
    }}},
    "required": ["regions"],
    "additionalProperties": False,
}


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


def initial_positions(ids: list[str], nn: np.ndarray, sims_of, previous: dict) -> np.ndarray | None:
    """Yesterday's position for papers we've laid out before; for new ones, the similarity-weighted
    centre of their nearest papers that do have one. None when there's no usable previous map."""
    old = {pid: xy for pid, xy in zip(previous.get("ids", []), previous.get("xy", []))}
    known = np.array([pid in old for pid in ids])
    if known.sum() < max(50, len(ids) // 2):
        return None
    init = np.zeros((len(ids), 2))
    init[known] = [old[pid] for pid, k in zip(ids, known) if k]
    centre = init[known].mean(axis=0)
    for i in np.where(~known)[0]:
        js = [j for j in nn[i] if known[j]]
        if not js:
            init[i] = centre
            continue
        w = np.clip(sims_of(i, js), 0, None) ** 4 + 1e-6
        init[i] = (init[js] * w[:, None]).sum(axis=0) / w.sum()
    return init + np.random.default_rng(SEED).normal(0, 1e-3, init.shape)  # UMAP dislikes exact duplicates


def align(xy: np.ndarray, shared: list[int], target: np.ndarray) -> np.ndarray:
    """Similarity transform (rotation/reflection, uniform scale, shift) fitted by Procrustes on the
    papers both maps share, then applied to every paper — so today's map sits in yesterday's frame."""
    mu_x, mu_t = xy[shared].mean(axis=0), target.mean(axis=0)
    a, b = xy[shared] - mu_x, target - mu_t
    u, s, vt = np.linalg.svd(a.T @ b)
    return ((xy - mu_x) @ (u @ vt)) * (s.sum() / (a ** 2).sum()) + mu_t


def normalise(xy: np.ndarray) -> np.ndarray:
    """Scale the central 99% into the unit square, the same factor on both axes so the layout keeps its
    shape, and centre the shorter axis. Stragglers outside get pinned at the edge. Only used for a fresh
    layout; a stable one stays in yesterday's frame."""
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    span = (hi - lo).max() or 1.0
    return (xy - lo) / span + (1 - (hi - lo) / span) / 2


def regions(xy: np.ndarray, ids: list[str], unit: np.ndarray, papers: dict[str, dict], previous: dict,
            name, log) -> tuple[np.ndarray, list[dict]]:
    from sklearn.cluster import KMeans

    k = min(N_REGIONS, max(2, len(ids) // 60))
    prev_regions = previous.get("regions") or []
    seed = np.array([[r["x"], r["y"]] for r in prev_regions]) if len(prev_regions) == k else None
    km = KMeans(n_clusters=k, init=seed if seed is not None else "k-means++", n_init=1 if seed is not None else 4,
                random_state=SEED).fit(xy)
    labels = km.labels_

    # Keep a name when most of a region's papers sat in one region yesterday.
    old_region = dict(zip(previous.get("ids", []), previous.get("c", [])))
    names: list[str | None] = [None] * k
    claims = []  # (share of members from that old region, new region, old region)
    for c in range(k):
        members = [ids[i] for i in np.where(labels == c)[0]]
        olds = [old_region[m] for m in members if m in old_region]
        if olds:
            top = max(set(olds), key=olds.count)
            if olds.count(top) / len(members) >= KEEP_NAME and top < len(prev_regions):
                claims.append((olds.count(top) / len(members), c, top))
    taken = set()
    for _, c, top in sorted(claims, reverse=True):  # a region that split: only the bigger share keeps the name
        if top not in taken and prev_regions[top].get("label"):
            names[c] = prev_regions[top]["label"]
            taken.add(top)
    todo = [c for c in range(k) if not names[c]]
    if todo:
        groups = []
        for c in todo:
            idx = np.where(labels == c)[0]
            centroid = unit[idx].mean(axis=0)
            closest = idx[np.argsort(-(unit[idx] @ centroid))[:NAME_TITLES]]
            groups.append(f"[{c}]\n" + "\n".join(f"- {papers.get(ids[i], {}).get('title', '')}" for i in closest))
        got = name(NAME_INSTRUCTIONS, "\n\n".join(groups), NAME_SCHEMA) or {}
        for r in got.get("regions", []):
            if r.get("c") in todo and r.get("label"):
                names[r["c"]] = r["label"].strip()
    log(f"  map: {k} regions, {k - len(todo)} kept their name, {len(todo)} named"
        + ("" if all(names) else f" ({sum(not n for n in names)} unnamed — no model key?)"))
    out = [{"label": names[c] or "", "x": round(float(km.cluster_centers_[c][0]), 4),
            "y": round(float(km.cluster_centers_[c][1]), 4), "n": int((labels == c).sum())} for c in range(k)]
    return labels, out


def run(out_dir: Path, log, previous: dict | None = None, papers: dict[str, dict] | None = None, name=None) -> dict | None:
    """`previous` is yesterday's map.json; `papers` maps id -> record (for region titles); `name` is
    ai.complete_json, or None to leave regions unnamed. Returns {"region": {id: index}, "labels": [...]}
    so the feed can carry each paper's region without loading the map."""
    previous, papers = previous or {}, papers or {}
    meta = json.loads((out_dir / "vectors.json").read_text())
    ids = meta["ids"]
    if len(ids) <= K + 1:
        return None
    vecs = np.frombuffer((out_dir / "vectors.bin").read_bytes(), dtype=np.int8).reshape(-1, meta["dim"]).astype(np.float32)
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-6)
    nn = neighbours(unit)

    import umap

    init = initial_positions(ids, nn, lambda i, js: unit[js] @ unit[i], previous)
    # Starting from yesterday, a short optimisation settles new papers in without re-arranging the rest.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=SEED,
                        init=init if init is not None else "spectral",
                        **({"n_epochs": STABLE_EPOCHS, "learning_rate": STABLE_LR} if init is not None else {}))
    xy = reducer.fit_transform(unit)
    if init is not None:
        old = {pid: p for pid, p in zip(previous["ids"], previous["xy"])}
        shared = [i for i, pid in enumerate(ids) if pid in old]
        xy = align(xy, shared, np.array([old[ids[i]] for i in shared]))
        lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
        if (lo < -0.1).any() or (hi > 1.1).any():  # drifted out of frame: shrink back uniformly (keeps the shape)
            xy = normalise(xy)
    else:
        xy = normalise(xy)
    xy = np.clip(xy, -0.05, 1.05)  # pins the last few stragglers at the edge

    labels, regs = regions(xy, ids, unit, papers, previous, name or (lambda *a: None), log)
    payload = {"ids": ids, "xy": np.round(xy.astype(np.float64), 4).tolist(), "nn": nn.tolist(),
               "c": labels.astype(int).tolist(), "regions": regs}
    (out_dir / "map.json").write_text(json.dumps(payload, separators=(",", ":")))
    log(f"  map: {len(ids)} papers laid out ({'stable, from yesterday' if init is not None else 'fresh'})")
    return {"region": dict(zip(ids, labels.astype(int).tolist())), "labels": [r["label"] for r in regs]}

