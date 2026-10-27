"""Assign interest topics and PI-watchlist hits to papers (rules from config/*.yaml)."""

import re
import unicodedata


def _keyword_re(kw: str) -> re.Pattern:
    # Short ALL-CAPS tokens (STING, ERV, CD8) are case-sensitive whole words (plural ok);
    # everything else is a case-insensitive prefix match at a word start.
    if len(kw) <= 5 and kw.isupper():
        return re.compile(rf"(?<![A-Za-z0-9]){re.escape(kw)}s?(?![a-z0-9])")
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(kw)}", re.IGNORECASE)


class Tagger:
    def __init__(self, topics: list[dict], people: list[dict], sections: dict | None = None):
        venue_limits = {k: {v.lower() for v in cfg.get("only_venues", [])}
                        for k, cfg in (sections or {}).items()}
        self.topics = []
        for t in topics:
            self.topics.append({
                "slug": t["slug"],
                "section": t.get("section", "bio"),
                "only_venues": venue_limits.get(t.get("section", "bio"), set()),
                "categories": {c.lower() for c in t.get("categories", [])},
                "arxiv": tuple(t.get("arxiv", [])),
                "venues": {v.lower() for v in t.get("venues", [])},
                "keywords": [_keyword_re(k) for k in t.get("keywords", [])],
                "title_keywords": [_keyword_re(k) for k in t.get("title_keywords", [])],
            })
        self.people = [(p["name"], *_name_parts(p["name"])) for p in people]
        self.section_of = {t["slug"]: t["section"] for t in self.topics}

    def topics_for(self, p: dict) -> list[str]:
        cats = p.get("arxiv_cats") or []
        return [t["slug"] for t in self.topics
                if (not t["only_venues"] or p["venue"].lower() in t["only_venues"])
                and (p["category"] in t["categories"]
                or p["venue"].lower() in t["venues"]
                or (t["arxiv"] and any(c.startswith(t["arxiv"]) for c in cats))
                or self._keyword_hit(t, p))]

    def keyword_topics_for(self, p: dict, section: str) -> list[str]:
        """Topics in `section` matched by keywords alone — gates high-volume sources like cs.LG."""
        return [t["slug"] for t in self.topics if t["section"] == section and self._keyword_hit(t, p)]

    @staticmethod
    def _keyword_hit(t: dict, p: dict) -> bool:
        text = f"{p['title']} {p['abstract']} {' '.join(p.get('keywords') or [])}"
        return (any(k.search(text) for k in t["keywords"])
                or any(k.search(p["title"]) for k in t["title_keywords"]))

    def pis_for(self, p: dict) -> list[str]:
        """Watched PIs among the authors. Sticky: author lists get truncated on output, so a
        PI found on an earlier run (and still on the watchlist) stays tagged."""
        authors = [_name_parts(a) for a in p["authors"]]
        found = {name for name, first, last in self.people
                 if any(last == a_last and a_first[:1] == first[:1] for a_first, a_last in authors)}
        found |= set(p.get("pis") or []) & {name for name, _, _ in self.people}
        return sorted(found)

    def is_pi_author(self, author: str, pis: list[str]) -> bool:
        first, last = _name_parts(author)
        return any(last == p_last and first[:1] == p_first[:1]
                   for name, p_first, p_last in self.people if name in pis)


def _name_parts(name: str) -> tuple[str, str]:
    """('c', 'uhler') from 'Caroline Uhler' / 'C. Uhler' / 'Uhler C' — accent-insensitive."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    parts = [x for x in re.split(r"[\s.]+", name) if x]
    if not parts:
        return "", ""
    if len(parts) >= 2 and len(parts[-1]) == 1:  # "Uhler C" (PubMed style)
        return parts[-1], parts[0]
    return parts[0], parts[-1]


def by_similarity(candidates: list[dict], tagged: list[dict], vecs: dict, section_of: dict[str, str],
                  k: int = 8, min_votes: int = 6) -> list[dict]:
    """Topics for papers the rules missed, from the papers most like them.

    Broad journals (Nature, Science, PNAS, Nature Communications…) have no subject categories, so
    keyword rules miss most of their biology. A candidate gets topic T when at least `min_votes` of its
    `k` nearest tagged papers have T, and only if it's as close to them as a typical tagged paper is
    to its own neighbours (so an astronomy paper can't borrow a topic from its least-unlike biology
    papers). Sets p["topics"] and p["tagged_by"] on the candidates it tags, and returns those."""
    import numpy as np

    pool = [p for p in tagged if p["id"] in vecs and any(section_of.get(t) == "bio" for t in p["topics"])]
    cands = [p for p in candidates if p["id"] in vecs]
    if len(pool) < 100 or not cands:
        return []
    unit = lambda rows: rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(1e-6)
    m = unit(np.stack([vecs[p["id"]] for p in pool]).astype(np.float32))
    # The bar: how close a tagged paper typically is to its k nearest tagged papers (30th percentile).
    sample = m[np.random.default_rng(0).choice(len(m), min(1000, len(m)), replace=False)]
    s = sample @ m.T
    kth = np.sort(s, axis=1)[:, -(k + 1)]  # the +1 skips the paper itself
    bar = float(np.percentile(kth, 30))
    c = unit(np.stack([vecs[p["id"]] for p in cands]).astype(np.float32))
    sims = c @ m.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    out = []
    for i, p in enumerate(cands):
        if sims[i, top[i]].min() < bar:
            continue
        votes: dict[str, int] = {}
        for j in top[i]:
            for t in pool[j]["topics"]:
                if section_of.get(t) == "bio":
                    votes[t] = votes.get(t, 0) + 1
        if topics := sorted(t for t, n in votes.items() if n >= min_votes):
            p["topics"], p["tagged_by"] = topics, "similar papers"
            out.append(p)
    return out
