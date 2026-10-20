"""Favorite papers: the reader's hand-picked landmarks, from the ♥ button on the site.

Favorites live in the synced state (the sync gist, read via BIOPAPERS_GIST_ID) as
{paper id: {"ts", "label"?, "paper"?}}. Unlike the feed they never age out, and they can be any
paper at all: the site lets you add one by DOI or arXiv ID, and the build resolves those
through OpenAlex. Favorites that aren't in the current window ship in
site/data/favorites.json, so they can be embedded, put on the map, and matched against new papers.
"""

from .common import norm_doi
from .sources import openalex

KEEP = ("id", "doi", "title", "authors", "abstract", "date", "venue", "kind", "url")


def load(state: dict, papers: dict[str, dict], log) -> list[dict]:
    """Records for the favorites that aren't already among `papers`."""
    favs = state.get("favorites") or {}
    extras, unresolved = [], []
    for fid, f in favs.items():
        if fid in papers:
            continue
        snap = f.get("paper") or {}
        if snap.get("title"):
            extras.append({k: snap[k] for k in KEEP if k in snap} | {"id": fid})
        elif norm_doi(fid):
            unresolved.append(norm_doi(fid))
    if unresolved:
        try:
            found = openalex.lookup_dois(unresolved)
        except Exception as ex:
            log(f"  favorites: OpenAlex lookup FAILED ({ex})")
            found = {}
        extras += [{k: p[k] for k in KEEP if k in p} for p in found.values()]
        if missing := [d for d in unresolved if d not in found]:
            log(f"  favorites: couldn't find {', '.join(missing)} in OpenAlex")
    for p in extras:
        p["favorite_only"] = True  # on the map and in similarity, never in the feed tabs
    log(f"  favorites: {len(favs)} total, {len(extras)} outside the window")
    return extras
