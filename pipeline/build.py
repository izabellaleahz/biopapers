"""Daily build: fetch every source -> merge with previous data -> dedup -> tag -> write site/data.

Usage:
    python -m pipeline.build                       # daily: last 3 days, merged with site/data/papers.json
    python -m pipeline.build --days 30             # backfill
    python -m pipeline.build --previous https://<user>.github.io/biopapers/data
    python -m pipeline.build --only biorxiv,rss    # quick local run with a subset of sources

Outputs:
    site/data/papers.json   last 10 days + tab/PI config (the site loads this first)
    site/data/older.json    the rest of the window (loaded in the background)
    site/data/ml.json       ML-only papers for the ML / CS section (loaded when you open it)
    site/data/vectors.*     int8 title+abstract embeddings for every paper above (see embed.py)
    site/data/map.json      2D UMAP layout + nearest-neighbour graph of those (see layout.py)
    site/data/favorites.json   ♥ favorites outside the window, so they're on the map (see favorites.py)
    site/data/ai_pending.json   AI-scoring batches still running, collected next build (see ai.py)
    site/data/ai_dropped.json   scores of untagged journal papers that didn't make the cut (not re-scored)
    site/alerts.xml         RSS feed of PI-watchlist + buzzing papers (subscribe for alerts)
    archive/papers-YYYY-MM.jsonl.gz   every paper seen, untruncated and never pruned (see archive.py)
"""

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from xml.sax.saxutils import escape

from .common import ROOT, client, load_config, read_gist_state, strip_label, title_key, today
from . import ai, archive, embed, favorites, layout, tagging
from .sources import buzz, huggingface, journals, openalex, preprints
from .tagging import Tagger

OUT_DIR = ROOT / "site" / "data"
RECENT_DAYS = 10  # papers.json = last 10 days (loads first); older.json = the rest
ALERTS_PATH = ROOT / "site" / "alerts.xml"
ALL_SOURCES = ["biorxiv", "medrxiv", "arxiv", "rss", "openalex", "search", "pis", "ml", "hf", "buzz",
               "embed", "ai"]  # the last two are enrichment steps, not sources
BUZZ_ALERT_SCORE = 5       # bio: e.g. 100 HN points, 5 Bluesky posts, or 2 press stories
BUZZ_ALERT_SCORE_ML = 15   # ML-only: ~300 HF upvotes — HF votes are plentiful, so a higher bar
MAX_AUTHORS = 13  # keep first 10 + last 3 (PIs are usually last)
MAX_ABSTRACT = 1500
MAX_ABSTRACT_ML = 900
PI_KEEP_DAYS = 180  # watched-PI papers stick around longer than the general feed
OPENALEX_LOOKBACK = 10  # days; OpenAlex lists journal papers 1-3 days after they're out
KEEP_UNTAGGED_SCORE = 3  # a journal paper no rule tagged stays if its AI score is at least this


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_json(base: str | None, name: str) -> dict | None:
    """One file from the last build: `base` is the data dir, local or the live site's URL."""
    src = f"{base.rstrip('/')}/{name}" if base else ""
    try:
        if src.startswith("http"):
            with client() as c:
                resp = c.get(src)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
        if src and Path(src).exists():
            return json.loads(Path(src).read_text())
    except Exception as ex:
        log(f"! couldn't load previous data from {src}: {ex}")
    return None


def load_previous(base: str | None) -> list[dict]:
    return [p for name in ("papers.json", "older.json", "ml.json")
            for p in (fetch_json(base, name) or {}).get("papers", [])]


def timed(label: str, fn, *args) -> list | dict:
    """Run one source; a failure is logged and yields nothing rather than killing the build."""
    t0 = time.time()
    try:
        out = fn(*args)
        log(f"  {label}: {len(out)} ({time.time() - t0:.0f}s)")
        return out
    except Exception as ex:
        log(f"  {label}: FAILED ({ex.__class__.__name__}: {ex})")
        return {} if label.startswith("buzz") else []


def fetch_everything(days: int, keep_days: int, only: set[str], tagger: Tagger) -> tuple[list[dict], dict]:
    sources = load_config("sources")
    pre = sources.get("preprints", {})
    papers: list[dict] = []

    for server in ("biorxiv", "medrxiv"):
        if server in only and pre.get(server):
            papers += timed(server, preprints.fetch_rxiv, server, days)

    if "arxiv" in only:
        keep_all, kw_only = pre.get("arxiv_keep_all", []), pre.get("arxiv_keyword_only", [])
        got = timed("arxiv", preprints.fetch_arxiv, keep_all + kw_only, days)
        papers += [p for p in got
                   if any(c.startswith(tuple(keep_all)) for c in p.get("arxiv_cats", []))
                   or tagger.keyword_topics_for(p, "bio")]

    ml = sources.get("ml", {})
    if "ml" in only and ml.get("arxiv"):
        papers += timed("ml arxiv", preprints.fetch_arxiv, ml["arxiv"], min(days, ml.get("firehose_days", 3)))
    if "ml" in only and ml.get("arxiv_keyword_only"):
        got = timed("ml arxiv (keyword-gated)", preprints.fetch_arxiv, ml["arxiv_keyword_only"], days)
        papers += [p for p in got if tagger.keyword_topics_for(p, "ml")]
    if "hf" in only and ml.get("huggingface"):
        # Upvotes keep climbing for days, so always refresh at least the last week.
        hf_days = min(max(days, 7), ml.get("huggingface_days", 30))
        papers += timed("huggingface", huggingface.fetch_daily, hf_days)

    if "rss" in only:
        papers += journals.fetch_all_feeds(sources.get("rss", {}), log)
    if "openalex" in only:  # OpenAlex indexes a few days late: look further back (repeats merge by DOI)
        papers += journals.fetch_issn_journals(sources.get("openalex", {}), max(days, OPENALEX_LOOKBACK), log)

    if "search" in only:
        for t in load_config("topics")["topics"]:
            for q in t.get("openalex_search", []):
                papers += timed(f"search {q}", openalex.works_by_filter,
                                f"title_and_abstract.search:{q}", keep_days)

    if "pis" in only:
        for person in load_config("people").get("people", []):
            if aid := person.get("openalex"):
                papers += timed(f"pi {person['name']}", openalex.works_by_filter,
                                f"author.id:{aid}", PI_KEEP_DAYS)

    signals: dict[str, dict] = {}
    if "buzz" in only:
        cfg = sources.get("buzz", {})
        look = cfg.get("lookback_days", 7)
        fetchers = {"hackernews": (buzz.hackernews, look), "bluesky": (buzz.bluesky, look),
                    "sciencedaily": (buzz.sciencedaily,)}
        for name, (fn, *args) in fetchers.items():
            if cfg.get(name):
                for doi, sig in timed(f"buzz {name}", fn, *args).items():
                    signals.setdefault(doi, {}).update(sig)
    return papers, signals


def merge(previous: list[dict], fresh: list[dict]) -> dict[str, dict]:
    """By id; fresh data wins, but first-seen date, buzz and enrichment carry over."""
    merged = {p["id"]: p for p in previous}
    for p in fresh:
        old = merged.get(p["id"], {})
        new = {**old, **{k: v for k, v in p.items() if v is not None}}
        for k in ("abstract", "authors"):  # don't let a thin RSS record clobber an enriched one
            if len(old.get(k) or "") > len(p[k]):
                new[k] = old[k]
        merged[p["id"]] = new
    return merged


def enrich(papers: dict[str, dict], signals: dict[str, dict]) -> None:
    """Fill thin RSS records (short abstract / no authors) from OpenAlex, and pull in
    buzzing papers we don't have yet."""
    # New papers often aren't in OpenAlex yet, so keep retrying for a week.
    week_ago = (today() - dt.timedelta(days=7)).isoformat()
    thin = [p["doi"] for p in papers.values()
            if p.get("doi") and not p.get("enriched") and p.get("added", "9999") >= week_ago
            and (len(p["abstract"]) < 400 or not p["authors"])]
    missing = [d for d in signals if d not in papers]
    try:
        found = openalex.lookup_dois(thin + missing)
    except Exception as ex:
        log(f"  enrich: FAILED ({ex})")
        return
    for doi in thin:
        p = papers[doi]
        if o := found.get(doi):
            p["enriched"] = True
            if len(o["abstract"]) > len(p["abstract"]):
                p["abstract"] = o["abstract"]
            if len(o["authors"]) > len(p["authors"]):
                p["authors"] = o["authors"]
    for doi in missing:
        if o := found.get(doi):
            papers[doi] = {**o, "enriched": True}
    log(f"  enrich: {sum(d in found for d in thin)}/{len(thin)} thin records filled, "
        f"{sum(d in found for d in missing)}/{len(missing)} buzzing papers added")


def apply_buzz(papers: dict[str, dict], signals: dict[str, dict]) -> None:
    for doi, sig in signals.items():
        if p := papers.get(doi):
            p["buzz"] = sig
    for p in papers.values():
        b = p.get("buzz") or {}
        p["buzz_score"] = round(b.get("hn", 0) / 20 + b.get("bsky", 0) + b.get("bsky_likes", 0) / 50
                                + 3 * len(b.get("news", [])) + (p.get("hf") or {}).get("upvotes", 0) / 20, 2)


def dedup_versions(papers: dict[str, dict]) -> dict[str, dict]:
    """Collapse a preprint and its journal version (same title) into the journal record."""
    by_title: dict[str, list[dict]] = {}
    for p in papers.values():
        key = title_key(p["title"])
        # Short titles ("Editorial", "In brief") collide across unrelated papers — never merge those.
        by_title.setdefault(key if len(key) >= 30 else p["id"], []).append(p)
    out = {}
    for group in by_title.values():
        group.sort(key=lambda p: (p["kind"] != "article", p["date"]))
        keep, rest = group[0], group[1:]
        if rest:
            keep["aliases"] = sorted({a for r in rest for a in [r["id"], *r.get("aliases", [])]})
            if pre := next((r for r in rest if r["kind"] == "preprint"), None):
                keep.setdefault("preprint_url", pre["url"])
            keep["added"] = min(r.get("added", r["date"]) for r in group)
            keep["buzz_score"] = max(r.get("buzz_score", 0) for r in group)
        out[keep["id"]] = keep
    return out


def sections_for(p: dict, tagger: Tagger, ml_cats: tuple) -> list[str]:
    """Which top-level sections (Bio / ML) a paper shows up in."""
    secs = {tagger.section_of[t] for t in p["topics"]}
    if p["pis"]:
        secs.add("bio")
    if p.get("hf") or any(c.startswith(ml_cats) for c in p.get("arxiv_cats", [])):
        secs.add("ml")
    if not secs and p.get("buzz_score", 0) > 0:  # off-topic but buzzing
        secs.add("ml" if p["venue"] == "arXiv" else "bio")
    return sorted(secs)


def trim(p: dict, tagger: Tagger) -> dict:
    """The copy that ships to the site: long author lists and abstracts cut (the archive keeps the full record)."""
    p = {**p}
    if len(p["authors"]) > MAX_AUTHORS:
        p["n_authors"] = len(p["authors"])
        middle = [a for a in p["authors"][10:-3] if tagger.is_pi_author(a, p["pis"])]
        p["authors"] = p["authors"][:10] + ["…", *middle] + p["authors"][-3:]
    # ML-only papers get a shorter abstract (most have an HF TL;DR, and there are a lot of them).
    cap = MAX_ABSTRACT_ML if p["sections"] == ["ml"] else MAX_ABSTRACT
    p["abstract"] = strip_label(p["abstract"])[:cap]
    return p


def finalize(papers: dict[str, dict], tagger: Tagger, keep_days: int, ml_cfg: dict) -> tuple[list[dict], list[dict]]:
    """(papers for the site, journal papers in the window that no rule tagged — see tagging.by_similarity)."""
    cutoff = (today() - dt.timedelta(days=keep_days)).isoformat()
    pi_cutoff = (today() - dt.timedelta(days=PI_KEEP_DAYS)).isoformat()
    firehose_cutoff = (today() - dt.timedelta(days=ml_cfg.get("firehose_days", 3))).isoformat()
    ml_cats = tuple(ml_cfg.get("arxiv", [])) or ("\0",)
    kept, untagged = [], []
    for p in papers.values():
        p["topics"] = tagger.topics_for(p)
        p["pis"] = tagger.pis_for(p)
        p.setdefault("added", today().isoformat())
        p["sections"] = sections_for(p, tagger, ml_cats)
        if not p["sections"]:
            if p["kind"] == "article" and p["date"] >= cutoff:
                untagged.append(p)
            continue
        # Buzz-only papers can be old (HN loves a 2015 classic) — age those by when we first saw them.
        ref = p["date"] if (p["topics"] or p["pis"]) else max(p["date"], p["added"])
        if ref < (pi_cutoff if p["pis"] else cutoff):
            continue
        # The ML firehose is huge: past a few days, keep only papers someone is paying attention to
        # (or that sit in a niche tab marked `keep`).
        if (p["sections"] == ["ml"] and not p.get("buzz_score") and p["date"] < firehose_cutoff
                and not tagger.keep_topics.intersection(p["topics"])):
            continue
        kept.append(trim(p, tagger))
    kept.sort(key=lambda p: (p["date"], p.get("buzz_score", 0)), reverse=True)
    return kept, untagged


def write_alerts(papers: list[dict], site_url: str) -> None:
    recent = (today() - dt.timedelta(days=14)).isoformat()
    items = []
    for p in papers:
        reasons = [f"PI: {', '.join(p['pis'])}"] if p["pis"] else []
        bar = BUZZ_ALERT_SCORE_ML if p["sections"] == ["ml"] else BUZZ_ALERT_SCORE
        if p.get("buzz_score", 0) >= bar:
            reasons.append(f"buzz {p['buzz_score']:.0f}")
        if not reasons or p["added"] < recent:
            continue
        title = f"[{' · '.join(reasons)}] {p['title']}"
        items.append(
            f"<item><title>{escape(title)}</title>"
            f"<link>{escape(p['url'])}</link><guid isPermaLink=\"false\">{escape(p['id'])}</guid>"
            f"<description>{escape(p['venue'] + ' — ' + p['abstract'][:500])}</description>"
            f"<pubDate>{dt.date.fromisoformat(p['added']).strftime('%a, %d %b %Y 00:00:00 GMT')}</pubDate></item>"
        )
    ALERTS_PATH.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>biopapers alerts</title><link>{escape(site_url)}</link>"
        "<description>New papers from watched PIs, and papers getting attention</description>"
        + "".join(items) + "</channel></rss>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3, help="fetch lookback (default 3)")
    ap.add_argument("--backfill-days", type=int, default=30, help="lookback used when no previous data")
    ap.add_argument("--keep-days", type=int, default=30, help="drop papers older than this")
    ap.add_argument("--previous", default=str(OUT_DIR), help="data dir (path or URL) of the last build")
    ap.add_argument("--only", default=",".join(ALL_SOURCES), help=f"subset of {ALL_SOURCES}")
    ap.add_argument("--site-url", default="", help="public site URL (for alerts.xml)")
    ap.add_argument("--ai-wait", type=float, default=20, help="minutes to wait for the AI-scoring batch")
    args = ap.parse_args()

    topics_file = load_config("topics")
    topics_cfg = topics_file["topics"]
    people_cfg = load_config("people").get("people", [])
    sections_cfg = topics_file.get("sections", {"bio": {"label": "Bio"}})
    tagger = Tagger(topics_cfg, people_cfg, sections_cfg)

    previous = load_previous(args.previous)
    days = args.days if previous else max(args.days, args.backfill_days)
    log(f"previous: {len(previous)} papers · fetching last {days} days")

    only = set(args.only.split(","))
    fresh, signals = fetch_everything(days, args.keep_days, only, tagger)
    papers = merge(previous, fresh)
    enrich(papers, signals)
    apply_buzz(papers, signals)
    papers = dedup_versions(papers)
    final, untagged = finalize(papers, tagger, args.keep_days, load_config("sources").get("ml", {}))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        synced = read_gist_state(os.environ.get("BIOPAPERS_GIST_ID"))
    except Exception as ex:
        log(f"! couldn't read the sync gist: {ex}")
        synced = {}
    in_window = {p["id"]: p for p in final}
    for p in final:
        for a in p.get("aliases", []):
            in_window.setdefault(a, p)
    extras = favorites.load(synced, in_window, log)

    # Journal papers no rule tagged get scored too, and stay if they matter to you (AI score) or sit
    # among papers of one topic (similarity). Scores of the ones dropped are remembered, so a paper
    # OpenAlex keeps listing for a week or two isn't re-scored every day.
    dropped_before = (fetch_json(args.previous, "ai_dropped.json") or {}).get("papers", {})
    for p in untagged:
        if not p.get("ai") and p["id"] in dropped_before:
            p["ai"] = dropped_before[p["id"]]
    pending = (fetch_json(args.previous, "ai_pending.json") or {}).get("batches", [])
    if "ai" in only:
        pending = ai.run(final + untagged, pending, args.ai_wait, log, synced)
        for p in final:  # final holds trimmed copies; the archive reads `papers`
            if "ai" in p:
                papers[p["id"]]["ai"] = p["ai"]
    (OUT_DIR / "ai_pending.json").write_text(json.dumps({"batches": pending}))

    vecs, similar = {}, set()
    if "embed" in only:
        vecs = embed.vectors(final + untagged + extras, embed.load_previous(args.previous), log)
        similar = {p["id"] for p in tagging.by_similarity(untagged, final, vecs, tagger.section_of)}
    relevant = lambda p: (s := (p.get("ai") or {}).get("score")) is not None and s >= KEEP_UNTAGGED_SCORE
    kept = [p for p in untagged if p["id"] in similar or relevant(p)]
    for p in kept:
        p["sections"] = ["bio"]
    kept_ids = {p["id"] for p in kept}
    (OUT_DIR / "ai_dropped.json").write_text(json.dumps({"papers": {
        p["id"]: {"score": p["ai"]["score"], "v": p["ai"].get("v"), **({"declined": True} if p["ai"].get("declined") else {})}
        for p in untagged if p["id"] not in kept_ids and p.get("ai")}}, separators=(",", ":")))
    log(f"  untagged journal papers: {len(kept)} of {len(untagged)} kept "
        f"({len(similar)} tagged by similarity, {sum(relevant(p) and p['id'] not in similar for p in kept)} by AI score)")
    final = sorted(final + [trim(p, tagger) for p in kept], key=lambda p: (p["date"], p.get("buzz_score", 0)), reverse=True)
    extras = [e for e in extras if e["id"] not in kept_ids]

    if "embed" in only:
        embed.write(final + extras, vecs, OUT_DIR)
        layout.run(OUT_DIR, log, previous=fetch_json(args.previous, "map.json"),
                   papers={p["id"]: p for p in final + extras}, name=ai.complete_json if "ai" in only else None)
    (OUT_DIR / "favorites.json").write_text(json.dumps({"papers": extras}, ensure_ascii=False, separators=(",", ":")))
    recent_cut = (today() - dt.timedelta(days=RECENT_DAYS)).isoformat()
    bio = [p for p in final if "bio" in p["sections"]]
    recent = [p for p in bio if p["date"] >= recent_cut]
    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sections": {k: v["label"] for k, v in sections_cfg.items()},
        "topics": [{"slug": t["slug"], "label": t["label"], "color": t["color"],
                    "section": t.get("section", "bio")} for t in topics_cfg],
        "people": [{"name": p["name"], "note": p.get("note", "")} for p in people_cfg],
        "papers": recent,
    }
    out = OUT_DIR / "papers.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    older = OUT_DIR / "older.json"
    older.write_text(json.dumps({"papers": [p for p in bio if p["date"] < recent_cut]},
                                ensure_ascii=False, separators=(",", ":")))
    ml_out = OUT_DIR / "ml.json"
    # Papers in both sections already ship in the bio files; ml.json holds the ML-only rest.
    ml_out.write_text(json.dumps({"papers": [p for p in final if p["sections"] == ["ml"]]},
                                 ensure_ascii=False, separators=(",", ":")))
    write_alerts(final, args.site_url)
    archived = archive.update(list(papers.values()), merge)

    counts = {t["slug"]: sum(t["slug"] in p["topics"] for p in final) for t in topics_cfg}
    log(f"wrote {len(final)} papers (bio {out.stat().st_size / 1e6:.1f} + {older.stat().st_size / 1e6:.1f} MB, "
        f"ml {ml_out.stat().st_size / 1e6:.1f} MB) · "
        f"PI hits {sum(bool(p['pis']) for p in final)} · buzzing {sum(p.get('buzz_score', 0) > 0 for p in final)}")
    log("  " + " · ".join(f"{k} {v}" for k, v in counts.items()))
    log("  archive: " + " · ".join(f"{m} {n}" for m, n in archived.items()))


if __name__ == "__main__":
    main()
