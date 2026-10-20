"""Attention signals — "is this paper blowing up?"

X/Twitter has no usable free API, so we approximate with sources that do:
  - Hacker News (Algolia search API, free, no key) — stories linking to papers
  - Bluesky (where much of science Twitter moved) — posts sharing paper links.
    Unauthenticated search is thin; set BSKY_HANDLE + BSKY_APP_PASSWORD for more.
  - ScienceDaily top stories — press coverage, resolved to the paper's DOI.

Each fetcher returns {doi: {...signal...}}; build.py merges them onto papers.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import feedparser

from ..common import client, find_doi, norm_doi

PAPER_DOMAINS = ["biorxiv.org", "medrxiv.org", "nature.com", "science.org", "cell.com",
                 "arxiv.org", "doi.org", "pnas.org", "elifesciences.org"]


def doi_from_url(url: str) -> str | None:
    """Map a paper landing-page URL to its DOI where the URL makes that possible."""
    if not url:
        return None
    if m := re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", url):
        return f"10.48550/arxiv.{m.group(1)}"
    if m := re.search(r"nature\.com/articles/([a-z0-9.-]+)", url):
        return f"10.1038/{m.group(1).lower()}"
    if m := re.search(r"elifesciences\.org/(?:articles|reviewed-preprints)/(\d+)", url):
        return f"10.7554/elife.{m.group(1)}"
    doi = find_doi(url)
    if doi and doi.startswith("10.1101/"):  # bioRxiv/medRxiv URL suffixes
        doi = norm_doi(re.sub(r"(\.full(\.pdf)?|\.abstract|\.supplementary-material|\.article-.*)$", "", doi))
    return doi


def hackernews(days: int) -> dict[str, dict]:
    since = int(time.time()) - days * 86400
    out: dict[str, dict] = {}
    with client() as c:
        for domain in PAPER_DOMAINS:
            resp = c.get("https://hn.algolia.com/api/v1/search", params={
                "query": domain, "tags": "story", "hitsPerPage": 100,
                "numericFilters": f"created_at_i>{since}",
            })
            resp.raise_for_status()
            for h in resp.json()["hits"]:
                if domain not in (h.get("url") or "") or not (doi := doi_from_url(h["url"])):
                    continue
                cur = out.setdefault(doi, {"hn": 0, "hn_url": ""})
                if (h.get("points") or 0) > cur["hn"]:
                    cur["hn"] = h["points"]
                    cur["hn_url"] = f"https://news.ycombinator.com/item?id={h['objectID']}"
    return out


def _bsky_session(c) -> tuple[str, dict]:
    """(search base URL, auth headers) — authenticated if creds are set."""
    handle, pw = os.environ.get("BSKY_HANDLE"), os.environ.get("BSKY_APP_PASSWORD")
    if handle and pw:
        resp = c.post("https://bsky.social/xrpc/com.atproto.server.createSession",
                      json={"identifier": handle, "password": pw})
        resp.raise_for_status()
        return "https://bsky.social", {"Authorization": f"Bearer {resp.json()['accessJwt']}"}
    return "https://api.bsky.app", {}


def _post_links(post: dict) -> set[str]:
    rec = post.get("record") or {}
    embed = rec.get("embed") or {}
    ext = embed.get("external") or (embed.get("media") or {}).get("external") or {}
    links = {ext["uri"]} if ext.get("uri") else set()
    for facet in rec.get("facets") or []:
        links |= {f["uri"] for f in facet.get("features", []) if f.get("uri")}
    return links


def bluesky(days: int) -> dict[str, dict]:
    since = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - days * 86400))
    out: dict[str, dict] = {}
    with client(timeout=20) as c:
        base, headers = _bsky_session(c)
        pages = 5 if headers else 1  # unauthenticated search doesn't paginate
        for domain in PAPER_DOMAINS:
            cursor = None
            for _ in range(pages):
                params = {"q": domain, "since": since, "limit": 100, "sort": "top"}
                if cursor:
                    params["cursor"] = cursor
                try:
                    resp = c.get(f"{base}/xrpc/app.bsky.feed.searchPosts", params=params, headers=headers)
                    resp.raise_for_status()
                except Exception:
                    break  # flaky endpoint — take what we have for this domain
                data = resp.json()
                for post in data.get("posts", []):
                    for doi in {d for u in _post_links(post) if (d := doi_from_url(u))}:
                        cur = out.setdefault(doi, {"bsky": 0, "bsky_likes": 0})
                        cur["bsky"] += 1
                        cur["bsky_likes"] += (post.get("likeCount") or 0) + (post.get("repostCount") or 0)
                if not (cursor := data.get("cursor")):
                    break
    return out


SCIENCEDAILY_FEEDS = ["https://www.sciencedaily.com/rss/top/science.xml",
                      "https://www.sciencedaily.com/rss/top/health.xml"]


def sciencedaily() -> dict[str, dict]:
    """Press coverage: each story page cites the paper ("Journal Reference: ... DOI")."""
    with client() as c:
        entries = {e.link: e.title for url in SCIENCEDAILY_FEEDS
                   for e in feedparser.parse(c.get(url).content).entries}

        def resolve(item):
            link, title = item
            try:
                page = c.get(link).text
            except Exception:
                return None
            m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s\"<>]+)", page)
            return (norm_doi(m.group(1)), {"title": title, "url": link}) if m else None

        with ThreadPoolExecutor(8) as pool:
            hits = [h for h in pool.map(resolve, entries.items()) if h]
    out: dict[str, dict] = {}
    for doi, story in hits:
        out.setdefault(doi, {"news": []})["news"].append(story)
    return out
