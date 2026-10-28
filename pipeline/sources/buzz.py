"""Attention signals — "is this paper blowing up?"

  - X/Twitter (paid, pay-per-use API) — the most-engaged posts linking to papers, one query for bio
    venues and one for CS (arXiv etc.), each capped at a post budget. Needs X_BEARER_TOKEN.
  - Hacker News (Algolia search API, free, no key) — stories linking to papers
  - ScienceDaily top stories — press coverage, resolved to the paper's DOI.

X returns {doi: {"x": {account: engagement}, "x_url": ...}}: who shared it, not how many posts, so
one account posting a link five times counts once. Accounts are hashed (the build keeps
them across runs to count distinct sharers, and that file is public). Bots and aggregators — a
handle with "bot"/"arxiv" in it, or an account that shared many different papers in one run — are
dropped: they post everything, so they say nothing about what's popular.
HN and press return {doi: {...signal...}}; build.py merges all of them onto papers.
"""

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import feedparser

from ..common import client, find_doi, norm_doi

PAPER_DOMAINS = ["biorxiv.org", "medrxiv.org", "nature.com", "science.org", "cell.com",
                 "arxiv.org", "doi.org", "pnas.org", "elifesciences.org",
                 "huggingface.co/papers", "alphaxiv.org"]
BOT_HANDLE = re.compile(r"(?<![a-z])bot(?![a-z])|arxiv|_daily|daily_?papers", re.I)


def doi_from_url(url: str) -> str | None:
    """Map a paper landing-page URL to its DOI where the URL makes that possible."""
    if not url:
        return None
    if m := re.search(r"(?:arxiv\.org/(?:abs|pdf|html)|huggingface\.co/papers|alphaxiv\.org/(?:abs|overview))"
                      r"/(\d{4}\.\d{4,5})", url):
        return f"10.48550/arxiv.{m.group(1)}"
    if m := re.search(r"nature\.com/articles/([a-z0-9.-]+)", url):
        return f"10.1038/{m.group(1).lower()}"
    if m := re.search(r"elifesciences\.org/(?:articles|reviewed-preprints)/(\d+)", url):
        return f"10.7554/elife.{m.group(1)}"
    doi = find_doi(url)
    if doi and doi.startswith("10.1101/"):  # bioRxiv/medRxiv URL suffixes
        doi = norm_doi(re.sub(r"(\.full(\.pdf)?|\.abstract|\.supplementary-material|\.article-.*)$", "", doi))
    return doi


def account_key(network: str, account_id: str) -> str:
    return hashlib.sha1(f"{network}:{account_id}".encode()).hexdigest()[:10]


def people(shares: list[tuple], max_papers: int) -> list[tuple]:
    """(doi, account id, handle, engagement, ...) shares without bots and aggregators."""
    papers_of: dict[str, set] = {}
    for doi, acct, *_ in shares:
        papers_of.setdefault(acct, set()).add(doi)
    return [s for s in shares if not BOT_HANDLE.search(s[2]) and len(papers_of[s[1]]) <= max_papers]


def by_account(shares: list[tuple], network: str, max_papers: int) -> dict[str, dict]:
    """Shares -> {doi: {network: {account: engagement}}}, people only. An account's engagement on a
    paper is its best post about it."""
    out: dict[str, dict] = {}
    for doi, acct, _, eng, *_ in people(shares, max_papers):
        accts = out.setdefault(doi, {network: {}})[network]
        key = account_key(network, acct)
        accts[key] = max(accts.get(key, 0), eng)
    return out


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


X_SEARCH = "https://api.x.com/2/tweets/search/recent"


def _x_query(domains: list[str], min_likes: int) -> str:
    q = "(" + " OR ".join(f'url:"{d}"' for d in domains) + ") -is:retweet"
    return q + (f" min_likes:{min_likes}" if min_likes else "")


def x(hours: float, queries: dict[str, dict], log=print) -> dict[str, dict]:
    """The most-engaged X posts of the last `hours` linking to papers. `queries` is
    {name: {"domains": [...], "max_posts": n, "min_likes": n}}: each is billed per post read, so each
    stops at its budget. Relevancy order puts the most-engaged posts first, so a budget cuts the tail."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        log("    x: no X_BEARER_TOKEN, skipped")
        return {}
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600))
    shares = []
    with client(timeout=30) as c:
        c.headers["Authorization"] = f"Bearer {token}"
        for name, q in queries.items():
            budget, read, token_next = q.get("max_posts", 300), 0, None
            min_likes = q.get("min_likes", 0)
            while read < budget:
                params = {"query": _x_query(q["domains"], min_likes), "start_time": start,
                          "max_results": max(10, min(100, budget - read)), "sort_order": "relevancy",
                          "tweet.fields": "public_metrics,entities,author_id",
                          "expansions": "author_id", "user.fields": "username"}
                if token_next:
                    params["next_token"] = token_next
                resp = c.get(X_SEARCH, params=params)
                if resp.status_code == 400 and min_likes and not read:
                    log(f"    x {name}: min_likes rejected ({resp.text[:200]}), retrying without")
                    min_likes = 0
                    continue
                if resp.status_code == 429:
                    log(f"    x {name}: rate limited, stopping at {read} posts")
                    break
                if resp.is_error:
                    log(f"    x {name}: HTTP {resp.status_code} {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                users = {u["id"]: u.get("username", "") for u in (data.get("includes") or {}).get("users", [])}
                posts = data.get("data") or []
                read += len(posts)
                for t in posts:
                    m = t.get("public_metrics") or {}
                    eng = m.get("like_count", 0) + m.get("retweet_count", 0) + m.get("quote_count", 0)
                    urls = [u.get("unwound_url") or u.get("expanded_url") or ""
                            for u in (t.get("entities") or {}).get("urls", [])]
                    for doi in {d for u in urls if (d := doi_from_url(u))}:
                        handle = users.get(t["author_id"], "")
                        shares.append((doi, t["author_id"], handle, eng,
                                       f"https://x.com/{handle or 'i'}/status/{t['id']}"))
                if not (token_next := (data.get("meta") or {}).get("next_token")) or not posts:
                    break
            log(f"    x {name}: {read} posts read")
    # One day of posts: an account sharing 5+ different papers is an aggregator.
    out = by_account(shares, "x", max_papers=4)
    for doi, _, _, eng, url in sorted(people(shares, 4), key=lambda s: s[3]):
        out[doi]["x_url"] = url  # the most-engaged post (sorted ascending, so the last one wins)
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
