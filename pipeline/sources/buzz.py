"""Attention signals — "is this paper blowing up?"

  - X/Twitter (paid, pay-per-use API) — posts linking to papers above a like threshold, one query for
    bio venues and one for CS (arXiv etc.), each capped at a post budget. Needs X_BEARER_TOKEN.
  - Hacker News (Algolia search API, free, no key) — stories linking to papers
  - ScienceDaily top stories — press coverage, resolved to the paper's DOI.

X returns {doi: {"x": {account: engagement}, "x_url": ...}}: who shared it, not how many posts, so
one account posting a link five times counts once. Accounts are hashed (the build keeps
them across runs to count distinct sharers, and that file is public). Bots and aggregators — a
handle with "bot"/"arxiv" in it, or an account that shared many different papers in one run — are
dropped: they post everything, so they say nothing about what's popular.
Curated accounts (config/sources.yaml `buzz.x.people`: scientists whose picks are worth following) are
also read whatever their likes, reposts and quotes included, and are never treated as aggregators;
their handles ship (`x_by`) and each counts double (build.buzz_score).
HN and press return {doi: {...signal...}}; build.py merges all of them onto papers.
"""

import hashlib
from collections import Counter
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import feedparser

from ..common import client, find_doi, norm_doi
from . import openalex

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
    return find_doi(url)  # bioRxiv/medRxiv URL suffixes (v1.full.pdf) are cut by norm_doi


def account_key(network: str, account_id: str) -> str:
    return hashlib.sha1(f"{network}:{account_id}".encode()).hexdigest()[:10]


def people(shares: list[tuple], max_papers: int, curated: frozenset = frozenset()) -> list[tuple]:
    """(doi, account id, handle, engagement, ...) shares without bots and aggregators (curated handles,
    lowercase, are kept however much they post)."""
    papers_of: dict[str, set] = {}
    for doi, acct, *_ in shares:
        papers_of.setdefault(acct, set()).add(doi)
    return [s for s in shares if s[2].lower() in curated
            or (not BOT_HANDLE.search(s[2]) and len(papers_of[s[1]]) <= max_papers)]


def by_account(shares: list[tuple], network: str, max_papers: int, curated: frozenset = frozenset()) -> dict[str, dict]:
    """Shares -> {doi: {network: {account: engagement}, network_by: [curated handles]}}, people only.
    An account's engagement on a paper is its best post about it."""
    out: dict[str, dict] = {}
    for doi, acct, handle, eng, *_ in people(shares, max_papers, curated):
        sig = out.setdefault(doi, {network: {}})
        key = account_key(network, acct)
        sig[network][key] = max(sig[network].get(key, 0), eng)
        if handle.lower() in curated and handle not in sig.setdefault(f"{network}_by", []):
            sig[f"{network}_by"].append(handle)
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
                    cur["hn_at"] = (h.get("created_at") or "")[:10]
    return out


X_SEARCH = "https://api.x.com/2/tweets/search/recent"
X_COUNTS = "https://api.x.com/2/tweets/counts/recent"
X_TIME = "%Y-%m-%dT%H:%M:%SZ"


def _x_query(terms: list[str], min_likes: int, reposts: bool = False) -> str:
    # Plain `url:"<domain>"` misses most publisher links in X's index (8 bioRxiv posts a week, against
    # 298 for the bare word), so the terms mix url: tokens and keywords — see config/sources.yaml.
    q = "(" + " OR ".join(terms) + ") has:links" + ("" if reposts else " -is:retweet")
    return q + (f" min_likes:{min_likes}" if min_likes else "")


X_QUERY_MAX = 512


def people_queries(cfg: dict) -> dict[str, dict]:
    """`buzz.x.people` ({"max_posts", "min_likes", "accounts": {group: [handles]}}) -> `from:` queries,
    each group split evenly into as few as fit X's query length. Reposts count: a curated account
    reposting a paper is its pick."""
    out = {}
    for group, handles in (cfg.get("accounts") or {}).items():
        k = 1
        while True:
            chunks = [handles[i::k] for i in range(k)]
            if all(len(_x_query([f"from:{h}" for h in c], 9999, True)) <= X_QUERY_MAX for c in chunks):
                break
            k += 1
        for i, c in enumerate(chunks, 1):
            out[f"{group} people" + (f" {i}" if k > 1 else "")] = {
                "terms": [f"from:{h}" for h in c], "reposts": True,
                "max_posts": cfg.get("max_posts", 150), "min_likes": cfg.get("min_likes", [0])}
    return out


def curated_handles(cfg: dict) -> frozenset:
    return frozenset(h.lower() for hs in (cfg.get("accounts") or {}).values() for h in hs)


NOT_PAPERS = re.compile(r"(^|\.)(x|twitter|youtube|youtu|instagram|linkedin|facebook|tiktok|github|bsky|"
                        r"t|bit|google|wikipedia|reddit|substack|medium)\.[a-z]+$")
MAX_TITLE_LOOKUPS = 400  # per pull (OpenAlex, free): links that carry no DOI, matched by their page title


def _paper_ref(link: dict) -> str | None:
    """A post's link -> a DOI, or "pmid:<id>" / "title:<page title>" to look up. expanded_url is the link
    as posted; unwound_url follows redirects (doi.org -> the paper, but a ScienceDirect link -> its login
    page), so both are tried."""
    urls = [link.get("expanded_url") or "", link.get("unwound_url") or ""]
    for u in urls:
        if doi := doi_from_url(u):
            return doi
    for u in urls:
        if m := re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})", u):
            return f"pmid:{m.group(1)}"
    host = (re.match(r"https?://([^/]+)", urls[1] or urls[0]) or [None, ""])[1].lower()
    if (title := (link.get("title") or "").strip()) and host and not NOT_PAPERS.search(host):
        return f"title:{title}"
    return None


def _resolve(shares: list[tuple], log) -> list[tuple]:
    """Shares whose ref is a PubMed id or page title -> their DOI (via OpenAlex); unresolved ones dropped."""
    pmids = sorted({r[5:] for r, *_ in shares if r.startswith("pmid:")})
    titles = [t for t, _ in Counter(r[6:] for r, *_ in shares if r.startswith("title:")).most_common(MAX_TITLE_LOOKUPS)]
    try:
        by_pmid = openalex.dois_for_pmids(pmids) if pmids else {}
        by_title = openalex.dois_for_titles(titles) if titles else {}
    except Exception as ex:
        log(f"    x: resolving links failed ({ex.__class__.__name__}: {ex})")
        by_pmid, by_title = {}, {}
    log(f"    x: links without a DOI: {len(by_pmid)}/{len(pmids)} PubMed ids, {len(by_title)}/{len(titles)} page titles matched a paper")
    ref = lambda r: by_pmid.get(r[5:]) if r.startswith("pmid:") else by_title.get(r[6:]) if r.startswith("title:") else r
    return [(d, *rest) for r, *rest in shares if (d := ref(r))]


def x(start: float, end: float, queries: dict[str, dict], log=print, curated: frozenset = frozenset()) -> dict[str, dict]:
    """Every X post between `start` and `end` (epoch seconds) linking to papers, above a like threshold.
    `queries` is {name: {"terms": [...], "max_posts": n, "min_likes": [n, ...]}}. X bills per post read,
    so each query first counts its posts (counts are billed per request, not per post) at each threshold
    and reads at the lowest one that fits `max_posts`: the whole popular set, not a sample. (Most-liked-
    first ordering would be simpler, but X stops it after ~150 posts.) A query with "reposts" also takes
    the links in the posts its posts repost or quote (an expansion, which X may bill as reads too)."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        log("    x: no X_BEARER_TOKEN, skipped")
        return {}
    window = {"start_time": time.strftime(X_TIME, time.gmtime(start)), "end_time": time.strftime(X_TIME, time.gmtime(end))}
    shares = []

    def call(c, url, name, params):
        resp = c.get(url, params=params)
        if resp.is_error:
            log(f"    x {name}: HTTP {resp.status_code} {resp.text[:300]}")
        resp.raise_for_status()
        return resp.json()

    with client(timeout=30) as c:
        c.headers["Authorization"] = f"Bearer {token}"
        for name, q in queries.items():
            budget = q.get("max_posts", 300)
            counts = {}
            for likes in q.get("min_likes", [0]):
                meta = call(c, X_COUNTS, name, {"query": _x_query(q["terms"], likes, q.get("reposts")),
                                                "granularity": "day", **window})
                counts[likes] = meta.get("meta", {}).get("total_tweet_count", 0)
                if counts[likes] <= budget:
                    break
            if counts[likes] > budget:
                log(f"    x {name}: {counts[likes]} posts even at {likes}+ likes — reading the newest {budget}")
            read, token_next = 0, None
            while read < budget:
                params = {"query": _x_query(q["terms"], likes, q.get("reposts")), **window, "sort_order": "recency",
                          "max_results": max(10, min(100, budget - read)),
                          "tweet.fields": "public_metrics,entities,author_id" + (",referenced_tweets" if q.get("reposts") else ""),
                          "expansions": "author_id" + (",referenced_tweets.id" if q.get("reposts") else ""),
                          "user.fields": "username"}
                if token_next:
                    params["next_token"] = token_next
                data = call(c, X_SEARCH, name, params)
                users = {u["id"]: u.get("username", "") for u in (data.get("includes") or {}).get("users", [])}
                quoted = {r["id"]: r for r in (data.get("includes") or {}).get("tweets", [])}
                posts = data.get("data") or []
                read += len(posts)
                for t in posts:
                    links = list((t.get("entities") or {}).get("urls", []))
                    m = t.get("public_metrics") or {}
                    for r in t.get("referenced_tweets") or []:
                        if (orig := quoted.get(r["id"])) and r["type"] in ("retweeted", "quoted"):
                            links += (orig.get("entities") or {}).get("urls", [])
                            if r["type"] == "retweeted":  # a repost's own counts are empty: credit the original's
                                m = orig.get("public_metrics") or m
                    eng = m.get("like_count", 0) + m.get("retweet_count", 0) + m.get("quote_count", 0)
                    handle = users.get(t["author_id"], "")
                    for ref in {r for u in links if (r := _paper_ref(u))}:
                        shares.append((ref, t["author_id"], handle, eng,
                                       f"https://x.com/{handle or 'i'}/status/{t['id']}"))
                if not (token_next := (data.get("meta") or {}).get("next_token")) or not posts:
                    break
            tried = ", ".join(f"{n}+ likes: {v}" for n, v in counts.items())
            log(f"    x {name}: counted {tried} · read {read} posts at {likes}+ likes")
    shares = _resolve(shares, log)
    if curated:
        seen = {h.lower() for _, _, h, *_ in shares} & curated
        log(f"    x people: {len(seen)}/{len(curated)} curated accounts shared a paper"
            + (f" · none from {', '.join(sorted(curated - seen))}" if curated - seen else ""))
    # Two days of posts: an account sharing 7+ different papers is an aggregator (or a journal's own).
    out = by_account(shares, "x", 6, curated)
    for doi, _, _, eng, url in sorted(people(shares, 6, curated), key=lambda s: s[3]):
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
    day = time.strftime("%Y-%m-%d", time.gmtime())
    for doi, story in hits:
        out.setdefault(doi, {"news": [], "news_at": day})["news"].append(story)
    return out
