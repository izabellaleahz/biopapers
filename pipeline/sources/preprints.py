"""Preprint servers: bioRxiv + medRxiv (official API) and arXiv (export API)."""

import datetime as dt
import time

import feedparser

from ..common import client, flip_name, get, make_paper, today

RXIV_API = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json"
ARXIV_API = "https://export.arxiv.org/api/query"


def fetch_rxiv(server: str, days: int) -> list[dict]:
    """All bioRxiv/medRxiv preprints posted in the last N days (paged via cursor)."""
    end = today()
    start = end - dt.timedelta(days=days)
    papers, cursor = [], 0
    with client() as c:
        while True:
            resp = get(c, RXIV_API.format(server=server, start=start, end=end, cursor=cursor))
            data = resp.json()
            batch = data.get("collection", [])
            if not batch:
                break
            for r in batch:
                doi, version = r.get("doi"), r.get("version")
                p = make_paper(
                    doi=doi,
                    title=r.get("title"),
                    authors=[flip_name(a) for a in (r.get("authors") or "").split(";")],
                    abstract=r.get("abstract"),
                    date=r.get("date"),
                    venue="bioRxiv" if server == "biorxiv" else "medRxiv",
                    kind="preprint",
                    url=f"https://www.{server}.org/content/{doi}v{version}" if version else "",
                    category=r.get("category"),
                )
                if p:
                    papers.append(p)
            cursor += len(batch)
            if cursor >= int(data.get("messages", [{}])[0].get("total", 0)):
                break
    return papers


_arxiv_last_call = 0.0
ARXIV_SPACING = 3.5  # seconds between any two arXiv API calls in a build, across all fetches


def _arxiv_wait() -> None:
    global _arxiv_last_call
    time.sleep(max(0.0, _arxiv_last_call + ARXIV_SPACING - time.time()))
    _arxiv_last_call = time.time()


def fetch_arxiv(categories: list[str], days: int, page_size: int = 200) -> list[dict]:
    """Recent arXiv submissions in the given categories ('q-bio' = all q-bio.*).

    Pages newest-first until entries are older than the lookback window. arXiv asks
    for ~3s between API calls.
    """
    query = " OR ".join(f"cat:{c}.*" if "." not in c else f"cat:{c}" for c in categories)
    cutoff = (today() - dt.timedelta(days=days)).isoformat()
    papers, start = [], 0
    with client(timeout=60) as c:
        while True:
            _arxiv_wait()
            resp = get(c, ARXIV_API, params={
                "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
                "start": start, "max_results": page_size,
            })
            entries = feedparser.parse(resp.content).entries
            if not entries:
                break
            for e in entries:
                arxiv_id = e.id.rsplit("/abs/", 1)[-1].split("v")[0]
                p = make_paper(
                    doi=f"10.48550/arxiv.{arxiv_id}",
                    title=e.title,
                    authors=[a.get("name", "") for a in e.get("authors", [])],
                    abstract=e.get("summary"),
                    date=e.published[:10],
                    venue="arXiv",
                    kind="preprint",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    category=e.get("arxiv_primary_category", {}).get("term", ""),
                )
                if p:
                    p["arxiv_cats"] = [t["term"] for t in e.get("tags", [])]
                    papers.append(p)
            if entries[-1].published[:10] < cutoff:
                break
            start += page_size
    return [p for p in papers if p["date"] >= cutoff]
