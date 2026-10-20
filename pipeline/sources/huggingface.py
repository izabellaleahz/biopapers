"""Hugging Face Daily Papers — the curated, upvoted ML feed (the closest free thing to alphaXiv's
trending). Free JSON API, no key. ~60–100 papers per weekday."""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

from ..common import client, get, make_paper, today

API = "https://huggingface.co/api/daily_papers"


def _to_paper(item: dict) -> dict | None:
    p = item.get("paper") or {}
    arxiv_id = p.get("id")
    if not arxiv_id:
        return None
    org = (item.get("organization") or p.get("organization") or {}).get("fullname") or ""
    paper = make_paper(
        doi=f"10.48550/arxiv.{arxiv_id}",
        title=p.get("title") or item.get("title"),
        authors=[a.get("name", "") for a in p.get("authors", [])],
        abstract=p.get("summary") or item.get("summary"),
        date=(p.get("publishedAt") or item.get("publishedAt") or "")[:10] or None,
        venue="arXiv",
        kind="preprint",
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )
    if paper:
        paper["keywords"] = p.get("ai_keywords") or []
        paper["hf"] = {k: v for k, v in {
            "upvotes": p.get("upvotes", 0),
            "comments": item.get("numComments", 0),
            "tldr": p.get("ai_summary") or "",
            "github": p.get("githubRepo") or "",
            "stars": p.get("githubStars") or 0,
            "org": org,
            "project": p.get("projectPage") or "",
        }.items() if v}
    return paper


def fetch_daily(days: int) -> list[dict]:
    """Every Daily Papers entry for the last N days (one request per day; weekends are empty).
    Re-fetching recent days each run keeps upvote counts fresh."""
    dates = [(today() - dt.timedelta(days=i)).isoformat() for i in range(days)]
    with client() as c:
        def one(d):
            return get(c, API, params={"date": d, "limit": 100}).json()

        with ThreadPoolExecutor(6) as pool:
            items = [x for batch in pool.map(one, dates) for x in batch]
    return [p for x in items if (p := _to_paper(x))]
