"""Journals: RSS/Atom feeds, plus OpenAlex-by-ISSN for journals without usable feeds."""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import feedparser

from ..common import client, find_doi, make_paper, parse_date
from . import openalex


def _entry_date(e) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        if t := e.get(key):
            return dt.date(*t[:3]).isoformat()
    return parse_date(e.get("prism_publicationdate") or e.get("dc_date"))


def _entry_authors(e) -> list[str]:
    names = [a.get("name", "") for a in e.get("authors", []) if a.get("name")]
    # Some feeds pack everyone into one comma-separated dc:creator string.
    if len(names) == 1 and names[0].count(",") >= 2:
        names = [n.strip() for n in names[0].split(",")]
    return names


def fetch_feed(venue: str, url: str) -> list[dict]:
    with client() as c:
        resp = c.get(url)
        resp.raise_for_status()
    papers = []
    for e in feedparser.parse(resp.content).entries:
        doi = find_doi(e.get("prism_doi"), e.get("dc_identifier"), e.get("id"), e.get("link"))
        if not doi and "elifesciences.org/articles/" in e.get("link", ""):
            doi = f"10.7554/elife.{e.link.rstrip('/').rsplit('/', 1)[-1]}"
        content = e.get("content", [{}])[0].get("value", "") if e.get("content") else ""
        p = make_paper(
            doi=doi,
            title=e.get("title"),
            authors=_entry_authors(e),
            abstract=max(e.get("summary", ""), content, key=len),
            date=_entry_date(e),
            venue=venue,
            kind="article",
            url=e.get("link", ""),
        )
        if p:
            papers.append(p)
    return papers


def fetch_all_feeds(feeds: dict[str, str], log) -> list[dict]:
    def one(item):
        venue, url = item
        try:
            got = fetch_feed(venue, url)
            log(f"  rss  {venue}: {len(got)}")
            return got
        except Exception as ex:  # one bad feed shouldn't sink the build
            log(f"  rss  {venue}: FAILED ({ex.__class__.__name__}: {ex})")
            return []

    with ThreadPoolExecutor(8) as pool:
        return [p for batch in pool.map(one, feeds.items()) for p in batch]


def fetch_issn_journals(journals: dict[str, str], days: int, log) -> list[dict]:
    papers = []
    for venue, issn in journals.items():
        try:
            got = openalex.works_by_filter(f"primary_location.source.issn:{issn}", days, venue=venue)
            log(f"  oa   {venue}: {len(got)}")
            papers += got
        except Exception as ex:
            log(f"  oa   {venue}: FAILED ({ex.__class__.__name__}: {ex})")
    return papers
