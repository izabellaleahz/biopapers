"""OpenAlex: works by filter (ISSN / author), and batch lookup by DOI for enrichment."""

import datetime as dt

import re
from concurrent.futures import ThreadPoolExecutor

from ..common import client, get, make_paper, norm_doi, openalex_params, title_key, today

API = "https://api.openalex.org/works"
SELECT = ("doi,title,publication_date,type,authorships,abstract_inverted_index,"
          "primary_location,ids,primary_topic")
WORK_TYPES = "article|preprint|review|letter"  # skip datasets, software releases, errata


def _abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def to_paper(w: dict, venue: str | None = None) -> dict | None:
    if (w.get("title") or "").startswith("Abstract ") or w.get("type") in ("erratum", "paratext"):
        return None  # conference abstracts / corrections
    loc = w.get("primary_location") or {}
    source = (loc.get("source") or {}).get("display_name") or ""
    is_preprint = w.get("type") == "preprint" or "rxiv" in source.lower()
    for short in ("bioRxiv", "medRxiv", "arXiv"):  # "bioRxiv (Cold Spring Harbor Laboratory)"
        if short.lower() in source.lower():
            source = short
    return make_paper(
        doi=w.get("doi"),
        title=w.get("title"),
        authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
        abstract=_abstract(w.get("abstract_inverted_index")),
        date=w.get("publication_date"),
        venue=venue or source or "Other",
        kind="preprint" if is_preprint else "article",
        url=loc.get("landing_page_url") or "",
        # Its research area ("Life Sciences" / "Computer Science"): files a paper that buzzed in from
        # outside every tab under the right section, or under none.
        **{k: v for k, v in (("domain", ((w.get("primary_topic") or {}).get("domain") or {}).get("display_name")),
                             ("field", ((w.get("primary_topic") or {}).get("field") or {}).get("display_name"))) if v},
    )


def works_by_filter(flt: str, days: int, venue: str | None = None, max_pages: int = 10) -> list[dict]:
    since = (today() - dt.timedelta(days=days)).isoformat()
    papers = []
    with client() as c:
        # Plain page numbers: cursor paging past page 1 now needs an OpenAlex API key.
        for page in range(1, max_pages + 1):
            resp = c.get(API, params={
                "filter": f"{flt},from_publication_date:{since},type:{WORK_TYPES}", "select": SELECT,
                "per-page": 200, "page": page, **openalex_params(),
            })
            resp.raise_for_status()
            data = resp.json()
            papers += [p for w in data["results"] if (p := to_paper(w, venue))]
            if len(data["results"]) < 200:
                break
    return papers


def _work_dois(w: dict) -> set[str]:
    """Every DOI a work goes by. An arXiv paper's main DOI is often another registrant's (10.65215/...),
    with the arXiv DOI only among its locations."""
    dois = {norm_doi(w.get("doi"))}
    for loc in w.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        if m := re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", url):
            dois.add(f"10.48550/arxiv.{m.group(1)}")
        elif "doi.org/" in url:
            dois.add(norm_doi(url))
    return dois - {None}


def lookup_dois(dois: list[str]) -> dict[str, dict]:
    """{doi: paper} for every DOI OpenAlex knows (the paper keyed and id'd by the DOI asked for).
    Batched 50 per request."""
    found = {}
    with client() as c:
        for i in range(0, len(dois), 50):
            chunk = dois[i : i + 50]
            resp = c.get(API, params={
                "filter": "doi:" + "|".join(chunk), "select": SELECT + ",locations", "per-page": 50,
                **openalex_params(),
            })
            resp.raise_for_status()
            for w in resp.json()["results"]:
                if not (p := to_paper(w)):
                    continue
                for doi in _work_dois(w) & set(chunk):
                    found[doi] = {**p, "id": doi, "doi": doi,
                                  **({"venue": "arXiv", "kind": "preprint", "url": f"https://arxiv.org/abs/{doi[16:]}"}
                                     if doi.startswith("10.48550/arxiv.") else {})}
    return found


def dois_for_pmids(pmids: list[str]) -> dict[str, str]:
    """{PubMed id: DOI}. Batched 50 per request."""
    out = {}
    with client() as c:
        for i in range(0, len(pmids), 50):
            chunk = pmids[i : i + 50]
            data = get(c, API, params={"filter": "ids.pmid:" + "|".join(chunk), "select": "doi,ids",
                                       "per-page": 50, **openalex_params()}).json()
            for w in data["results"]:
                pmid = (w.get("ids") or {}).get("pmid", "").rsplit("/", 1)[-1]
                if pmid and (doi := norm_doi(w.get("doi"))):
                    out[pmid] = doi
    return out


def dois_for_titles(titles: list[str], workers: int = 8) -> dict[str, str]:
    """{page title: DOI} for link titles that are a paper's title, give or take the site's name
    ("<title> | Nature Medicine", "<title> - ScienceDirect"). Exact matches only: a news story about a
    paper has its own headline, so it can't be mistaken for the paper."""
    def one(c, title):
        parts = [p.strip() for p in re.split(r"\s[|–—-]\s", title) if p.strip()]
        guess = max(parts, key=len, default="")
        if len(key := title_key(guess)) < 25:
            return None
        try:
            data = get(c, API, tries=2, params={"search": guess, "per-page": 3, "select": "doi,title",
                                                **openalex_params()}).json()
        except Exception:
            return None
        for w in data.get("results", []):
            if title_key(w.get("title") or "") == key and (doi := norm_doi(w.get("doi"))):
                return doi
        return None

    with client() as c, ThreadPoolExecutor(workers) as pool:
        return {t: d for t, d in zip(titles, pool.map(lambda t: one(c, t), titles)) if d}
