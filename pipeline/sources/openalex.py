"""OpenAlex: works by filter (ISSN / author), and batch lookup by DOI for enrichment."""

import datetime as dt

from ..common import client, make_paper, norm_doi, openalex_params, today

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


def lookup_dois(dois: list[str]) -> dict[str, dict]:
    """{doi: paper} for every DOI OpenAlex knows. Batched 50 per request."""
    found = {}
    with client() as c:
        for i in range(0, len(dois), 50):
            chunk = dois[i : i + 50]
            resp = c.get(API, params={
                "filter": "doi:" + "|".join(chunk), "select": SELECT, "per-page": 50,
                **openalex_params(),
            })
            resp.raise_for_status()
            for w in resp.json()["results"]:
                if (p := to_paper(w)) and (doi := norm_doi(w.get("doi"))):
                    found[doi] = p
    return found
