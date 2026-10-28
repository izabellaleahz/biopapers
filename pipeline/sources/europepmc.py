"""Europe PMC: full abstracts (from PubMed) for journal papers whose feeds only give a teaser.

Nature, Cell and Science feeds often carry a one-line summary or nothing at all, and OpenAlex has
no abstracts for many paywalled journals. PubMed indexes these papers with the full abstract a few
days after they're out, and Europe PMC lets us look up many DOIs per request (no key needed).
"""

from ..common import client, clean_text, get, norm_doi

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BATCH = 50
# PubMed publication types that aren't research (or a review) — dropped from the feed.
NOT_RESEARCH = {"editorial", "news", "published erratum", "retraction of publication", "biography",
                "portrait", "interview", "congress", "historical article"}


def lookup_dois(dois: list[str]) -> dict[str, dict]:
    """{doi: {"abstract": str, "research": bool}} for the DOIs Europe PMC knows."""
    found = {}
    with client(timeout=60) as c:
        for i in range(0, len(dois), BATCH):
            chunk = dois[i:i + BATCH]
            query = " OR ".join(f'DOI:"{d}"' for d in chunk)
            data = get(c, API, params={"query": query, "resultType": "core", "format": "json",
                                       "pageSize": 2 * BATCH}).json()
            for r in data.get("resultList", {}).get("result", []):
                doi = norm_doi(r.get("doi"))
                if not doi:
                    continue
                types = {t.lower() for t in (r.get("pubTypeList") or {}).get("pubType", [])}
                found[doi] = {"abstract": clean_text(r.get("abstractText")),
                              "research": not (types & NOT_RESEARCH)}
    return found
