"""Shared helpers: HTTP client, the paper record, and DOI/title/author/date normalization."""

import datetime as dt
import html
import json
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

USER_AGENT = "Mozilla/5.0 (compatible; biopapers/1.0; +https://github.com/shranchi0/biopapers)"

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#&]+")
TAG_RE = re.compile(r"<[^>]+>")


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                        transport=httpx.HTTPTransport(retries=2))  # retries connect errors/resets


def get(c: httpx.Client, url: str, tries: int = 4, **kw) -> httpx.Response:
    """GET with retries + backoff — for long paginated pulls where one dropped page shouldn't sink the source.
    Rate limits (429) and overload (503) get a real pause: the server's Retry-After, else 20s, 40s, 60s."""
    for attempt in range(tries):
        try:
            resp = c.get(url, **kw)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as ex:
            if attempt == tries - 1:
                raise
            status = ex.response.status_code if isinstance(ex, httpx.HTTPStatusError) else None
            if status in (429, 503):
                retry_after = ex.response.headers.get("retry-after", "")
                time.sleep(min(120, int(retry_after)) if retry_after.isdigit() else 20 * (attempt + 1))
            else:
                time.sleep(3 * (attempt + 1))


def openalex_params() -> dict:
    """OpenAlex polite-pool / key params, only if configured via env."""
    params = {}
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    return params


def today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


# ---- normalization ----------------------------------------------------------

def clean_text(s: str | None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", s))).strip()


def strip_label(abstract: str) -> str:
    """Drop a leading 'Abstract' / 'Summary' heading that some feeds include in the text."""
    return re.sub(r"^(Abstract|ABSTRACT|Summary|SUMMARY)[:.]?\s+(?=[A-Z0-9])", "", abstract)


def find_doi(*texts: str | None) -> str | None:
    """First DOI found in any of the given strings, normalized to lowercase."""
    for t in texts:
        if t and (m := DOI_RE.search(t)):
            return norm_doi(m.group(0))
    return None


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
    doi = re.sub(r"v\d+$", "", doi) if doi.startswith("10.1101/") else doi  # bioRxiv version suffix
    return doi.rstrip(".,;)") or None


def title_key(title: str) -> str:
    """Loose key for matching a preprint to its journal version."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())[:120]


def parse_date(s: str | None) -> str | None:
    """Best-effort -> YYYY-MM-DD."""
    if not s:
        return None
    s = s.strip()
    if m := re.match(r"(\d{4}-\d{2}-\d{2})", s):
        return m.group(1)
    try:
        return parsedate_to_datetime(s).date().isoformat()  # RFC 822, as in RSS
    except (TypeError, ValueError):
        return None


def flip_name(name: str) -> str:
    """'Uhler, C.' -> 'C. Uhler'; leaves 'Caroline Uhler' alone."""
    name = name.strip()
    if "," in name:
        last, first = (p.strip() for p in name.split(",", 1))
        return f"{first} {last}".strip()
    return name


def make_paper(*, doi: str | None, title: str, authors: list[str], abstract: str, date: str | None,
               venue: str, kind: str, url: str, category: str = "", **extra) -> dict | None:
    """The one record shape every source returns. None if it lacks the essentials."""
    title = clean_text(title)
    doi = norm_doi(doi)
    if not title or not (doi or url):
        return None
    return {
        "id": doi or url,
        "doi": doi,
        "title": title,
        "authors": [a for a in (clean_text(a) for a in authors) if a],
        "abstract": strip_label(clean_text(abstract)),
        "date": date or today().isoformat(),
        "venue": venue,
        "kind": kind,  # "preprint" | "article"
        "url": url or f"https://doi.org/{doi}",
        "category": (category or "").strip().lower(),
        **extra,
    }


GIST_FILE = "biopapers-state.json"


def read_gist_state(gist_id: str | None) -> dict:
    """The site's synced state (saves, votes, favorites...) from the reader's sync gist, or {}.
    Secret gists are readable by ID without auth."""
    if not gist_id:
        return {}
    with client() as c:
        f = get(c, f"https://api.github.com/gists/{gist_id}").json()["files"].get(GIST_FILE)
        if not f:
            return {}
        return json.loads(get(c, f["raw_url"]).text if f.get("truncated") else f["content"])
