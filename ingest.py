"""biopapers MVP v0 — bioRxiv ingestion.

Pulls recent bioRxiv preprints from the open bioRxiv API (no auth), keeps only
those in our lab topics, and upserts them into SQLite.

Usage:
    python ingest.py           # last 30 days
    python ingest.py --days 7
"""

import argparse
import datetime as dt

import httpx

import config
import db

# The details endpoint takes an explicit YYYY-MM-DD/YYYY-MM-DD date range and
# paginates via a cursor. It serves a fixed batch per call (currently 30).
API = "https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}/json"


def build_url(doi: str, version: str | None) -> str:
    v = f"v{version}" if version else ""
    return f"https://www.biorxiv.org/content/{doi}{v}"


def fetch_recent(days: int) -> list[dict]:
    """Fetch raw preprint records from the last N days, paging through the cursor."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    records: list[dict] = []
    cursor = 0
    with httpx.Client(timeout=30) as client:
        while True:
            url = API.format(start=start, end=end, cursor=cursor)
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("collection", [])
            if not batch:
                break
            records.extend(batch)
            total = int(data.get("messages", [{}])[0].get("total", 0))
            cursor += len(batch)
            if cursor >= total:
                break
    return records


def ingest(days: int) -> tuple[int, int]:
    """Fetch + store. Returns (total_fetched, kept_in_topics)."""
    db.init_db()
    records = fetch_recent(days)
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    kept = 0
    with db.connect() as conn:
        for r in records:
            topic = config.topic_for_category(r.get("category", ""))
            if topic is None:
                continue  # not a lab topic — skip for v0
            paper = {
                "doi": r.get("doi"),
                "title": (r.get("title") or "").strip(),
                "authors": (r.get("authors") or "").strip(),
                "category": (r.get("category") or "").strip(),
                "topic": topic,
                "abstract": (r.get("abstract") or "").strip(),
                "date": r.get("date"),
                "url": build_url(r.get("doi", ""), r.get("version")),
                "fetched_at": fetched_at,
            }
            if not paper["doi"] or not paper["title"]:
                continue
            db.upsert_paper(conn, paper)
            kept += 1
    return len(records), kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest recent bioRxiv preprints for lab topics.")
    ap.add_argument("--days", type=int, default=30, help="How many recent days to pull (default 30).")
    args = ap.parse_args()

    print(f"Fetching last {args.days} days from bioRxiv...")
    total, kept = ingest(args.days)
    print(f"Fetched {total} preprints; kept {kept} in lab topics.")


if __name__ == "__main__":
    main()
