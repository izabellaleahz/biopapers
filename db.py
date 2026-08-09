"""biopapers MVP v0 — SQLite storage.

Barebones single-file DB. One `papers` table with a `votes` counter column.
(Per-user votes / a separate votes table is a v1 concern — see CONSIDERATIONS.md.)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "biopapers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doi         TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    authors     TEXT,
    category    TEXT,
    topic       TEXT,          -- our lab-topic slug (see config.TOPICS)
    abstract    TEXT,
    date        TEXT,          -- preprint date, YYYY-MM-DD
    url         TEXT,
    votes       INTEGER NOT NULL DEFAULT 0,
    fetched_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic);
CREATE INDEX IF NOT EXISTS idx_papers_date  ON papers(date);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    """Insert a paper, preserving existing votes if the DOI already exists."""
    conn.execute(
        """
        INSERT INTO papers (doi, title, authors, category, topic, abstract, date, url, fetched_at)
        VALUES (:doi, :title, :authors, :category, :topic, :abstract, :date, :url, :fetched_at)
        ON CONFLICT(doi) DO UPDATE SET
            title=excluded.title,
            authors=excluded.authors,
            category=excluded.category,
            topic=excluded.topic,
            abstract=excluded.abstract,
            date=excluded.date,
            url=excluded.url,
            fetched_at=excluded.fetched_at
        """,
        paper,
    )


def fetch_feed(topic: str | None = None, sort: str = "votes", limit: int = 100) -> list[sqlite3.Row]:
    """Return papers, optionally filtered to one topic, sorted by votes or date."""
    order = "votes DESC, date DESC" if sort == "votes" else "date DESC, votes DESC"
    sql = "SELECT * FROM papers WHERE topic IS NOT NULL"
    params: list = []
    if topic:
        sql += " AND topic = ?"
        params.append(topic)
    sql += f" ORDER BY {order} LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def counts_by_topic() -> dict[str, int]:
    """Return {topic_slug: paper_count}, plus 'all' for the total."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT topic, COUNT(*) n FROM papers WHERE topic IS NOT NULL GROUP BY topic"
        ).fetchall()
    counts = {r["topic"]: r["n"] for r in rows}
    counts["all"] = sum(counts.values())
    return counts


def add_vote(paper_id: int) -> int | None:
    """Increment a paper's vote count; return the new total (or None if not found)."""
    with connect() as conn:
        row = conn.execute(
            "UPDATE papers SET votes = votes + 1 WHERE id = ? RETURNING votes", (paper_id,)
        ).fetchone()
        return row["votes"] if row else None
