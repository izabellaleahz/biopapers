"""Permanent archive: every paper the build sees, untruncated, in monthly gzipped JSONL files.

The site only keeps a rolling window; the archive keeps everything. Papers are filed by the
month they were first seen (`added`), and each build rewrites only the current and previous
month — the only ones a paper in the window can belong to. In CI those two files are pulled
from and pushed back to the `archive` GitHub release; older months are never touched.

    gh release download archive -R izabellaleahz/biopapers -p 'papers-*' -D archive   # get it all

    from pipeline.archive import load
    papers = load()  # every paper ever seen, one record per paper
"""

import datetime as dt
import gzip
import json
from pathlib import Path

from .common import ROOT, today

ARCHIVE_DIR = ROOT / "archive"


def month_path(month: str, base: Path = ARCHIVE_DIR) -> Path:
    return base / f"papers-{month}.jsonl.gz"


def live_months() -> list[str]:
    """The months a build may write: this one and the last."""
    first = today().replace(day=1)
    return [first.strftime("%Y-%m"), (first - dt.timedelta(days=1)).strftime("%Y-%m")]


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def update(papers: list[dict], merge) -> dict[str, int]:
    """Upsert papers into the live months' files; `merge` is build.merge (keeps the longer
    abstract/author list, so a truncated copy from the site never clobbers a full one)."""
    ARCHIVE_DIR.mkdir(exist_ok=True)
    counts = {}
    for month in live_months():
        fresh = [p for p in papers if p.get("added", "")[:7] == month]
        path = month_path(month)
        if not fresh and not path.exists():
            continue
        merged = merge(read(path), fresh)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for pid in sorted(merged):
                f.write(json.dumps(merged[pid], ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp.replace(path)
        counts[month] = len(merged)
    return counts


def load(base: Path = ARCHIVE_DIR) -> list[dict]:
    """Every archived paper, newest record winning, with preprints that were later folded
    into their journal version dropped."""
    by_id: dict[str, dict] = {}
    for path in sorted(base.glob("papers-*.jsonl.gz")):
        for p in read(path):
            by_id[p["id"]] = p
    folded = {a for p in by_id.values() for a in p.get("aliases", [])}
    return [p for pid, p in by_id.items() if pid not in folded]
