# biopapers

Lowkey, alphaXiv-inspired bio paper feed for the Gilbert lab.

**MVP v0** — pulls recent [bioRxiv](https://api.biorxiv.org/) preprints, buckets them
into lab topics, and lets you upvote what's interesting. Server-rendered FastAPI + SQLite.
No auth, anonymous votes — see [`CONSIDERATIONS.md`](CONSIDERATIONS.md) for the roadmap.

## Topics
Cancer Biology · Immunology / T cell · Micro & Viral · Molecular Biology · Bio ML
(mapped onto bioRxiv categories in [`config.py`](config.py) — edit to taste.)

## Quickstart

```bash
pip install -r requirements.txt

# 1. Pull recent preprints into biopapers.db (default: last 30 days)
python ingest.py --days 14

# 2. Serve the feed
uvicorn main:app --reload --port 8000
# open http://127.0.0.1:8000
```

Re-run `python ingest.py` anytime to fetch newer papers — existing upvotes are preserved.
(For a real deployment, run it on a daily cron.)

## Files
| File | Role |
|------|------|
| `config.py` | Lab topic → bioRxiv category mapping |
| `db.py` | SQLite schema + queries |
| `ingest.py` | Pull from the bioRxiv API, upsert into the DB |
| `main.py` | FastAPI app: feed (filter by topic, sort by top/newest) + upvote |
| `templates/index.html` | Server-rendered feed UI |

## Known v0 limits (intentional)
- **bioRxiv only** — Nature/Cell/Science + medRxiv + aggregators come later.
- **Anonymous votes**, one column counter — no per-user votes or auth yet.
- **Category-based topics** — no LLM/embedding relevance scoring yet.

See [`CONSIDERATIONS.md`](CONSIDERATIONS.md) for where this is headed.
