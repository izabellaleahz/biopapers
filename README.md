# biopapers

Lowkey, alphaXiv-inspired paper feed for the Gilbert lab, in two sections:

- **Bio** — new papers from the major journals + bioRxiv/medRxiv/arXiv q-bio, bucketed into
  interest tabs, with a PI watchlist and a "Buzz" tab for papers blowing up online.
- **ML / CS** — alphaXiv-style: every new arXiv ML paper (cs.LG/AI/CL/CV/RO/MA, stat.ML) plus
  Hugging Face Daily Papers, with a "Trending" tab (HF upvotes · HN · Bluesky), AI TL;DRs,
  code links + GitHub stars, and alphaXiv discussion links.

Saves, upvotes and ranking settings work across both.

Static site — a daily GitHub Action fetches everything, writes JSON, and deploys to
GitHub Pages. No server, no database. Your saves/upvotes/settings live in your browser.

## How it works

```
GitHub Actions (daily, 12:00 UTC)                 GitHub Pages
python -m pipeline.build                          site/index.html + app.js
  fetch:  bioRxiv · medRxiv · arXiv (q-bio + ML)    Bio | ML / CS sections
          Hugging Face Daily Papers                 tabs · ranking · save / upvote / hide
          ~35 journal RSS feeds                      settings + data in localStorage
          OpenAlex (ISSN journals, PI watchlist,
                    niche-topic literature search)
  buzz:   Hacker News · Bluesky · ScienceDaily
  merge with yesterday's data → dedup → tag
  → site/data/papers.json, older.json, ml.json ─▶  site/alerts.xml (RSS of PI + buzz hits)
```

## Customize

All in `config/` — edit, push, and the next build picks it up (and re-tags everything already collected).

| File | What it controls |
|------|------------------|
| `config/topics.yaml` | Sections + interest tabs: keywords, journals, preprint categories, OpenAlex searches |
| `config/people.yaml` | PI watchlist (Uhler, Hacohen, Spranger) — name + OpenAlex ID |
| `config/sources.yaml` | Journal RSS feeds, OpenAlex-by-ISSN journals, ML arXiv categories + HF, buzz sources |

In the site itself (⚙︎): ranking weights for **For You** (recency, buzz, PIs, similarity to
what you save/upvote, your upvotes), recency half-life, tab order/visibility, per-topic weights,
export/import of your data, saved → BibTeX.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pipeline.build --days 30          # first run / backfill (~10–15 min)
.venv/bin/python -m pipeline.build                    # daily incremental (last 3 days, merged)
.venv/bin/python -m pipeline.build --only rss,buzz    # quick run with a subset of sources

python3 -m http.server -d site 8000                   # open http://127.0.0.1:8000
```

## Deploy to GitHub Pages

1. Repo **Settings → Pages → Source: GitHub Actions**.
   (Pages on a *private* repo needs a paid GitHub plan; the site itself is public either way.)
2. Push to `main`, or run **Actions → daily build → Run workflow** with `days = 30` for the first backfill.
3. Site: `https://<owner>.github.io/biopapers/` · alerts: `.../alerts.xml`.

Each run reads the previous build from the live site, so history accumulates without
committing data. Optional repo secrets/variables:

| Name | Kind | Why |
|------|------|-----|
| `BSKY_HANDLE`, `BSKY_APP_PASSWORD` | secrets | Authenticated Bluesky search — much better buzz coverage than anonymous ([make an app password](https://bsky.app/settings/app-passwords)) |
| `OPENALEX_MAILTO` | variable | Contact email for OpenAlex's polite pool |
| `OPENALEX_API_KEY` | secret | If OpenAlex rate-limits anonymous use |

## Known limits

- **Votes/saves are per-browser** (localStorage). Export JSON to back up or move devices.
  Shared lab votes would need a small backend (e.g. Supabase) — see `CONSIDERATIONS.md`.
- **No X/Twitter** — its API isn't free. Buzz uses HN + Bluesky + press coverage instead.
- **Abstracts:** Nature/Science/Cell RSS give short summaries; we backfill from OpenAlex,
  which often has no abstract for Elsevier/Springer papers.
- **Window:** 30 days for the general feed, 180 days for watched PIs. The ML arXiv firehose
  (~1,000 papers/day) is kept 3 days; after that only papers with HF upvotes or buzz stay.
  Saved papers keep a snapshot, so they never age out.
- **alphaXiv** has no public API — we link to each paper's alphaXiv page rather than
  pulling its votes/comments. "Trending" uses Hugging Face Daily Papers upvotes instead.
- **Keyword tagging** is rule-based; tune `config/topics.yaml` when a tab is too loose or too strict.
