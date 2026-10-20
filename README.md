# biopapers

Lowkey, alphaXiv-inspired paper feed for the Gilbert lab, in two sections:

- **Bio** — new papers from the major journals + bioRxiv/medRxiv/arXiv q-bio, bucketed into
  interest tabs, with a PI watchlist and a "Buzz" tab for papers blowing up online.
- **ML / CS** — alphaXiv-style: every new arXiv ML paper (cs.LG/AI/CL/CV/RO/MA, stat.ML) plus
  Hugging Face Daily Papers, with a "Trending" tab (HF upvotes · HN · Bluesky), AI TL;DRs,
  code links + GitHub stars, and alphaXiv discussion links.

Saves, upvotes and ranking settings work across both.

Static site — a daily GitHub Action fetches everything, writes JSON, and deploys to
GitHub Pages. No server, no database. Your saves/upvotes/settings live in your browser, and can
sync across devices through a secret GitHub Gist (⚙︎ → Sync across devices).

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
  embed (bge-small, local) → UMAP map + nearest neighbours
  AI score + TL;DR per paper (Claude Haiku 5.5, batch API)
  → site/data/papers.json, older.json, ml.json ─▶  site/alerts.xml (RSS of PI + buzz hits)
  → archive/papers-YYYY-MM.jsonl.gz ─▶ "archive" GitHub release (permanent, never pruned)
```

## Archive

The site shows a rolling window; the archive keeps **every paper the build has ever seen** —
including off-topic ones and the full ML firehose — with untruncated abstracts and author lists,
one gzipped JSONL file per month (filed by first-seen date). It lives as assets on the repo's
[`archive` release](https://github.com/izabellaleahz/biopapers/releases/tag/archive); each build
rewrites only the current and previous month.

```bash
gh release download archive -R izabellaleahz/biopapers -p 'papers-*' -D archive
.venv/bin/python -c "from pipeline.archive import load; print(len(load()))"
```

## Customize

All in `config/` — edit, push, and the next build picks it up (and re-tags everything already collected).

| File | What it controls |
|------|------------------|
| `config/topics.yaml` | Sections + interest tabs: keywords, journals, preprint categories, OpenAlex searches |
| `config/people.yaml` | PI watchlist (Uhler, Hacohen, Spranger) — name + OpenAlex ID |
| `config/sources.yaml` | Journal RSS feeds, OpenAlex-by-ISSN journals, ML arXiv categories + HF, buzz sources |
| `config/profile.md` | Who the AI scorer is reading for — plain prose; edit it to change what counts as a must-read |

In the site itself (⚙︎): ranking weights for **For You** (recency, buzz, PIs, similarity to
what you save/upvote, your upvotes), recency half-life, tab order/visibility, per-topic weights,
export/import of your data, saved → BibTeX.

## AI scores, embeddings and the map

- **Must read** — every paper gets a 0–10 relevance score, a one-line TL;DR and a few words on
  why it matters, from Claude Haiku 5.5 reading the title + abstract against `config/profile.md`
  (plus your saved / upvoted papers, if `BIOPAPERS_GIST_ID` is set). Papers scoring ≥ 7 land in the
  Must read tab, and the score feeds the For You ranking ("AI relevance" weight in ⚙︎). Runs through
  the Message Batches API at half price — about $2/month. A batch that isn't done after 20 minutes is
  collected by the next day's build.
- **Embeddings** — bge-small-en-v1.5 over title + abstract, computed on the Actions runner (free).
  They power "Similar to what I save / upvote" in For You and the **similar papers** button.
- **Map** — a UMAP layout of the embeddings, colored by topic. ♥ marks what you saved or upvoted,
  with lines to their nearest neighbours; click any dot to see it and the 20 papers most like it.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pipeline.build --days 30          # first run / backfill (~10–15 min)
.venv/bin/python -m pipeline.build                    # daily incremental (last 3 days, merged)
.venv/bin/python -m pipeline.build --only rss,buzz    # quick run with a subset of sources (skips embed + ai)

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
| `ANTHROPIC_API_KEY` | secret | AI scores + TL;DRs (skipped without it; everything else still runs) |
| `BIOPAPERS_GIST_ID` | variable | Your sync gist's ID (from ⚙︎ → Sync) — lets the scorer learn from what you save |

## Known limits

- **Votes/saves** live in localStorage; turn on Gist sync to carry them across browsers (needs a
  GitHub token with only the `gist` scope, pasted once per device). Shared lab votes would need a
  small backend (e.g. Supabase) — see `CONSIDERATIONS.md`.
- **No X/Twitter** — its API isn't free. Buzz uses HN + Bluesky + press coverage instead.
- **Abstracts:** Nature/Science/Cell RSS give short summaries; we backfill from OpenAlex,
  which often has no abstract for Elsevier/Springer papers.
- **Window:** 30 days for the general feed, 180 days for watched PIs (the archive keeps everything). The ML arXiv firehose
  (~1,000 papers/day) is kept 3 days; after that only papers with HF upvotes or buzz stay.
  Saved papers keep a snapshot, so they never age out.
- **alphaXiv** has no public API — we link to each paper's alphaXiv page rather than
  pulling its votes/comments. "Trending" uses Hugging Face Daily Papers upvotes instead.
- **Keyword tagging** is rule-based; tune `config/topics.yaml` when a tab is too loose or too strict.
