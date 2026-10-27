# biopapers

Lowkey, alphaXiv-inspired paper feed for the Gilbert lab, in two sections:

- **Bio** — new papers from the major journals + bioRxiv/medRxiv/arXiv q-bio, bucketed into
  interest tabs, with a PI watchlist and a "Buzz" tab for papers blowing up online.
- **ML / CS** — alphaXiv-style: every new arXiv ML paper (cs.LG/AI/CL/CV/RO/MA, stat.ML) plus
  Hugging Face Daily Papers, with a "Trending" tab (HF upvotes · HN · Bluesky), AI TL;DRs,
  code links + GitHub stars, and alphaXiv discussion links.

Save (☆), favorite (♥) and not-interested (✕) work across both, and so does the one search box:
it filters your feed and also finds any paper in the literature (title, DOI or arXiv ID).

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

**For You** is the home tab, with chips for Must read, ♥ Near favorites and each favorite label.
In the site itself (⚙︎): ranking weights for For You (recency, AI score, buzz, PIs, similarity to
your favorites and saves), recency half-life, tab order/visibility, per-topic weights,
export/import of your data, saved → BibTeX.

## AI scores, embeddings and the map

- **Must read** — every paper gets a 0–10 relevance score, a one-line TL;DR and a few words on
  why it matters, from Claude Haiku 5.5 reading the title + abstract against `config/profile.md`
  (plus your saved / upvoted papers, if `BIOPAPERS_GIST_ID` is set). Papers scoring ≥ 6 land in the
  Must read chip of For You, and the score feeds the For You ranking ("AI relevance" weight in ⚙︎). Uses
  `ANTHROPIC_API_KEY` (Message Batches API, half price, ~$2/month) or else `OPENROUTER_API_KEY`
  (`anthropic/claude-haiku-5.5`, direct calls, ~$3–4/month, at most 3,000 papers per build).
- **Take-home (on demand)** — a button on every paper asks Claude for its main message, key findings,
  approach and caveats, only when you click. It reads the open-access full text from Europe PMC when
  there is one (else the abstract) and calls OpenRouter straight from your browser with a key you paste
  in ⚙︎ (kept in that browser only — make a separate key with a credit limit). Pick Opus / Sonnet /
  Haiku 5.5 there; results are cached and synced, so each paper is paid for once.
- **Embeddings** — bge-small-en-v1.5 over title + abstract, computed on the Actions runner (free).
  They power "Similar to what I save / upvote" in For You and the **similar papers** button.
- **Graph** — its own view next to Bio | ML / CS: a UMAP layout of the embeddings, colored by topic,
  with ~30 named regions (k-means on the layout, named by Haiku; a region keeps its name while most of
  its papers stay). Each day's layout starts from yesterday's and is aligned back onto it, so the map
  stays put and new papers settle in next to their neighbours.
  ♥ favorites are labelled hubs; "around favorites" narrows it to them plus the 25 papers closest to
  each. Click any dot to see it and the 20 papers most like it.
- **♥ Favorites** — landmark papers, separate from saves. ♥ any card, or look up *any* paper in the
  Graph's search box (your feed, or the whole literature via OpenAlex, by title / DOI / arXiv ID).
  They sync through the gist, never age out, steer For You and the AI scorer, and give each label its
  own chip in For You (♥ Near favorites, ♥ <label>), which works like a topic defined by example papers.
  A favorite from outside the feed is embedded right in the browser (same model, via transformers.js —
  a one-time 34 MB download) and placed next to its nearest papers immediately. With Gist sync +
  `BIOPAPERS_GIST_ID` the daily build also picks favorites up (for the AI scorer and other devices);
  those are published in `data/favorites.json` like the rest of the site's data.

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
| `ANTHROPIC_API_KEY` | secret | AI scores + TL;DRs via Anthropic (batch, cheapest) |
| `OPENROUTER_API_KEY` | secret | …or via OpenRouter, if there's no Anthropic key (neither: scoring is skipped) |
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
  Journal papers no rule tags aren't simply dropped: they stay if their AI score is ≥ 3 (For You shows
  them from 4), and get a topic tab when most of their nearest tagged papers share one.
- **Journal coverage:** RSS feeds only list a journal's newest items (8 for every Nature journal), so
  busy journals are also pulled from OpenAlex by ISSN (complete, ~1–3 days late). Cell Press stays
  RSS-only: OpenAlex lists it under issue dates, weeks late.
