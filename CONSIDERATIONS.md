# biopapers — Considerations & Design Notes

> A living scratchpad for what this thing should be. Lowkey, alphaXiv-inspired bio paper feed.
> Add notes freely — nothing here is a commitment yet.

---

## 0. One-liner / North Star

*What is this, in one sentence?*

> A personalized, crowdsourced feed of new + interesting biology papers, focused on Gilbert-lab-relevant topics.

### ✅ Decisions so far (2026-08-08)
- **Audience:** the **Gilbert lab** — shared internal feed. Needs light auth; crowdsourcing = lab upvotes, comments, journal-club queue.
- **Delivery:** **web app** — browsable feed UI with filters, votes, comments.
- **Filtering / scoring:** **LLM / embeddings** — classify topic + score relevance over title+abstract.

*Implications:* internal-only hosting + simple auth (SSO or shared login); "crowd" = lab members, so votes are high-trust and low-volume; LLM classifier means an embeddings/model pipeline in ingestion (cost + rate limits to watch).

### 🛠️ Built: MVP v0 (2026-08-08)
Barebones proof-of-concept shipped — FastAPI + SQLite, bioRxiv only. See `README.md`.
- `ingest.py` pulls recent bioRxiv preprints (open API, no auth), buckets by category → lab topic.
- `main.py` serves a server-rendered feed: filter by topic, sort top/newest, one-click upvote.
- Verified: 1,000 papers ingested over 14 days across all 5 topics; upvote + filter + sort working.

**Deliberately deferred to v1** (the interesting stuff): auth + per-user votes, comments / journal-club queue,
multi-source ingest (Nature/Cell/Science + medRxiv + aggregators), LLM/embedding relevance scoring, TL;DRs.

### 🛠️ Built: v1 static site (2026-10-16)
- **Hosting:** GitHub Pages, rebuilt daily by a GitHub Action. FastAPI + SQLite retired.
- **Votes/saves:** per-browser localStorage for now (option A). Upgrade paths: private-Gist sync (B)
  or Supabase + GitHub login for shared lab votes (C).
- **Sources:** bioRxiv, medRxiv, arXiv, ~35 journal RSS feeds, OpenAlex (ISSN journals, PI watchlist,
  niche-topic search). Buzz = HN + Bluesky + ScienceDaily (X/Twitter API isn't free).
- **New topics:** B cells & antibodies, PhIP-seq, viral mimicry, trogocytosis, CRISPR & Perturb-seq.
- **PI watchlist:** Caroline Uhler, Nir Hacohen, Stefani Spranger (`config/people.yaml`).
- **Ranking:** user-tunable weighted blend (recency / buzz / PI / similarity-to-saves / upvotes), per-topic weights.
- **Alerts:** `alerts.xml` RSS feed of PI + buzz hits.
- **Next up:** embeddings for "For You" similarity, LLM TL;DRs, email/Slack push alerts, shared votes.

---

## 1. The Feed — "what's new and interesting" (crowdsourcing)

Two things blended together: **new** (recency) and **interesting** (signal). Notes on each:

### Signals for "interesting"
- [ ] Social attention — X/Twitter, **Bluesky** (a lot of science moved here), Mastodon
- [ ] **Altmetric** score / attention data
- [ ] Citation velocity (early citations relative to age) — via OpenAlex / Semantic Scholar
- [ ] **alphaXiv** comments/discussion on the preprint
- [ ] **PubPeer** activity (post-publication review, red flags)
- [ ] Download / view counts where exposed (bioRxiv gives some)
- [ ] Lab-internal signal — upvotes / saves / "we should read this" from lab members
- [ ] Newsletter mentions (e.g. field-specific Substacks, journal editor picks)

### Crowdsourcing mechanics (the alphaXiv-inspired part)
- [ ] Upvote / downvote or ⭐ save
- [ ] Comments / threaded discussion per paper
- [ ] Tags added by users (folksonomy) vs. auto-tags
- [ ] "Journal club" queue — flag for group discussion
- [ ] Who's the crowd? just the lab? the field? → ties to the North Star question above
- [ ] Reputation / weighting — do trusted curators' votes count more?

### Ranking / dedup notes
- [ ] A paper appears as **preprint → then journal version** — must dedup (match by DOI / title / authors)
- [ ] Blend recency + interest score; let user toggle "Newest" vs "Hottest"
- [ ] Avoid echo chamber — reserve some slots for serendipity / adjacent fields

---

## 2. Subfields / Topics I care about (Gilbert-lab-focused)

Priority topics — feed should bias toward these:
- [ ] **Cancer biology**
- [ ] **Immunology** (esp. **T cell** biology)
- [ ] **Microbiology**
- [ ] **Virology**
- [ ] **Molecular biology**
- [ ] **Bio ML** / computational biology / ML-for-biology
- [ ] *(Gilbert lab specific — add: CRISPR screens, functional genomics, Perturb-seq, gene regulation?)*

Notes on classification:
- [ ] How do we assign a paper to a topic? Options: source journal/section, MeSH terms (PubMed), keyword rules, or an **LLM classifier / embeddings** over title+abstract
- [ ] Topics aren't mutually exclusive — allow multiple tags per paper
- [ ] Let me tune weights per topic (e.g. more T-cell, less general micro)
- [ ] A "must-not-miss" author/lab watchlist could be its own filter

---

## 3. Sources to pull from

### Already on my list
- Science + sub-journals
- Cell + sub-journals
- Nature + sub-journals
- bioRxiv
- medRxiv

### Suggested additions — "what other sources?"

**Preprint servers**
- [ ] **arXiv** — `q-bio` (quant bio) + `cs.LG` / `stat.ML` (essential for **Bio ML**)
- [ ] **Research Square**, **Preprints.org**, **ChemRxiv** (chem-bio) — lower priority

**High-value journals beyond the big 3 (by subfield)**
- [ ] **eLife**, **PNAS** — broad, high quality
- [ ] Immunology / T cell: **Immunity**, **Nature Immunology**, **Science Immunology**, **J. Exp. Med. (JEM)**, **Blood**, **J. Immunology**
- [ ] Cancer: **Cancer Cell**, **Nature Cancer**, **Cancer Discovery**
- [ ] Micro / viral: **Cell Host & Microbe**, **Nature Microbiology**, **mBio**, **PLOS Pathogens**, **J. Virology**
- [ ] Mol bio / genomics: **Molecular Cell**, **Genome Biology**, **Genome Research**, **Nucleic Acids Research**, **EMBO J.**, **Cell Genomics**, **Cell Systems**
- [ ] Bio ML / methods: **Nature Methods**, **Nature Biotechnology**, **Nature Machine Intelligence**, **PLOS Comp Bio**, **Bioinformatics**
- [ ] Open access: **PLOS Biology**, **Science Advances**, **Nature Communications**, **Cell Reports**

**Aggregators / metadata APIs (probably the real backbone — free, structured)**
- [ ] **Europe PMC** — great free API, indexes journals *and* preprints, full-text search
- [ ] **PubMed / NCBI E-utilities** — huge, MeSH terms, has API
- [ ] **OpenAlex** — open scholarly graph, generous free API (citations, topics, authors)
- [ ] **Semantic Scholar API** — TLDRs, citation context, influence
- [ ] **Crossref** — DOI metadata, good for dedup / matching preprint↔journal

**Bio ML / conference-specific**
- [ ] **OpenReview** (NeurIPS, ICML, ICLR), **Papers with Code**, **Hugging Face Papers**

**Crowd / social layers**
- [ ] **alphaXiv**, **PubPeer**, **Altmetric**, **Bluesky** science feeds

> **Ingestion note:** most journals + preprint servers publish **RSS/Atom feeds** — often the simplest, most reliable pull mechanism. Aggregator APIs (Europe PMC / OpenAlex / Crossref) may cover many journals at once, reducing per-source scrapers. Watch out for **paywalls / ToS** on scraping publisher sites directly.

---

## 4. Open questions / decisions to make

- [x] ~~**Audience:**~~ → **Gilbert lab** (internal, shared)
- [x] ~~**Delivery:**~~ → **web app**
- [x] ~~**Filtering engine:**~~ → **LLM / embeddings**
- [ ] **Ingestion strategy:** per-source RSS scrapers vs. lean on Europe PMC + OpenAlex aggregators?
- [ ] **Cadence:** real-time, daily digest, weekly?
- [ ] **Auth:** how do lab members sign in? (Arc SSO / Google Workspace / shared login?)
- [ ] **Hosting:** where does the web app live? (internal server / Modal / cloud?)
- [ ] **LLM pipeline:** which model for classify + score + TL;DR? embeddings store? (cost/rate limits)
- [ ] **Storage:** where do papers + metadata + votes live? (sqlite? postgres?)
- [ ] **Dedup:** matching preprint ↔ published version (DOI/title/author heuristics)
- [ ] **De-dup of *my* attention:** mark read / hide seen
- [ ] **Cost / rate limits:** which APIs need keys, which have quotas

---

## 5. Nice-to-haves / later
- [ ] Auto-summaries (TL;DR per paper via LLM)
- [ ] "Why you're seeing this" explanation per feed item
- [ ] Author / lab watchlists
- [ ] Export to reference manager (Zotero) / BibTeX
- [ ] Weekly "you might have missed" recap
- [ ] Connect to Arc Papers MCP / internal tooling

---

*Scratch space below — dump raw thoughts here:*

>
