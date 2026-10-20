// biopapers — static client. Loads data/papers.json (+ older.json and the embeddings in the
// background, ml.json when the ML / CS section is opened, map.json + favorites.json for the
// Graph), ranks and renders the feed.
// Saves / upvotes / favorites / settings live in localStorage, optionally synced to a secret GitHub Gist.

"use strict";

const PAGE = 40;
const STORE_KEY = "biopapers:v1";
const SYNC_KEY = "biopapers:sync"; // { token, gistId } — kept apart so Export JSON never contains the token
const SYNC_MAPS = ["votes", "saved", "hidden", "favorites", "summaries"];
const SYNC_SETTINGS = ["weights", "topicWeights", "halfLife", "tabOrder", "hiddenTabs"];
const DAY = 86400000;
const MUST_READ = 7; // AI score (0–10) at which a paper lands in "Must read"
const NEAR_K = 25; // each favorite pulls its 25 most similar papers into "Near favorites"

// Fixed tabs around each section's topic tabs (from config/topics.yaml).
const SPECIAL_TABS = {
  foryou: { label: "For You", color: "var(--ink)" },
  mustread: { label: "Must read", color: "#e6b422" },
  nearfav: { label: "♥ Near favorites", color: "var(--hot)" },
  pis: { label: "PI Watch", color: "#a98fd6" },
  buzz: { label: "Buzz", color: "var(--hot)" },
  mlforyou: { label: "For You", color: "var(--ink)" },
  trending: { label: "Trending", color: "var(--hot)" },
  saved: { label: "Saved", color: "#f2b15c" },
};
const SECTION_TABS = {
  bio: { head: ["foryou", "mustread", "nearfav"], tail: ["pis", "buzz", "saved"] },
  ml: { head: ["mlforyou", "mustread", "nearfav", "trending"], tail: ["saved"] },
};
const HOT_TABS = new Set(["buzz", "trending"]);

const WEIGHTS = {
  recency: { label: "Recency", def: 1 },
  ai: { label: "AI relevance (Claude's 0–10 score for you)", def: 1 },
  buzz: { label: "Buzz (HF upvotes · HN · Bluesky · press)", def: 0.6 },
  pi: { label: "Watched PIs", def: 1 },
  affinity: { label: "Similar to my favorites, saves and upvotes", def: 0.8 },
  vote: { label: "My upvotes", def: 0.3 },
};

const SORTS = { foryou: "For You", date: "Newest", buzz: "Buzz" };

// ---- state ---------------------------------------------------------------

const defaults = () => ({
  votes: {}, saved: {}, hidden: {}, favorites: {}, summaries: {}, removed: {}, settingsAt: 0,
  weights: Object.fromEntries(Object.entries(WEIGHTS).map(([k, w]) => [k, w.def])),
  topicWeights: {}, halfLife: 7, tabOrder: [], hiddenTabs: [],
  section: "bio", tabBy: { bio: "foryou", ml: "mlforyou" }, sortByTab: {},
  lastVisit: null, prevVisit: null, view: "feed", graphSection: "all", graphAround: true,
});

function loadState() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return defaults();
    const saved = JSON.parse(raw), d = defaults();
    return { ...d, ...saved, weights: { ...d.weights, ...saved.weights } }; // new weights get their default
  } catch { return defaults(); }
}

let state = loadState();
let sync = (() => { try { return JSON.parse(localStorage.getItem(SYNC_KEY)) || {}; } catch { return {}; } })();
const syncStatus = { at: null, error: null, busy: false, again: false };
let syncTimer = null, syncedJSON = null;

const settingsJSON = (s) => JSON.stringify(SYNC_SETTINGS.map((k) => s[k]));
let settingsSeen = settingsJSON(state);

function saveLocal() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* private mode etc. */ }
}
function persist() {
  const sj = settingsJSON(state);
  if (sj !== settingsSeen) { state.settingsAt = Date.now(); settingsSeen = sj; }
  saveLocal();
  scheduleSync();
}

// Count a visit only if the last one was >1h ago, so reloads don't clear "new".
(function trackVisit() {
  const now = Date.now();
  if (!state.lastVisit || now - state.lastVisit > 3600000) {
    state.prevVisit = state.lastVisit;
    state.lastVisit = now;
    persist();
  }
})();
const prevVisitDay = state.prevVisit ? new Date(state.prevVisit).toISOString().slice(0, 10) : null;

// ---- data ----------------------------------------------------------------

let DATA = { sections: {}, topics: [], people: [], papers: [], generated: null };
let byId = new Map();
let topicSection = new Map();
let mlLoaded = null; // promise once ml.json is requested
let VEC = null; // { dim, data: Int8Array, norms, index: Map(id -> row) } once vectors load
let MAP = null, mapLoaded = null; // { ids, xy, nn, index } once map.json loads
let favsLoaded = null; // favorites.json: favorites outside the window (on the graph, not in feeds)

const curTab = () => state.tabBy[state.section];
const setTab = (t) => { state.tabBy[state.section] = t; };
let ui = { shown: PAGE, query: "", newOnly: false, pi: null, favLabel: null, graphQuery: "",
  takehomeOpen: new Set(), takehomeBusy: new Map() }; // busy: id -> status text (or "error: …")

const has = (map, p) => !!(map[p.id] || (p.aliases || []).some((a) => map[a]));
const isNew = (p) => prevVisitDay && p.added > prevVisitDay;

async function fetchPapers(name) {
  try {
    const res = await fetch(`data/${name}`, { cache: "no-cache" });
    return res.ok ? (await res.json()).papers : [];
  } catch { return []; }
}

function addPapers(list) {
  for (const p of list) {
    if (!byId.has(p.id)) { DATA.papers.push(p); byId.set(p.id, p); }
  }
  affinityCache = nearCache = null;
}

function loadML() {
  mlLoaded ||= fetchPapers("ml.json").then((list) => { addPapers(list); render({ keepPage: true }); });
  return mlLoaded;
}

async function load() {
  const res = await fetch("data/papers.json", { cache: "no-cache" });
  if (!res.ok) throw new Error(`papers.json: ${res.status}`);
  const { papers, ...meta } = await res.json();
  DATA = { ...meta, papers: [] };
  topicSection = new Map(DATA.topics.map((t) => [t.slug, t.section || "bio"]));
  if (!DATA.sections?.[state.section]) state.section = "bio";
  addPapers(papers);
  render();
  if (state.section === "ml") await loadML();
  addPapers(await fetchPapers("older.json"));
  render({ keepPage: true });
  loadVectors();
}

async function loadVectors() {
  try {
    const [meta, buf] = await Promise.all([
      fetch("data/vectors.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)),
      fetch("data/vectors.bin", { cache: "no-cache" }).then((r) => (r.ok ? r.arrayBuffer() : null)),
    ]);
    if (!meta || !buf) return;
    const data = new Int8Array(buf), dim = meta.dim, n = meta.ids.length;
    if (data.length !== n * dim) return;
    const norms = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0; for (let j = i * dim; j < (i + 1) * dim; j++) s += data[j] * data[j];
      norms[i] = Math.sqrt(s) || 1;
    }
    VEC = { dim, data, norms, index: new Map(meta.ids.map((id, i) => [id, i])) };
    affinityCache = nearCache = null;
    render({ keepPage: true });
    // Favorites the build hasn't embedded need the map loaded to be placed (Near favorites uses them too).
    if (Object.keys(state.favorites).some((id) => !VEC.index.has(id))) loadMap();
  } catch { /* no embeddings yet: TF-IDF fallback */ }
}

function loadMap() {
  favsLoaded ||= fetchPapers("favorites.json").then((list) => {
    for (const p of list) if (!byId.has(p.id)) byId.set(p.id, { ...p, sections: ["bio", "ml"] });
    affinityCache = nearCache = null;
  });
  mapLoaded ||= Promise.all([
    fetch("data/map.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    favsLoaded,
  ]).then(([m]) => { if (m) MAP = { ...m, index: new Map(m.ids.map((id, i) => [id, i])) }; })
    .finally(() => { mapNeedsFit = true; render({ keepPage: true }); placeFavorites(); });
  return mapLoaded;
}

const vecRow = (p) => [p.id, ...(p.aliases || [])].map((k) => VEC.index.get(k)).find((i) => i !== undefined);

// Papers most like `p`, by embedding cosine (or the map's neighbour lists before vectors load).
function similarTo(p, n = 20) {
  const r = VEC && vecRow(p);
  if (r === undefined || !VEC) {
    const i = MAP?.index.get(p.id);
    return i === undefined ? [] : MAP.nn[i].map((j) => byId.get(MAP.ids[j])).filter(Boolean);
  }
  const { dim, data, norms } = VEC, base = r * dim, out = [];
  for (const [id, i] of VEC.index) {
    if (i === r || !byId.has(id)) continue;
    let dot = 0; for (let j = 0; j < dim; j++) dot += data[base + j] * data[i * dim + j];
    out.push([dot / (norms[r] * norms[i]), id]);
  }
  return out.sort((a, b) => b[0] - a[0]).slice(0, n).map(([, id]) => byId.get(id));
}

// ---- ranking ---------------------------------------------------------------

const STOP = new Set(("the and for with that this from are was were been into their which these those our " +
  "using use used via between within across can may also show shows here than more most both such " +
  "however whether while study studies data results result based new novel role analysis approach " +
  "method methods model models cell cells human mice mouse").split(" "));

function tokens(p) {
  const text = `${p.title} ${p.title} ${(p.abstract || "").slice(0, 800)}`.toLowerCase();
  const out = new Map();
  for (const w of text.match(/[a-z][a-z0-9-]{2,}/g) || []) {
    if (!STOP.has(w)) out.set(w, (out.get(w) || 0) + 1);
  }
  for (const t of p.topics || []) out.set(`#${t}`, 2);
  return out;
}

let affinityCache = null; // Map(id -> 0..1), rebuilt when the corpus or likes change
let nearCache = null; // Map(id -> { fav, sim }) for "Near favorites", rebuilt alongside it

function affinityScores() {
  if (affinityCache) return affinityCache;
  const liked = new Set([...Object.keys(state.favorites), ...Object.keys(state.saved), ...Object.keys(state.votes)]);
  affinityCache = new Map();
  if (!liked.size) return affinityCache;
  if (VEC && embeddingAffinity(liked)) return affinityCache;

  const docs = DATA.papers.map((p) => [p.id, tokens(p)]);
  const df = new Map();
  for (const [, t] of docs) for (const w of t.keys()) df.set(w, (df.get(w) || 0) + 1);
  const idf = (w) => Math.log((docs.length + 1) / ((df.get(w) || 0) + 1));
  const vec = (t) => {
    const v = new Map(); let n = 0;
    for (const [w, c] of t) { const x = (1 + Math.log(c)) * idf(w); v.set(w, x); n += x * x; }
    return [v, Math.sqrt(n) || 1];
  };

  // Profile = sum of liked papers' vectors (saved snapshots cover papers aged out of the window).
  const profile = new Map();
  for (const id of liked) {
    const p = byId.get(id) || state.saved[id]?.paper || state.favorites[id]?.paper;
    if (!p) continue;
    const [v, n] = vec(tokens(p));
    for (const [w, x] of v) profile.set(w, (profile.get(w) || 0) + x / n);
  }
  let pn = 0; for (const x of profile.values()) pn += x * x; pn = Math.sqrt(pn) || 1;

  let max = 0;
  for (const [id, t] of docs) {
    const [v, n] = vec(t);
    let dot = 0; for (const [w, x] of v) dot += x * (profile.get(w) || 0);
    const s = dot / (n * pn);
    affinityCache.set(id, s); max = Math.max(max, s);
  }
  if (max > 0) for (const [id, s] of affinityCache) affinityCache.set(id, s / max);
  return affinityCache;
}

// Cosine to the mean of liked papers' embeddings, stretched so the median paper scores 0 and the
// closest scores 1 (raw cosines bunch up between ~0.6 and 0.9). False if no liked paper has a vector.
function embeddingAffinity(liked) {
  const { dim, data, norms } = VEC, prof = new Float32Array(dim);
  let any = false;
  for (const id of liked) {
    const r = VEC.index.get(id);
    if (r === undefined) continue;
    any = true;
    for (let j = 0; j < dim; j++) prof[j] += data[r * dim + j] / norms[r];
  }
  if (!any) return false;
  let pn = 0; for (const x of prof) pn += x * x; pn = Math.sqrt(pn) || 1;
  const sims = [];
  for (const p of DATA.papers) {
    const r = vecRow(p);
    if (r === undefined) continue;
    let dot = 0; for (let j = 0; j < dim; j++) dot += prof[j] * data[r * dim + j];
    sims.push([p.id, dot / (pn * norms[r])]);
  }
  const sorted = sims.map(([, s]) => s).sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length / 2)] ?? 0, hi = sorted[sorted.length - 1] ?? 1;
  for (const [id, s] of sims) affinityCache.set(id, Math.max(0, (s - lo) / ((hi - lo) || 1)));
  return true;
}

// For each favorite with an embedding, its NEAR_K most similar feed papers; a paper near several
// favorites belongs to the closest. Map(paper id -> { fav: favorite id, sim }).
function nearFavorites() {
  if (nearCache) return nearCache;
  nearCache = new Map();
  if (!VEC) return nearCache;
  const { dim, data, norms } = VEC;
  for (const fid of Object.keys(state.favorites)) {
    const f = VEC.index.get(fid);
    if (f === undefined) continue;
    const sims = [];
    for (const p of DATA.papers) {
      const r = vecRow(p);
      if (r === undefined || r === f) continue;
      let dot = 0; for (let j = 0; j < dim; j++) dot += data[f * dim + j] * data[r * dim + j];
      sims.push([dot / (norms[f] * norms[r]), p.id]);
    }
    sims.sort((a, b) => b[0] - a[0]);
    for (const [sim, id] of sims.slice(0, NEAR_K)) {
      if (!(nearCache.get(id)?.sim >= sim)) nearCache.set(id, { fav: fid, sim });
    }
  }
  return nearCache;
}

const favPaper = (id) => byId.get(id) || state.favorites[id]?.paper;
const shortTitle = (t, n = 5) => { const w = (t || "").split(/\s+/); return w.slice(0, n).join(" ") + (w.length > n ? "…" : ""); };
const favLabel = (id) => state.favorites[id]?.label || shortTitle(favPaper(id)?.title);

const sectionTopics = (p) => (p.topics || []).filter((t) => topicSection.get(t) === state.section);

function topicWeight(p) {
  const topics = sectionTopics(p);
  if (!topics.length) return 1;
  return Math.max(...topics.map((t) => state.topicWeights[t] ?? 1));
}

function score(p, aff, now) {
  const w = state.weights;
  const age = Math.max(0, (now - Date.parse(p.date)) / DAY);
  const recency = Math.pow(0.5, age / state.halfLife);
  const buzz = Math.min(1, Math.log1p(p.buzz_score || 0) / Math.log1p(20));
  const pi = p.pis?.length ? 1 : 0;
  const vote = has(state.votes, p) || has(state.saved, p) ? 1 : 0;
  const ai = (p.ai?.score ?? 3) / 10; // unscored papers count as a middling 3
  return topicWeight(p) * (w.recency * recency + w.ai * ai + w.buzz * buzz + w.pi * pi
    + w.affinity * (aff.get(p.id) || 0) + w.vote * vote);
}

// ---- tabs / filtering -----------------------------------------------------------

function allTabs() {
  const { head, tail } = SECTION_TABS[state.section];
  const topicTabs = DATA.topics.filter((t) => (t.section || "bio") === state.section).map((t) => t.slug);
  const known = [...head, ...topicTabs, ...tail];
  const order = state.tabOrder.filter((t) => known.includes(t));
  return [...order, ...known.filter((t) => !order.includes(t))];
}

function tabInfo(slug) {
  return SPECIAL_TABS[slug] || DATA.topics.find((t) => t.slug === slug) || { label: slug, color: "var(--muted)" };
}

function inTab(p, tab) {
  if (!(p.sections || ["bio"]).includes(state.section)) return false;
  if (tab === "foryou") return (sectionTopics(p).length || p.pis?.length) && topicWeight(p) > 0;
  if (tab === "mlforyou") return topicWeight(p) > 0;
  if (tab === "pis") return p.pis?.length > 0;
  if (tab === "mustread") return (p.ai?.score ?? 0) >= MUST_READ && topicWeight(p) > 0;
  if (tab === "nearfav") return nearFavorites().has(p.id);
  if (HOT_TABS.has(tab)) return (p.buzz_score || 0) > 0;
  return p.topics?.includes(tab);
}

function savedPapers() {
  return Object.entries(state.saved)
    .sort((a, b) => b[1].ts - a[1].ts)
    .map(([id, s]) => byId.get(id) || DATA.papers.find((p) => p.aliases?.includes(id)) || s.paper)
    .filter(Boolean);
}

const sortFor = (tab) => state.sortByTab[tab] || (HOT_TABS.has(tab) ? "buzz" : "foryou");

function visiblePapers() {
  const tab = curTab();
  let list = tab === "saved" ? savedPapers() : DATA.papers.filter((p) => inTab(p, tab) && !has(state.hidden, p));
  if (tab === "pis" && ui.pi) list = list.filter((p) => p.pis.includes(ui.pi));
  if (tab === "nearfav" && ui.favLabel) {
    const near = nearFavorites();
    list = list.filter((p) => state.favorites[near.get(p.id)?.fav]?.label === ui.favLabel);
  }
  if (ui.newOnly) list = list.filter(isNew);
  if (ui.query) {
    const q = ui.query.toLowerCase();
    list = list.filter((p) => `${p.title} ${p.authors.join(" ")} ${p.abstract} ${p.venue}`.toLowerCase().includes(q));
  }
  if (tab === "saved") return list;
  const sort = sortFor(tab);
  if (sort === "date") return list.sort((a, b) => b.date.localeCompare(a.date) || (b.buzz_score || 0) - (a.buzz_score || 0));
  if (sort === "buzz") {
    // "Hot": attention decayed by age (HN-style gravity), so this week's breakout beats last month's.
    const now = Date.now();
    const hot = (p) => (p.buzz_score || 0) / Math.pow(Math.max(0, (now - Date.parse(p.date)) / DAY) + 2, 1.2);
    return list.sort((a, b) => hot(b) - hot(a) || b.date.localeCompare(a.date));
  }
  const aff = state.weights.affinity > 0 ? affinityScores() : new Map();
  const now = Date.now();
  const scored = list.map((p) => [score(p, aff, now), p]);
  return scored.sort((a, b) => b[0] - a[0]).map(([, p]) => p);
}

// ---- rendering ---------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function render({ keepPage = false } = {}) {
  if (!keepPage) ui.shown = PAGE;
  const tabs = allTabs().filter((t) => !state.hiddenTabs.includes(t) || t === curTab());
  if (!tabs.includes(curTab())) setTab(SECTION_TABS[state.section].head[0]);
  const tab = curTab();

  const graph = state.view === "graph";
  $("sections").innerHTML = Object.entries(DATA.sections || {}).map(([k, label]) =>
    `<button data-section="${k}" class="${!graph && k === state.section ? "active" : ""}">${esc(label)}</button>`).join("")
    + `<button data-view="graph" class="${graph ? "active" : ""}" title="Every paper laid out by similarity, centered on your ♥ favorites">Graph</button>`;

  const updated = DATA.generated ? new Date(DATA.generated).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";
  const inSection = DATA.papers.filter((p) => (p.sections || ["bio"]).includes(state.section)).length;
  const loading = state.section === "ml" && !inSection ? "loading ML papers…" : `${inSection.toLocaleString()} papers`;
  $("sub").textContent = `${loading} · updated ${updated}`;
  document.body.classList.toggle("graph-mode", graph);
  if (graph) { renderGraph(); return; }

  const visible = (p) => !has(state.hidden, p);
  const count = (t) => t === "saved" ? Object.keys(state.saved).length
    : DATA.papers.reduce((n, p) => n + (inTab(p, t) && visible(p) ? 1 : 0), 0);
  $("tabs").innerHTML = tabs.map((t) => {
    const info = tabInfo(t);
    return `<button class="pill ${t === tab ? "active" : ""}" data-tab="${t}" style="--c:${info.color}">
      <span class="dot"></span>${esc(info.label)} <span class="n">${count(t)}</span></button>`;
  }).join("");

  const sort = sortFor(tab);
  const sortLabel = (k, label) => k === "buzz" && state.section === "ml" ? "Hot" : label;
  $("sortbar").innerHTML = tab === "saved" ? `<button class="active">Recently saved</button>`
    : Object.entries(SORTS).map(([k, label]) =>
        `<button data-sort="${k}" class="${k === sort ? "active" : ""}">${sortLabel(k, label)}</button>`).join("");

  const nNew = DATA.papers.filter((p) => isNew(p) && inTab(p, tab === "saved" ? SECTION_TABS[state.section].head[0] : tab)).length;
  $("new-label").textContent = prevVisitDay ? `new since last visit (${nNew})` : "new since last visit";

  $("chips").innerHTML = tab === "pis"
    ? [null, ...DATA.people.map((p) => p.name)].map((name) =>
        `<button class="chip ${ui.pi === name ? "active" : ""}" data-pi="${esc(name ?? "")}"
          title="${esc(DATA.people.find((x) => x.name === name)?.note || "")}">${esc(name ?? "All PIs")}</button>`).join("")
    : tab === "nearfav"
    ? [null, ...new Set(Object.values(state.favorites).map((f) => f.label).filter(Boolean))].map((l) =>
        `<button class="chip ${ui.favLabel === l ? "active" : ""}" data-favlabel="${esc(l ?? "")}">${esc(l ?? "All favorites")}</button>`).join("")
    : "";

  const list = visiblePapers();
  $("feed").innerHTML = list.length
    ? list.slice(0, ui.shown).map(card).join("")
    : `<div class="empty">${tab === "saved" ? "Nothing saved yet — hit ☆ on a paper."
      : tab === "nearfav" ? (Object.keys(state.favorites).length ? "Nothing yet. Papers near your favorites show up once their embeddings load, after the next build for newly added ones."
        : "Hit ♥ on papers you love, or look up any paper in Graph, and new papers near them collect here.")
      : "No papers match."}</div>`;
  $("more").hidden = list.length <= ui.shown;
  $("more").textContent = `Show more (${list.length - ui.shown} left)`;
}

const fmtK = (n) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
const link = (href, text, title = "") =>
  `<a href="${esc(href)}" target="_blank" rel="noopener"${title ? ` title="${esc(title)}"` : ""}>${text}</a>`;

function card(p) {
  // Current section's topics first, so the card edge matches the tab you're in.
  const topics = (p.topics || []).map((t) => DATA.topics.find((x) => x.slug === t)).filter(Boolean)
    .sort((a, b) => ((b.section || "bio") === state.section) - ((a.section || "bio") === state.section));
  const color = topics[0]?.color || (p.pis?.length ? "#a98fd6" : (p.buzz_score ? "var(--hot)" : "var(--muted)"));
  const b = p.buzz || {}, hf = p.hf || {};
  const arxivId = p.doi?.startsWith("10.48550/arxiv.") ? p.doi.slice("10.48550/arxiv.".length) : null;
  const lastNames = new Set((p.pis || []).map((n) => n.split(" ").pop().toLowerCase()));
  const authors = (p.authors || []).map((a) =>
    lastNames.has(a.split(" ").pop().toLowerCase()) ? `<b>${esc(a)}</b>` : esc(a)).join(", ");
  const voted = has(state.votes, p), saved = has(state.saved, p), fav = has(state.favorites, p);
  const near = nearCache?.get(p.id);
  const hot = [
    hf.upvotes && arxivId ? `<a class="badge hot" href="https://huggingface.co/papers/${esc(arxivId)}" target="_blank" rel="noopener" title="Hugging Face Daily Papers upvotes">▲ ${hf.upvotes} HF</a>` : "",
    b.hn ? `<a class="badge hot" href="${esc(b.hn_url)}" target="_blank" rel="noopener" title="Hacker News points">HN ${b.hn}</a>` : "",
    b.bsky ? `<span class="badge hot" title="Bluesky posts sharing this (likes+reposts: ${b.bsky_likes || 0})">Bluesky ${b.bsky}</span>` : "",
    ...(b.news || []).slice(0, 2).map((n) => `<a class="badge hot" href="${esc(n.url)}" target="_blank" rel="noopener" title="${esc(n.title)}">press</a>`),
  ].join("");

  const links = [
    arxivId && link(`https://arxiv.org/pdf/${arxivId}`, "PDF"),
    arxivId && link(`https://www.alphaxiv.org/abs/${arxivId}`, "alphaXiv", "Discussion on alphaXiv"),
    hf.comments && arxivId && link(`https://huggingface.co/papers/${arxivId}`, `${hf.comments} comment${hf.comments > 1 ? "s" : ""}`),
    hf.github && link(hf.github, `code${hf.stars ? ` ★ ${fmtK(hf.stars)}` : ""}`),
    hf.project && link(hf.project, "project page"),
    p.preprint_url && link(p.preprint_url, "preprint version"),
  ].filter(Boolean);

  return `<article class="paper" style="--c:${color}" data-id="${esc(p.id)}">
    <div class="actions">
      <button class="${voted ? "on" : ""}" data-act="vote" aria-label="Upvote" title="Upvote">▲</button>
      <button class="${saved ? "on" : ""}" data-act="save" aria-label="Save" title="Save">${saved ? "★" : "☆"}</button>
      <button class="${fav ? "on fav" : ""}" data-act="fav" aria-label="Favorite" title="Favorite: a landmark paper the Graph and Near favorites center on">♥</button>
      <button class="hide" data-act="hide" aria-label="Hide" title="${curTab() === "saved" ? "Unsave" : "Hide"}">✕</button>
    </div>
    <div class="body">
      <h3><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
      <div class="meta">
        ${isNew(p) ? `<span class="badge new">new</span>` : ""}
        <span class="badge venue">${esc(p.venue)}${p.kind === "preprint" && !/rxiv/i.test(p.venue) ? " · preprint" : ""}</span>
        ${hf.org ? `<span class="badge org">${esc(hf.org)}</span>` : ""}
        <span class="date">${esc(p.date)}</span>
        ${(p.pis || []).map((n) => `<span class="badge pi">PI · ${esc(n)}</span>`).join("")}
        ${near ? `<span class="badge near" title="Near your favorite: ${esc(favPaper(near.fav)?.title)}">♥ ${esc(favLabel(near.fav))}</span>` : ""}
        ${p.ai ? `<span class="badge ai ${p.ai.score >= MUST_READ ? "hi" : ""}" title="Claude's relevance score for you: ${esc(p.ai.why)}">AI ${p.ai.score}</span>` : ""}
        ${hot}
        ${topics.map((t) => `<span class="badge" style="--c:${t.color}">${esc(t.label)}</span>`).join("")}
      </div>
      <div class="authors" title="${esc((p.authors || []).join(", "))}">${authors}${p.n_authors ? ` <span>(${p.n_authors})</span>` : ""}</div>
      ${p.ai?.tldr || hf.tldr ? `<div class="tldr"><b>TL;DR</b> ${esc(p.ai?.tldr || hf.tldr)}${p.ai?.why ? ` <span class="why">· ${esc(p.ai.why)}</span>` : ""}</div>` : ""}
      ${p.abstract ? `<div class="abstract">${esc(p.abstract)}</div>` : ""}
      ${takehomeHTML(p)}
      <div class="links">${links.join("")}<button type="button" class="linkbtn th-btn" data-act="takehome" title="A short take-home summary, written on demand">${state.summaries[p.id] ? (ui.takehomeOpen.has(p.id) ? "hide take-home" : "take-home ✓") : "take-home"}</button><button type="button" class="linkbtn" data-act="map" title="Show it on the Graph with its nearest neighbours">similar papers</button></div>
    </div>
  </article>`;
}

// ---- actions ---------------------------------------------------------------

function toggle(name, p, value) {
  const keys = [p.id, ...(p.aliases || [])].filter((k) => state[name][k]);
  if (keys.length) forget(name, keys);
  else state[name][p.id] = value;
  return !keys.length;
}

function snapshot(p, abstractLen = 600) {
  // Saved papers keep a copy so they survive aging out of the 30-day window. Favorites keep a
  // longer abstract: the build embeds them from this copy once they're out of the window.
  const { id, doi, title, authors, venue, date, url, kind, topics, pis, abstract } = p;
  return { id, doi, title, authors, venue, date, url, kind, topics, pis, abstract: (abstract || "").slice(0, abstractLen) };
}

function toggleFavorite(p) {
  toggle("favorites", p, { ts: Date.now(), paper: snapshot(p, 1500) });
  affinityCache = nearCache = null;
  mapNeedsFit = true;
  placeFavorites();
}

$("feed").addEventListener("click", (e) => {
  const abs = e.target.closest(".abstract");
  if (abs) { abs.classList.toggle("open"); return; }
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.closest(".paper").dataset.id;
  const p = byId.get(id) || state.saved[id]?.paper || state.favorites[id]?.paper;
  if (!p) return;
  const act = btn.dataset.act;
  if (act === "map") { showOnMap(p); return; }
  if (act === "takehome") { takeHome(p); return; }
  if (act === "takehome-redo") { takeHome(p, { redo: true }); return; }
  if (act === "fav") toggleFavorite(p);
  if (act === "vote") toggle("votes", p, Date.now());
  if (act === "save") toggle("saved", p, { ts: Date.now(), paper: snapshot(p) });
  if (act === "hide") {
    if (curTab() === "saved") toggle("saved", p);
    else state.hidden[p.id] = Date.now();
  }
  if (act !== "hide") affinityCache = nearCache = null;
  persist();
  render({ keepPage: true });
});

$("tabs").addEventListener("click", (e) => {
  const t = e.target.closest("[data-tab]");
  if (!t) return;
  setTab(t.dataset.tab); ui.pi = null; persist(); render();
  window.scrollTo({ top: 0 });
});
$("sortbar").addEventListener("click", (e) => {
  const s = e.target.closest("[data-sort]");
  if (!s) return;
  state.sortByTab[curTab()] = s.dataset.sort; persist(); render();
});
$("sections").addEventListener("click", (e) => {
  if (e.target.closest("[data-view=graph]")) {
    if (state.view === "graph") return;
    state.view = "graph"; mapNeedsFit = true; persist(); render();
    window.scrollTo({ top: 0 });
    return;
  }
  const b = e.target.closest("[data-section]");
  if (!b || (b.dataset.section === state.section && state.view !== "graph")) return;
  state.section = b.dataset.section; state.view = "feed"; ui.pi = null; persist(); render();
  if (state.section === "ml") loadML();
  window.scrollTo({ top: 0 });
});
$("chips").addEventListener("click", (e) => {
  const c = e.target.closest("[data-pi]");
  if (c) { ui.pi = c.dataset.pi || null; render(); }
  const f = e.target.closest("[data-favlabel]");
  if (f) { ui.favLabel = f.dataset.favlabel || null; render(); }
});
$("more").addEventListener("click", () => { ui.shown += PAGE; render({ keepPage: true }); });
$("new-only").addEventListener("change", (e) => { ui.newOnly = e.target.checked; render(); });
let searchTimer;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { ui.query = e.target.value.trim(); render(); }, 150);
});

// ---- graph -------------------------------------------------------------------------
// UMAP layout of the embeddings (pipeline/layout.py), its own view next to Bio | ML / CS. Dots take
// their topic's color; ♥ favorites are labelled hubs, and "around favorites" narrows the graph to
// them plus the papers closest to each. Saved / upvoted papers get a ring. The lookup box finds
// any paper, in the feed or anywhere in the literature (OpenAlex), to ♥ it. Click a dot to see that
// paper and the ones most like it below.

const mapView = { k: 1, tx: 0, ty: 0 };
let mapSel = null, mapHover = null, mapFocus = null, mapPts = [], mapSize = { w: 0, h: 0, s: 0 };
let mapAllLinks = false, mapNeedsFit = true, lookupResults = [];
let placeStatus = null, placing = null, extractor = null; // in-browser embedding of new favorites

function showOnMap(p) {
  mapFocus = p.id;
  state.view = "graph"; persist(); render();
  window.scrollTo({ top: 0 });
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isLiked = (p) => has(state.saved, p) || has(state.votes, p);
const isFav = (p) => has(state.favorites, p);
const favOnMap = () => Object.keys(state.favorites).map((id) => MAP?.index.get(id)).filter((i) => i !== undefined);
const repoUrl = location.hostname.endsWith(".github.io")
  ? `https://github.com/${location.hostname.split(".")[0]}/${location.pathname.split("/")[1]}` : null;

function sizeMap() {
  const c = $("map"), w = c.parentElement.clientWidth, h = Math.min(Math.round(window.innerHeight * 0.68), 680);
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.width = `${w}px`; c.style.height = `${h}px`;
  }
  mapSize = { w, h, s: Math.min(w, h) * 0.92, dpr };
}

// Layout coords (0..1) -> CSS pixels, before and after pan/zoom.
const mapBase = (i) => [(mapSize.w - mapSize.s) / 2 + MAP.xy[i][0] * mapSize.s, (mapSize.h - mapSize.s) / 2 + MAP.xy[i][1] * mapSize.s];
const mapScreen = (i) => { const [x, y] = mapBase(i); return [x * mapView.k + mapView.tx, y * mapView.k + mapView.ty]; };

function fitMap(indexes) {
  if (!indexes.length) { Object.assign(mapView, { k: 1, tx: 0, ty: 0 }); return; }
  const pts = indexes.map(mapBase), xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const k = Math.min(12, Math.max(1, Math.min(mapSize.w / (x1 - x0 + 80), mapSize.h / (y1 - y0 + 80))));
  Object.assign(mapView, { k, tx: mapSize.w / 2 - ((x0 + x1) / 2) * k, ty: mapSize.h / 2 - ((y0 + y1) / 2) * k });
}

function matchesGraph(p) {
  const q = ui.graphQuery.toLowerCase();
  return !q || `${p.title} ${(p.authors || []).join(" ")} ${p.abstract || ""} ${p.venue || ""}`.toLowerCase().includes(q);
}

function renderGraph() {
  $("more").hidden = true;
  loadML(); // the graph spans both sections
  for (const b of $("graph-sections").querySelectorAll("[data-gsec]")) b.classList.toggle("active", b.dataset.gsec === state.graphSection);
  $("graph-around").checked = state.graphAround;
  renderFavs();
  if (!MAP) {
    $("graph-legend").hidden = true;
    $("feed").innerHTML = `<div class="empty">${mapLoaded ? "No graph yet. It's built by the next daily run." : "Loading the graph…"}</div>`;
    loadMap();
    return;
  }
  $("graph-legend").hidden = false;
  sizeMap();
  const near = nearFavorites(), favIdx = new Set(favOnMap());
  const around = state.graphAround && favIdx.size > 0;
  const muted = cssVar("--muted");
  mapPts = [];
  MAP.ids.forEach((id, i) => {
    const p = byId.get(id);
    if (!p || has(state.hidden, p)) return;
    const fav = favIdx.has(i);
    if (p.favorite_only && !fav) return; // a favorite you've since removed
    if (!fav && state.graphSection !== "all" && !(p.sections || ["bio"]).includes(state.graphSection)) return;
    if (around && !fav && !near.has(id)) return;
    const topic = (p.topics || []).map((t) => DATA.topics.find((x) => x.slug === t)).find(Boolean);
    mapPts.push({ i, p, fav, near: near.get(id), color: topic?.color || (p.pis?.length ? "#a98fd6" : muted),
                  liked: isLiked(p), dim: !matchesGraph(p) });
  });
  if (mapFocus) {
    const i = MAP.index.get(mapFocus);
    if (i !== undefined) {
      if (!mapPts.some((q) => q.i === i)) { state.graphAround = false; state.graphSection = "all"; mapFocus = MAP.ids[i]; return renderGraph(); }
      mapSel = i;
      mapView.k = Math.max(mapView.k, 3);
      const [x, y] = mapBase(i);
      mapView.tx = mapSize.w / 2 - x * mapView.k; mapView.ty = mapSize.h / 2 - y * mapView.k;
    }
    mapFocus = null; mapNeedsFit = false;
  }
  if (mapNeedsFit) { fitMap(around ? mapPts.map((q) => q.i) : []); mapNeedsFit = false; }
  $("graph-note").textContent = around ? `${favIdx.size} favorite${favIdx.size > 1 ? "s" : ""} and the ${mapPts.length - favIdx.size} papers closest to them`
    : favIdx.size ? "" : "♥ papers (or look one up above) and the graph centers on them";
  drawMap();

  const sel = mapSel != null ? byId.get(MAP.ids[mapSel]) : null;
  $("feed").innerHTML = sel
    ? card(sel) + `<h4 class="map-h">Most similar</h4>` + similarTo(sel).map(card).join("")
    : `<div class="empty">Click a dot to see the paper and the papers most like it.</div>`;
}

function drawMap() {
  const c = $("map"), ctx = c.getContext("2d"), { w, h, dpr } = mapSize;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const ink = cssVar("--ink"), hot = cssVar("--hot"), accent = cssVar("--accent"), card = cssVar("--card");
  const onMap = new Set(mapPts.map((q) => q.i));
  const line = (i, j, alpha, color = ink, width = 1) => {
    if (j === undefined || !onMap.has(j)) return;
    const [x1, y1] = mapScreen(i), [x2, y2] = mapScreen(j);
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  };
  const around = state.graphAround && mapPts.some((q) => q.fav);
  for (const q of mapPts) {
    if (mapAllLinks) for (const j of MAP.nn[q.i].slice(0, 4)) line(q.i, j, 0.05);
    if (around && q.near) line(q.i, MAP.index.get(q.near.fav), 0.3, hot);
    else if (!around && q.fav) for (const j of MAP.nn[q.i].slice(0, 6)) line(q.i, j, 0.4, hot);
  }
  if (mapSel != null && onMap.has(mapSel)) for (const j of MAP.nn[mapSel]) line(mapSel, j, 0.85, ink, 1.5);

  const r = Math.min(around ? 5 : 4, 2 + Math.log2(mapView.k) + (around ? 1 : 0));
  for (const q of mapPts) {
    if (q.fav) continue;
    const [x, y] = mapScreen(q.i);
    if (x < -5 || y < -5 || x > w + 5 || y > h + 5) continue;
    ctx.globalAlpha = q.dim ? 0.08 : 0.8; ctx.fillStyle = q.color;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    if (q.liked && !q.dim) {
      ctx.globalAlpha = 1; ctx.strokeStyle = accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r + 2.5, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  ctx.textBaseline = "middle";
  for (const q of mapPts) {
    if (!q.fav) continue;
    const [x, y] = mapScreen(q.i);
    ctx.globalAlpha = q.dim ? 0.3 : 1;
    ctx.font = "20px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.fillStyle = hot; ctx.fillText("♥", x, y);
    const label = favLabel(q.p.id);
    ctx.font = "600 12px system-ui, sans-serif"; ctx.textAlign = "left";
    ctx.lineWidth = 3; ctx.strokeStyle = card; ctx.strokeText(label, x + 12, y);
    ctx.fillStyle = ink; ctx.fillText(label, x + 12, y);
  }
  for (const [i, ring] of [[mapHover, 8], [mapSel, 11]]) {
    if (i == null || !onMap.has(i)) continue;
    const [x, y] = mapScreen(i);
    ctx.globalAlpha = 1; ctx.strokeStyle = ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, ring, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function mapHit(x, y) {
  let best = null, bestD = 12 * 12;
  for (const q of mapPts) {
    const [px, py] = mapScreen(q.i), d = (px - x) ** 2 + (py - y) ** 2 - (q.fav ? 40 : 0);
    if (d < bestD) { bestD = d; best = q; }
  }
  return best;
}

// ---- placing new favorites (in-browser embeddings) ----
// A favorite the build hasn't embedded (looked up from the literature, or ♥'d since the last build)
// is embedded right here with the same model the build uses (bge-small-en-v1.5 via transformers.js;
// ~34 MB, downloaded once and cached by the browser) and dropped onto the graph near its 8 most
// similar papers. Its vector is kept in this browser, so it stays put across reloads.

const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";
const FAVVEC_KEY = "biopapers:favvec";
const loadFavVecs = () => { try { return JSON.parse(localStorage.getItem(FAVVEC_KEY)) || {}; } catch { return {}; } };
const saveFavVecs = (m) => { try { localStorage.setItem(FAVVEC_KEY, JSON.stringify(m)); } catch { /* */ } };
const toB64 = (v) => btoa(String.fromCharCode(...new Uint8Array(v.buffer, v.byteOffset, v.length)));
const fromB64 = (s) => new Int8Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
const embedText = (p) => `${p.title}. ${(p.abstract || "").slice(0, 1000)}`; // same as pipeline/embed.py

async function embedInBrowser(texts) {
  extractor ||= import(TJS_URL).then(({ pipeline }) =>
    pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" }));
  const out = (await (await extractor)(texts, { pooling: "cls", normalize: true })).tolist();
  // Same int8 quantization as the build: each vector scaled so its largest component is ±127.
  return out.map((v) => { const m = Math.max(...v.map(Math.abs)) || 1; return Int8Array.from(v, (x) => Math.round((x / m) * 127)); });
}

// Append one paper to the vectors and the graph, at the similarity-weighted centre of its 8 nearest papers.
function addToGraph(id, vec) {
  if (VEC.index.has(id) || MAP.ids.length !== VEC.index.size) return;
  const { dim } = VEC, n = VEC.index.size;
  const data = new Int8Array((n + 1) * dim); data.set(VEC.data); data.set(vec, n * dim);
  const norms = new Float32Array(n + 1); norms.set(VEC.norms);
  let ss = 0; for (const x of vec) ss += x * x; norms[n] = Math.sqrt(ss) || 1;
  const sims = [];
  for (let i = 0; i < n; i++) {
    let dot = 0; for (let j = 0; j < dim; j++) dot += vec[j] * data[i * dim + j];
    sims.push([dot / (norms[n] * norms[i]), i]);
  }
  const top = sims.sort((a, b) => b[0] - a[0]).slice(0, 8);
  let x = 0, y = 0, w = 0;
  for (const [sim, i] of top) { const k = Math.max(sim, 0) ** 4; x += MAP.xy[i][0] * k; y += MAP.xy[i][1] * k; w += k; }
  VEC = { ...VEC, data, norms }; VEC.index.set(id, n);
  MAP.ids.push(id); MAP.xy.push([x / (w || 1), y / (w || 1)]); MAP.nn.push(top.map(([, i]) => i)); MAP.index.set(id, n);
}

function placeFavorites() {
  if (!VEC || !MAP || placing) return placing;
  const missing = Object.keys(state.favorites).filter((id) => !MAP.index.has(id) && favPaper(id)?.title);
  if (!missing.length) return null;
  for (const id of missing) if (!byId.has(id)) byId.set(id, { ...favPaper(id), id, sections: ["bio", "ml"], favorite_only: true });
  const cache = loadFavVecs(), todo = missing.filter((id) => !cache[id]);
  placing = (async () => {
    try {
      if (todo.length) {
        placeStatus = "loading";
        if (state.view === "graph") renderFavs();
        const vecs = await embedInBrowser(todo.map((id) => embedText(favPaper(id))));
        todo.forEach((id, k) => { cache[id] = toB64(vecs[k]); });
        saveFavVecs(cache);
      }
      for (const id of missing) if (cache[id]) addToGraph(id, fromB64(cache[id]));
      placeStatus = null;
    } catch (err) {
      placeStatus = `Couldn't place new favorites on the graph (${err.message}).`;
    } finally {
      placing = null;
      affinityCache = nearCache = null; mapNeedsFit = true;
      render({ keepPage: true });
      if (Object.keys(state.favorites).some((id) => !MAP.index.has(id) && !placeStatus)) placeFavorites(); // ♥'d while we were busy
    }
  })();
  return placing;
}

// ---- favorites panel ----

function renderFavs() {
  const favs = Object.entries(state.favorites).sort((a, b) => b[1].ts - a[1].ts);
  const notes = [
    placeStatus === "loading" ? "Placing new favorites on the graph (the first time downloads a 34 MB model)…" : placeStatus,
    favs.length && !sync.token && "Turn on Sync in ⚙︎ to keep favorites on every device and let the daily AI scorer learn from them.",
  ].filter(Boolean);
  $("favs").innerHTML = `
    <h3>♥ Favorites <span class="n">${favs.length}</span></h3>
    <p class="hint">Landmark papers the graph centers on. Give them a label and new papers near them are
      grouped under it in the ♥ Near favorites tab. ${notes.join(" ")}</p>
    ${favs.length ? `<ul>${favs.map(([id, f]) => {
      const p = favPaper(id), on = MAP?.index.has(id);
      return `<li data-fav="${esc(id)}">
        <button type="button" class="fav-title" data-favact="show" ${on ? "" : "disabled"} title="${on ? "Show on the graph" : "Not on the graph yet"}">${esc(p?.title || id)}</button>
        <input type="text" data-favact="label" value="${esc(f.label || "")}" placeholder="label" aria-label="Label">
        <span class="st">${on ? "" : placeStatus === "loading" ? "placing…" : "not placed"}</span>
        <button type="button" class="rm" data-favact="remove" aria-label="Remove favorite" title="Remove">✕</button>
      </li>`;
    }).join("")}</ul>` : ""}`;
}

$("favs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-favact]"), id = b?.closest("[data-fav]")?.dataset.fav;
  if (!id) return;
  if (b.dataset.favact === "show") { mapFocus = id; render({ keepPage: true }); }
  if (b.dataset.favact === "remove") {
    forget("favorites", [id]); affinityCache = nearCache = null; mapNeedsFit = true;
    persist(); render({ keepPage: true });
  }
});
$("favs").addEventListener("change", (e) => {
  if (e.target.dataset.favact !== "label") return;
  const id = e.target.closest("[data-fav]").dataset.fav, f = state.favorites[id];
  if (!f) return;
  const label = e.target.value.trim();
  state.favorites[id] = { ...f, ts: Date.now(), ...(label ? { label } : {}) };
  if (!label) delete state.favorites[id].label;
  persist(); drawMap();
});

// ---- lookup: any paper, from the feed or OpenAlex ----

// "10.1101/2024.01.01.123v2" / "https://doi.org/…" / "arXiv:2401.01234v3" / an arxiv.org URL -> our paper id.
function paperRef(q) {
  const ax = q.match(/(?:arxiv\.org\/(?:abs|pdf)\/|arxiv:\s*|^)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$/i);
  if (ax) return `10.48550/arxiv.${ax[1]}`;
  const doi = q.match(/10\.\d{4,9}\/[^\s"'<>?#]+/);
  if (!doi) return null;
  let d = doi[0].toLowerCase().replace(/[.,;)]+$/, "");
  if (d.startsWith("10.1101/")) d = d.replace(/v\d+$/, ""); // bioRxiv version suffix, as the pipeline does
  return d;
}

function fromOpenAlex(w) {
  const inv = w.abstract_inverted_index, pos = [];
  if (inv) for (const [word, idxs] of Object.entries(inv)) for (const i of idxs) pos[i] = word;
  const doi = w.doi ? w.doi.replace(/^https?:\/\/doi\.org\//, "").toLowerCase() : null;
  const id = doi?.startsWith("10.1101/") ? doi.replace(/v\d+$/, "") : doi || w.id;
  const authors = (w.authorships || []).map((a) => a.author?.display_name).filter(Boolean);
  const src = w.primary_location?.source?.display_name || "";
  return {
    id, doi, title: w.title || w.display_name || "(untitled)",
    authors: authors.length > 13 ? [...authors.slice(0, 10), "…", ...authors.slice(-3)] : authors,
    abstract: pos.filter(Boolean).join(" "), date: w.publication_date || String(w.publication_year || ""),
    venue: /arxiv/i.test(src) ? "arXiv" : /biorxiv/i.test(src) ? "bioRxiv" : /medrxiv/i.test(src) ? "medRxiv" : src || "Other",
    kind: w.type === "preprint" ? "preprint" : "article",
    url: doi ? `https://doi.org/${doi}` : w.primary_location?.landing_page_url || w.id,
  };
}

let lookupSeq = 0;
async function lookup(q) {
  const seq = ++lookupSeq, ref = paperRef(q), lq = q.toLowerCase();
  const local = ref && byId.has(ref) ? [byId.get(ref)]
    : [...byId.values()].filter((p) => p.title.toLowerCase().includes(lq)).slice(0, 6);
  lookupResults = local.map((p) => ({ p, local: true }));
  renderLookup(q.length >= 3 ? "searching the literature…" : "");
  if (q.length < 3) return;
  const select = "id,doi,title,display_name,publication_date,publication_year,type,authorships,primary_location,abstract_inverted_index";
  const url = ref
    ? `https://api.openalex.org/works/doi:${encodeURIComponent(ref)}?select=${select}`
    : `https://api.openalex.org/works?search=${encodeURIComponent(q)}&per-page=8&select=${select}`;
  let remote = [];
  try {
    const res = await fetch(url);
    const body = res.ok ? await res.json() : null;
    remote = (body?.results || (body?.id ? [body] : [])).map(fromOpenAlex);
  } catch { /* offline: local results only */ }
  if (seq !== lookupSeq) return;
  const seen = new Set(lookupResults.map((r) => r.p.id));
  for (const p of remote) if (!seen.has(p.id)) { seen.add(p.id); lookupResults.push(byId.has(p.id) ? { p: byId.get(p.id), local: true } : { p, local: false }); }
  renderLookup(lookupResults.length ? "" : "No papers found.");
}

function renderLookup(status) {
  const row = (r, i) => {
    const p = r.p, fav = has(state.favorites, p), on = MAP?.index.has(p.id);
    const meta = [p.venue, (p.date || "").slice(0, 4), (p.authors || []).filter((a) => a !== "…").slice(0, 3).join(", ")].filter(Boolean).join(" · ");
    return `<li data-lk="${i}">
      <button type="button" class="heart ${fav ? "on" : ""}" data-lkact="fav" aria-label="${fav ? "Remove favorite" : "Add to favorites"}" title="${fav ? "Remove favorite" : "Add to favorites"}">♥</button>
      <div class="lk-body"><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a><div class="lk-meta">${esc(meta)}</div></div>
      ${on ? `<button type="button" class="lk-show" data-lkact="show">show</button>` : ""}
    </li>`;
  };
  const local = lookupResults.map((r, i) => [r, i]).filter(([r]) => r.local);
  const remote = lookupResults.map((r, i) => [r, i]).filter(([r]) => !r.local);
  $("lookup-results").hidden = !ui.graphQuery;
  $("lookup-results").innerHTML = (local.length ? `<div class="lk-sec">In your feed</div><ul>${local.map(([r, i]) => row(r, i)).join("")}</ul>` : "")
    + (remote.length ? `<div class="lk-sec">From the literature · OpenAlex</div><ul>${remote.map(([r, i]) => row(r, i)).join("")}</ul>` : "")
    + (status ? `<div class="lk-status">${esc(status)}</div>` : "");
}

let lookupTimer;
$("lookup").addEventListener("input", (e) => {
  clearTimeout(lookupTimer);
  ui.graphQuery = e.target.value.trim();
  if (!ui.graphQuery) { lookupResults = []; renderLookup(""); }
  lookupTimer = setTimeout(() => {
    if (MAP) { for (const q of mapPts) q.dim = !matchesGraph(q.p); drawMap(); }
    if (ui.graphQuery) lookup(ui.graphQuery);
  }, 300);
});
document.addEventListener("click", (e) => {
  // A ♥ in the results re-renders the list, detaching the clicked button: that's not an outside click.
  if (e.target.isConnected && !e.target.closest(".lookup-box")) $("lookup-results").hidden = true;
});
$("lookup").addEventListener("focus", () => { if (ui.graphQuery) $("lookup-results").hidden = false; });
$("lookup").addEventListener("keydown", (e) => { if (e.key === "Escape") $("lookup-results").hidden = true; });
$("lookup-results").addEventListener("click", (e) => {
  const b = e.target.closest("[data-lkact]"), r = lookupResults[b?.closest("[data-lk]")?.dataset.lk];
  if (!r) return;
  if (b.dataset.lkact === "show") { mapFocus = r.p.id; render({ keepPage: true }); return; }
  toggleFavorite(r.p);
  persist(); renderLookup(""); render({ keepPage: true });
});
$("graph-sections").addEventListener("click", (e) => {
  const b = e.target.closest("[data-gsec]");
  if (!b) return;
  state.graphSection = b.dataset.gsec; mapNeedsFit = true; persist(); render({ keepPage: true });
});
$("graph-around").addEventListener("change", (e) => {
  state.graphAround = e.target.checked; mapNeedsFit = true; persist(); render({ keepPage: true });
});

(function wireMap() {
  const c = $("map"), tip = $("map-tip"), pointers = new Map();
  let drag = null, pinch = null;
  const pos = (e) => { const b = c.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  const zoomAt = (x, y, f) => {
    const k = Math.min(40, Math.max(0.5, mapView.k * f)); f = k / mapView.k;
    mapView.tx = x - (x - mapView.tx) * f; mapView.ty = y - (y - mapView.ty) * f; mapView.k = k;
    drawMap();
  };
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [x, y] = pos(e); zoomAt(x, y, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  c.addEventListener("pointerdown", (e) => {
    c.setPointerCapture(e.pointerId); pointers.set(e.pointerId, pos(e));
    if (pointers.size === 1) drag = { start: pos(e), last: pos(e), moved: false };
    if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); drag = null; }
  });
  c.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [x, y]);
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / pinch); pinch = d; return;
    }
    if (drag) {
      if (Math.hypot(x - drag.start[0], y - drag.start[1]) > 4) drag.moved = true;
      if (drag.moved) { mapView.tx += x - drag.last[0]; mapView.ty += y - drag.last[1]; drag.last = [x, y]; tip.hidden = true; drawMap(); return; }
    }
    if (e.pointerType !== "mouse") return;
    const q = mapHit(x, y), i = q?.i ?? null;
    if (i !== mapHover) { mapHover = i; drawMap(); }
    tip.hidden = !q;
    if (q) {
      tip.textContent = q.near ? `${q.p.title}  ·  near ♥ ${favLabel(q.near.fav)}` : q.p.title;
      tip.style.left = `${Math.min(x + 14, mapSize.w - 260)}px`; tip.style.top = `${y + 14}px`;
    }
  });
  const end = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (drag && !drag.moved && pointers.size === 0) {
      const q = mapHit(...pos(e));
      mapSel = q ? q.i : null;
      render({ keepPage: true });
    }
    if (pointers.size === 0) drag = null;
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", (e) => { pointers.delete(e.pointerId); drag = null; pinch = null; });
  c.addEventListener("pointerleave", () => { tip.hidden = true; if (mapHover != null) { mapHover = null; drawMap(); } });
  $("map-links").addEventListener("change", (e) => { mapAllLinks = e.target.checked; drawMap(); });
  $("map-reset").addEventListener("click", () => { mapNeedsFit = true; render({ keepPage: true }); });
  window.addEventListener("resize", () => {
    if (state.view !== "graph" || !MAP) return;
    // The old pan/zoom is meaningless at the new size: re-centre on the selection, or re-fit.
    if (mapSel != null) mapFocus = MAP.ids[mapSel];
    else mapNeedsFit = true;
    renderGraph();
  });
})();

// ---- on-demand take-homes ----------------------------------------------------------
// The "take-home" button asks Claude (through OpenRouter, with a key kept in this browser) for a short
// summary of one paper: the main message, key findings, approach and caveats. It reads the open-access
// full text from Europe PMC when there is one, else the abstract. Results are cached (and synced with
// your other state), so each paper costs once; nothing runs until you click.

const OR_KEY = "biopapers:openrouter"; // { key, model } — never exported or synced
const TAKEHOME_MODELS = {
  "anthropic/claude-opus-5.5": { label: "Claude Opus 5.5", cost: "≈ 2¢ from an abstract, 6–8¢ from full text" },
  "anthropic/claude-sonnet-5.5": { label: "Claude Sonnet 5.5", cost: "≈ 1¢ from an abstract, 3–4¢ from full text" },
  "anthropic/claude-haiku-5.5": { label: "Claude Haiku 5.5", cost: "under ¼¢ from an abstract, ≈ ½¢ from full text" },
};
let orConf = (() => { try { return JSON.parse(localStorage.getItem(OR_KEY)) || {}; } catch { return {}; } })();
orConf.model = TAKEHOME_MODELS[orConf.model] ? orConf.model : "anthropic/claude-opus-5.5";
const saveOrConf = () => { try { localStorage.setItem(OR_KEY, JSON.stringify(orConf)); } catch { /* */ } };

const TAKEHOME_SYSTEM = `You write the take-home of one research paper for a busy scientist who will not read it. Be concrete and specific: name the system, method, and effect sizes where given. Don't hype, and don't restate the title. If you only have the abstract, stick to what it supports.
Return JSON with:
- take_home: 2-3 sentences: the main message, and why it matters.
- key_findings: 3-5 short bullets, each one concrete result (with numbers where the text gives them).
- approach: 1-2 sentences on how they did it (model system, data, method).
- caveats: 1-2 sentences on limitations or what the work doesn't show.`;
const TAKEHOME_SCHEMA = {
  type: "object", additionalProperties: false,
  required: ["take_home", "key_findings", "approach", "caveats"],
  properties: { take_home: { type: "string" }, key_findings: { type: "array", items: { type: "string" } },
                approach: { type: "string" }, caveats: { type: "string" } },
};

// Open-access full text from Europe PMC (sections as "## Title" + paragraphs), or null.
async function europePmcText(p) {
  if (!p.doi) return null;
  const q = encodeURIComponent(`DOI:"${p.doi}"`);
  const hits = (await (await fetch(`https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${q}&resultType=core&format=json`)).json())
    .resultList?.result || [];
  const hit = hits.find((x) => x.pmcid && x.isOpenAccess === "Y");
  if (!hit) return null;
  const res = await fetch(`https://www.ebi.ac.uk/europepmc/webservices/rest/${hit.pmcid}/fullTextXML`);
  if (!res.ok) return null;
  const body = new DOMParser().parseFromString(await res.text(), "application/xml").querySelector("body");
  if (!body) return null;
  body.querySelectorAll("xref, table, graphic, supplementary-material, ref-list").forEach((n) => n.remove());
  const parts = [];
  const walk = (el) => {
    for (const c of el.children) {
      if (c.tagName === "title") parts.push(`\n## ${c.textContent.trim()}`);
      else if (c.tagName === "p" || c.tagName === "caption") parts.push(c.textContent.replace(/\s+/g, " ").trim());
      else walk(c);
    }
  };
  walk(body);
  const text = parts.join("\n").trim();
  return text.length > 2000 ? text.slice(0, 60000) : null; // ~15k tokens at most
}

async function takeHome(p, { redo = false } = {}) {
  const id = p.id;
  if (state.summaries[id] && !redo) {
    ui.takehomeOpen.has(id) ? ui.takehomeOpen.delete(id) : ui.takehomeOpen.add(id);
    return render({ keepPage: true });
  }
  if (ui.takehomeBusy.has(id) && !ui.takehomeBusy.get(id).startsWith("error")) return;
  ui.takehomeOpen.add(id);
  if (!orConf.key) {
    ui.takehomeBusy.set(id, "error: Add an OpenRouter key in ⚙︎ → On-demand take-homes first.");
    return render({ keepPage: true });
  }
  ui.takehomeBusy.set(id, "Looking for open-access full text…"); render({ keepPage: true });
  let text = null, source = "abstract";
  try { text = await europePmcText(p); if (text) source = "full text (Europe PMC)"; } catch { /* abstract it is */ }
  ui.takehomeBusy.set(id, `Reading the ${text ? "full text" : "abstract"}…`); render({ keepPage: true });
  const paper = `${p.title}\n${[p.venue, p.date, (p.authors || []).filter((a) => a !== "…").slice(0, 6).join(", ")].filter(Boolean).join(" · ")}\n\n`
    + (text || `Abstract: ${p.abstract || "(no abstract available)"}`);
  try {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${orConf.key}`, "Content-Type": "application/json", "X-Title": "biopapers" },
      body: JSON.stringify({
        model: orConf.model, max_tokens: 6000, reasoning: { effort: "low" }, usage: { include: true },
        messages: [{ role: "system", content: TAKEHOME_SYSTEM }, { role: "user", content: paper }],
        response_format: { type: "json_schema", json_schema: { name: "takehome", strict: true, schema: TAKEHOME_SCHEMA } },
      }),
    });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(out.error?.message || `OpenRouter ${res.status}`);
    const choice = out.choices?.[0];
    if (choice?.finish_reason !== "stop") throw new Error(`the model stopped early (${choice?.finish_reason || "no answer"})`);
    const data = JSON.parse(choice.message.content);
    state.summaries[id] = { ts: Date.now(), model: orConf.model, source, cost: out.usage?.cost ?? null, data };
    ui.takehomeBusy.delete(id);
    persist();
  } catch (err) {
    ui.takehomeBusy.set(id, `error: ${err.message}`);
  }
  render({ keepPage: true });
}

function takehomeHTML(p) {
  if (!ui.takehomeOpen.has(p.id)) return "";
  const busy = ui.takehomeBusy.get(p.id), s = state.summaries[p.id];
  if (busy) return `<div class="takehome ${busy.startsWith("error") ? "err" : "busy"}">${esc(busy.replace(/^error: /, ""))}</div>`;
  if (!s) return "";
  const d = s.data, model = TAKEHOME_MODELS[s.model]?.label || s.model;
  return `<div class="takehome">
    <p class="th-main">${esc(d.take_home)}</p>
    ${d.key_findings?.length ? `<ul>${d.key_findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
    ${d.approach ? `<p><b>How</b> ${esc(d.approach)}</p>` : ""}
    ${d.caveats ? `<p><b>Caveats</b> ${esc(d.caveats)}</p>` : ""}
    <div class="th-meta">from the ${esc(s.source)} · ${esc(model)}${s.cost != null ? ` · $${(+s.cost).toFixed(3)}` : ""}
      · <button type="button" class="linkbtn" data-act="takehome-redo">redo</button></div>
  </div>`;
}

function renderTakehomeSettings() {
  $("or-box").innerHTML = `
    <p class="hint">The take-home button on each paper asks Claude for a short summary, only when you click it.
      Paste an <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener">OpenRouter key</a>; it stays in this
      browser and is never exported or synced. Best to make a separate key with a small credit limit.</p>
    <div class="btnrow">${orConf.key
      ? `<span class="hint">Key saved (…${esc(orConf.key.slice(-4))})</span><button type="button" data-or="forget">Forget key</button>`
      : `<input type="password" id="or-key" placeholder="sk-or-…" autocomplete="off" aria-label="OpenRouter key"><button type="button" data-or="save">Save</button>`}</div>
    <label class="row">Model
      <select id="or-model">${Object.entries(TAKEHOME_MODELS).map(([k, m]) =>
        `<option value="${k}" ${k === orConf.model ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select>
    </label>
    <p class="hint">${esc(TAKEHOME_MODELS[orConf.model].cost)} per paper.</p>`;
}
$("or-box").addEventListener("click", (e) => {
  const act = e.target.closest("[data-or]")?.dataset.or;
  if (act === "save") { const k = $("or-key").value.trim(); if (k) { orConf.key = k; saveOrConf(); } }
  if (act === "forget") { delete orConf.key; saveOrConf(); }
  if (act) renderTakehomeSettings();
});
$("or-box").addEventListener("change", (e) => {
  if (e.target.id === "or-model") { orConf.model = e.target.value; saveOrConf(); renderTakehomeSettings(); }
});

// ---- settings ---------------------------------------------------------------

function renderSettings() {
  $("weights").innerHTML = Object.entries(WEIGHTS).map(([k, w]) => `
    <label class="row">${esc(w.label)}
      <input type="range" min="0" max="2" step="0.1" value="${state.weights[k]}" data-weight="${k}">
      <output>${(+state.weights[k]).toFixed(1)}</output>
    </label>`).join("");
  $("halflife").value = String(state.halfLife);
  renderSync();
  renderTakehomeSettings();

  const tabs = allTabs();
  $("tabs-section").textContent = `— ${DATA.sections?.[state.section] || ""}`;
  $("tab-editor").innerHTML = tabs.map((t, i) => {
    const info = tabInfo(t), isTopic = !SPECIAL_TABS[t], w = state.topicWeights[t] ?? 1;
    return `<li data-tab="${t}">
      <input type="checkbox" ${state.hiddenTabs.includes(t) ? "" : "checked"} data-show title="Show tab" ${t === "foryou" ? "disabled" : ""}>
      <span class="name" style="--c:${info.color}"><span class="dot"></span>${esc(info.label)}</span>
      ${isTopic ? `<input type="range" min="0" max="2" step="0.25" value="${w}" data-tw title="Weight in For You"><output>${w}</output>` : ""}
      <button type="button" data-move="-1" ${i === 0 ? "disabled" : ""} aria-label="Move up">↑</button>
      <button type="button" data-move="1" ${i === tabs.length - 1 ? "disabled" : ""} aria-label="Move down">↓</button>
    </li>`;
  }).join("");
}

$("open-settings").addEventListener("click", () => { renderSettings(); $("settings").showModal(); });
$("settings").addEventListener("close", () => render());

$("weights").addEventListener("input", (e) => {
  const k = e.target.dataset.weight;
  if (!k) return;
  state.weights[k] = +e.target.value;
  e.target.nextElementSibling.textContent = (+e.target.value).toFixed(1);
  persist();
});
$("halflife").addEventListener("change", (e) => { state.halfLife = +e.target.value; persist(); });

$("tab-editor").addEventListener("input", (e) => {
  const li = e.target.closest("li"); if (!li) return;
  const t = li.dataset.tab;
  if (e.target.matches("[data-tw]")) {
    state.topicWeights[t] = +e.target.value;
    e.target.nextElementSibling.textContent = e.target.value;
  }
  if (e.target.matches("[data-show]")) {
    state.hiddenTabs = e.target.checked ? state.hiddenTabs.filter((x) => x !== t) : [...state.hiddenTabs, t];
  }
  persist();
});
$("tab-editor").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-move]"); if (!btn) return;
  const order = allTabs(), i = order.indexOf(btn.closest("li").dataset.tab), j = i + +btn.dataset.move;
  [order[i], order[j]] = [order[j], order[i]];
  state.tabOrder = order; persist(); renderSettings();
});

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

$("export-json").addEventListener("click", () =>
  download(`biopapers-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(state, null, 1), "application/json"));
$("import-json").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const incoming = JSON.parse(await f.text());
    state = { ...state, ...incoming,
      votes: { ...state.votes, ...incoming.votes }, saved: { ...state.saved, ...incoming.saved },
      hidden: { ...state.hidden, ...incoming.hidden } };
    persist(); affinityCache = nearCache = null; renderSettings(); render();
  } catch (err) { alert(`Couldn't import: ${err.message}`); }
  e.target.value = "";
});
$("export-bib").addEventListener("click", () => {
  const bib = savedPapers().map((p) => {
    const first = (p.authors?.[0] || "anon").split(" ").pop().replace(/[^A-Za-z]/g, "");
    const key = `${first}${p.date.slice(0, 4)}${p.title.split(/\s+/)[0].replace(/[^A-Za-z]/g, "")}`;
    const fields = { title: `{${p.title}}`, author: (p.authors || []).filter((a) => a !== "…").join(" and "),
      journal: p.venue, year: p.date.slice(0, 4), doi: p.doi, url: p.url };
    return `@article{${key},\n${Object.entries(fields).filter(([, v]) => v).map(([k, v]) => `  ${k} = {${v}}`).join(",\n")}\n}`;
  }).join("\n\n");
  download("biopapers-saved.bib", bib, "application/x-bibtex");
});
$("unhide").addEventListener("click", () => { forget("hidden", Object.keys(state.hidden)); persist(); render(); });
$("reset").addEventListener("click", () => {
  if (!confirm(`Clear all saves, upvotes and settings${sync.token ? " on every synced device" : " in this browser"}?`)) return;
  for (const m of SYNC_MAPS) forget(m, Object.keys(state[m]));
  state = { ...defaults(), removed: state.removed, lastVisit: state.lastVisit };
  persist(); affinityCache = nearCache = null; renderSettings(); render();
});

// ---- sync (secret GitHub Gist) ---------------------------------------------------------
// Votes / saves / hides merge per paper, newest wins. Un-doing one leaves a timestamped
// tombstone in `removed` so other devices drop it too. Settings sync as one block, last
// writer wins. Section, tab and sort choices stay per-device.

const GIST_FILE = "biopapers-state.json";
const TOMBSTONE_DAYS = 180;

const entryTs = (v) => (typeof v === "number" ? v : v?.ts) || 0;
const sortedObj = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => (a < b ? -1 : a > b)));
const syncedPart = (s) => Object.fromEntries(["removed", "settingsAt", ...SYNC_MAPS, ...SYNC_SETTINGS].map((k) => [k, s[k]]));

function forget(name, ids) {
  const now = Date.now();
  for (const id of ids) { delete state[name][id]; state.removed[`${name}:${id}`] = now; }
}

function mergeSynced(local, remote) {
  const cutoff = Date.now() - TOMBSTONE_DAYS * DAY;
  const removed = {};
  for (const src of [local.removed, remote.removed])
    for (const [k, t] of Object.entries(src || {})) if (t > cutoff) removed[k] = Math.max(removed[k] || 0, t);
  const out = { removed: sortedObj(removed) };
  for (const m of SYNC_MAPS) {
    const merged = {};
    for (const src of [local[m], remote[m]])
      for (const [id, v] of Object.entries(src || {}))
        if (entryTs(v) > (removed[`${m}:${id}`] || 0) && entryTs(v) >= entryTs(merged[id])) merged[id] = v;
    out[m] = sortedObj(merged);
  }
  const newer = (remote.settingsAt || 0) > (local.settingsAt || 0) ? remote : local;
  for (const k of SYNC_SETTINGS) out[k] = newer[k] ?? local[k];
  out.settingsAt = newer.settingsAt || 0;
  return out;
}

async function gh(path, { method = "GET", body } = {}) {
  const res = await fetch(`https://api.github.com${path}`, {
    method, cache: "no-store", body: body && JSON.stringify(body),
    headers: { Authorization: `Bearer ${sync.token}`, Accept: "application/vnd.github+json" },
  });
  if (res.status === 401) throw new Error("GitHub rejected the token — paste a new one");
  if (!res.ok) throw new Error(`GitHub ${res.status}`);
  return res.json();
}

async function findGist() {
  for (let page = 1; page <= 10; page++) {
    const list = await gh(`/gists?per_page=100&page=${page}`);
    const hit = list.find((g) => g.files[GIST_FILE]);
    if (hit || list.length < 100) return hit?.id || null;
  }
  return null;
}

async function readGist() {
  const f = (await gh(`/gists/${sync.gistId}`)).files[GIST_FILE];
  if (!f) return {};
  // The API inlines up to 1 MB; past that the content is only at raw_url.
  const text = f.truncated ? await (await fetch(f.raw_url, { cache: "no-store" })).text() : f.content;
  return JSON.parse(text || "{}");
}

function saveSyncConfig() {
  try { sync.token ? localStorage.setItem(SYNC_KEY, JSON.stringify(sync)) : localStorage.removeItem(SYNC_KEY); } catch { /* */ }
}

function scheduleSync() {
  if (!sync.token || JSON.stringify(syncedPart(state)) === syncedJSON) return;
  clearTimeout(syncTimer);
  syncTimer = setTimeout(syncNow, 2000);
}

async function syncNow() {
  if (!sync.token) return;
  if (syncStatus.busy) { syncStatus.again = true; return; }
  syncStatus.busy = true; clearTimeout(syncTimer);
  try {
    sync.gistId ||= await findGist();
    const remote = sync.gistId ? await readGist() : {};
    const before = JSON.stringify(syncedPart(state));
    // Merge into the state as it is *now*, so clicks made while the request was in flight survive.
    const merged = mergeSynced(syncedPart(state), remote);
    const json = JSON.stringify(merged);
    Object.assign(state, merged);
    settingsSeen = settingsJSON(state);
    saveLocal();
    syncedJSON = JSON.stringify(syncedPart(state));
    if (!sync.gistId) {
      const g = await gh("/gists", { method: "POST", body: {
        description: "biopapers sync — saves, upvotes and settings", public: false,
        files: { [GIST_FILE]: { content: json } } } });
      sync.gistId = g.id;
    } else if (json !== JSON.stringify(remote)) {
      await gh(`/gists/${sync.gistId}`, { method: "PATCH", body: { files: { [GIST_FILE]: { content: json } } } });
    }
    saveSyncConfig();
    Object.assign(syncStatus, { at: Date.now(), error: null });
    if (syncedJSON !== before) { affinityCache = nearCache = null; if (DATA.generated) render({ keepPage: true }); }
  } catch (err) {
    syncStatus.error = err.message;
    syncedJSON = null; // so the next change retries the push
  } finally {
    syncStatus.busy = false;
    if (syncStatus.again) { syncStatus.again = false; scheduleSync(); }
    if ($("settings").open) renderSync();
  }
}

function renderSync() {
  const err = syncStatus.error ? `<p class="hint sync-error">${esc(syncStatus.error)}</p>` : "";
  if (!sync.token) {
    $("sync-box").innerHTML = `
      <p class="hint">Keep saves, upvotes, hides and ranking settings in a secret GitHub Gist so they follow
        you to any browser. <a href="https://github.com/settings/tokens/new?scopes=gist&description=biopapers%20sync"
        target="_blank" rel="noopener">Create a token</a> with only the <code>gist</code> scope and paste it here
        on each device. The token stays in this browser.</p>
      <div class="btnrow"><input type="password" id="sync-token" placeholder="ghp_…" autocomplete="off" aria-label="GitHub token">
        <button type="button" data-sync="connect">Connect</button></div>${err}`;
    return;
  }
  const when = syncStatus.busy ? "syncing…"
    : syncStatus.at ? `last synced ${new Date(syncStatus.at).toLocaleTimeString([], { timeStyle: "short" })}` : "not synced yet";
  const gist = sync.gistId ? `<a href="https://gist.github.com/${esc(sync.gistId)}" target="_blank" rel="noopener">secret gist</a>` : "secret gist";
  $("sync-box").innerHTML = `
    <p class="hint">Synced to your ${gist} · ${when}. Secret gists are unlisted, not private: anyone with the link can read it.</p>
    <div class="btnrow"><button type="button" data-sync="now">Sync now</button>
      <button type="button" data-sync="off">Disconnect</button></div>${err}`;
}

$("sync-box").addEventListener("click", async (e) => {
  const act = e.target.closest("[data-sync]")?.dataset.sync;
  if (act === "connect") {
    const token = $("sync-token").value.trim();
    if (!token) return;
    sync = { token }; syncStatus.error = null; renderSync();
    await syncNow();
    if (syncStatus.error) { sync = {}; saveSyncConfig(); renderSync(); }
    else renderSettings();
  }
  if (act === "now") syncNow().then(renderSync);
  if (act === "off") { sync = {}; syncedJSON = null; Object.assign(syncStatus, { at: null, error: null }); saveSyncConfig(); renderSync(); }
});

// Pick up changes from other devices when you come back to the tab.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && sync.token && Date.now() - (syncStatus.at || 0) > 60000) syncNow();
});
if (sync.token) syncNow();

load().catch((err) => {
  $("sub").textContent = "couldn't load papers";
  $("feed").innerHTML = `<div class="empty">No data yet — run <code>python -m pipeline.build</code>.<br><small>${esc(err.message)}</small></div>`;
});
