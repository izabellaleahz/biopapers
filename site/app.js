// biopapers — static client. Loads data/papers.json (+ older.json and the embeddings in the
// background, ml.json when the ML / CS section is opened, map.json + favorites.json for the
// Graph), ranks and renders the feed.
// Saves / upvotes / favorites / PIs / settings live in localStorage, synced to your account (or a secret
// GitHub Gist). Signed out, the site is just what's trending: no personal tabs, ranking or scores.

"use strict";

const PAGE = 40;
const STORE_KEY = "biopapers:v1";
const SYNC_KEY = "biopapers:sync"; // { token, gistId } — kept apart so Export JSON never contains the token
const SYNC_MAPS = ["votes", "saved", "hidden", "favorites", "summaries", "clicks", "explored", "pis"];
const SYNC_SETTINGS = ["weights", "topicWeights", "halfLife", "tabOrder", "hiddenTabs", "learn"];
const DAY = 86400000;
const MUST_READ = 6; // AI score (0–10) at which a paper lands in "Must read"
const NEAR_K = 25; // each favorite pulls its 25 most similar papers into "Near favorites"

// Fixed tabs around each section's topic tabs (from config/topics.yaml).
const SPECIAL_TABS = {
  foryou: { label: "For You", color: "var(--ink)" },
  pis: { label: "PI Watch", color: "#a98fd6" },
  buzz: { label: "Buzz", color: "var(--hot)" },
  mlforyou: { label: "For You", color: "var(--ink)" },
  trending: { label: "Trending", color: "var(--hot)" },
  saved: { label: "Saved", color: "#f2b15c" },
};
const SECTION_TABS = {
  bio: { head: ["foryou"], tail: ["pis", "buzz", "saved"] },
  ml: { head: ["mlforyou", "trending"], tail: ["pis", "saved"] },
};
// Signed out, each section shows only what's trending in it.
const PUBLIC_TABS = { bio: "buzz", ml: "trending" };
const HOT_TABS = new Set(["buzz", "trending"]);

// The ranking features, in the order features() returns them; `def` is the starting weight.
const FEATURES = {
  ai: { label: "Claude's relevance score", def: 1 },
  fav: { label: "Near one of my ♥ favorites", def: 1 },
  saved: { label: "Like papers I saved", def: 0.6 },
  hidden: { label: "Like papers I marked not interested", def: -0.6, min: -2, max: 0 },
  recency: { label: "New", def: 1 },
  buzz: { label: "Buzz (HF upvotes · HN · Bluesky · press)", def: 0.6 },
  pi: { label: "From a watched PI", def: 1 },
  journal: { label: "Peer-reviewed (in a journal)", def: 0.3 },
};

const SORTS = { foryou: "For You", date: "Newest", buzz: "Buzz" };

// ---- state ---------------------------------------------------------------

const defaults = () => ({
  votes: {}, saved: {}, hidden: {}, favorites: {}, summaries: {}, explored: {}, pis: {}, removed: {}, settingsAt: 0,
  weights: Object.fromEntries(Object.entries(FEATURES).map(([k, w]) => [k, w.def])), learn: true, clicks: {},
  topicWeights: {}, halfLife: 7, tabOrder: [], hiddenTabs: [],
  section: "bio", tabBy: { bio: "foryou", ml: "mlforyou" }, sortByTab: {},
  lastVisit: null, prevVisit: null, view: "feed", graphSection: "all", graphAround: true,
});

// Who's signed in (see the accounts section). Each signed-in user has their own copy of the state in
// this browser; signed out uses the plain key.
const account = { user: null, profile: null, error: null };
const uid = () => account.user?.id;
const storeKey = () => (account.user ? `${STORE_KEY}:${account.user.id}` : STORE_KEY);

function loadState() {
  try {
    const raw = localStorage.getItem(storeKey());
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
  try { localStorage.setItem(storeKey(), JSON.stringify(state)); } catch { /* private mode etc. */ }
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
let ui = { shown: PAGE, query: "", newOnly: false, pi: null, forYou: null, // For You chip: null | "must" | "near" | "label:…"
  source: null, // For You source chip: null | "journals" | "preprints"
  lastCount: 0,
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
  mlLoaded ||= fetchPapers("ml.json").then((list) => { addPapers(list); migrateVotes(); render({ keepPage: true }); });
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
  maybeOnboard();
  if (state.section === "ml") await loadML();
  addPapers(await fetchPapers("older.json"));
  pruneLogs();
  migrateVotes();
  render({ keepPage: true });
  loadVectors();
}

// Clicks and exploration records only matter for recent ranking: drop them after 90 days.
function pruneLogs() {
  const cut = Date.now() - 90 * DAY;
  let changed = false;
  for (const m of ["clicks", "explored"]) {
    for (const [id, v] of Object.entries(state[m])) if (entryTs(v) < cut) { delete state[m][id]; changed = true; }
  }
  if (changed) persist();
}

let vecLoading = null;
function loadVectors() { vecLoading ||= readVectors(); return vecLoading; }
async function readVectors() {
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
    VEC = { dim, data, norms, ids: [...meta.ids], index: new Map(meta.ids.map((id, i) => [id, i])) };
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

// Upvotes were a lab-feed idea; for one person they meant "save". Fold each into a save once its paper
// is loaded (old ones whose paper has aged out stay as a quiet ranking signal).
function migrateVotes() {
  const ids = Object.keys(state.votes).filter((id) => byId.has(id));
  if (!ids.length) return;
  for (const id of ids) {
    const p = byId.get(id);
    if (!has(state.saved, p)) state.saved[id] = { ts: state.votes[id] || Date.now(), paper: snapshot(p) };
  }
  forget("votes", ids);
  persist();
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
// For You is ranked by a small model of what you like. Each paper gets a few features in 0..1:
// Claude's relevance score; how close it is to your nearest ♥ favorite, nearest saved paper and nearest
// paper you marked not interested (nearest, not the average, so separate interests don't blur into
// one); recency; buzz; a watched PI. Its score is a weighted sum. The weights start at the sliders in
// ⚙︎ and are then fitted to what you actually do (♥, ☆, take-homes and clicking through, against ✕
// and a sample of papers you scrolled past) by logistic regression pulled toward those starting
// values, so a few clicks can't swing it. A paper never counts as similar to itself, so a saved paper
// isn't scored "like your saves" because of itself, which keeps the fit honest.

const MAX_REFS = 150;      // the most recent saves / hides that count as references
const MIN_POSITIVES = 8;   // below this, rank with the starting weights as they are
const PRIOR_PULL = 0.03;   // L2 pull toward the starting weights

// Max cosine from every embedded paper to a set of reference papers (the paper itself excluded),
// with which reference was nearest. Adding a reference updates it in place (one pass over the vectors).
class NearestSet {
  constructor() { this.ids = new Set(); this.max = null; this.arg = null; this.n = -1; }
  sync(ids) {
    const n = VEC.index.size, want = new Set(ids.filter((id) => VEC.index.has(id)));
    if (this.n !== n || [...this.ids].some((id) => !want.has(id))) {
      this.ids = new Set(); this.n = n;
      this.max = new Float32Array(n).fill(-1); this.arg = new Int32Array(n).fill(-1);
    }
    for (const id of want) if (!this.ids.has(id)) this.add(id);
  }
  add(id) {
    const { dim, data, norms } = VEC, r = VEC.index.get(id), base = r * dim;
    for (let i = 0; i < this.n; i++) {
      if (i === r) continue;
      let dot = 0; const b = i * dim;
      for (let j = 0; j < dim; j++) dot += data[base + j] * data[b + j];
      const s = dot / (norms[r] * norms[i]);
      if (s > this.max[i]) { this.max[i] = s; this.arg[i] = r; }
    }
    this.ids.add(id);
  }
  // Raw cosines bunch up between ~0.6 and 0.95: stretch so the median paper is 0 and the top 0.5% is 1.
  scale() {
    if (!this.ids.size) return null;
    const vals = []; const step = Math.max(1, Math.floor(this.n / 3000));
    for (let i = 0; i < this.n; i += step) if (this.max[i] > -1) vals.push(this.max[i]);
    vals.sort((a, b) => a - b);
    return { lo: vals[Math.floor(vals.length * 0.5)] ?? 0, hi: vals[Math.floor(vals.length * 0.995)] ?? 1 };
  }
}
const REF = { fav: new NearestSet(), saved: new NearestSet(), hidden: new NearestSet() };

const recentIds = (map, n = MAX_REFS) => Object.entries(map).sort((a, b) => entryTs(b[1]) - entryTs(a[1])).slice(0, n).map(([id]) => id);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const hash32 = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; };

function features(p, rk, now) {
  const r = VEC ? vecRow(p) : undefined;
  const sim = (k) => (r === undefined || !rk.scale[k] || REF[k].max[r] < 0) ? 0
    : clamp01((REF[k].max[r] - rk.scale[k].lo) / ((rk.scale[k].hi - rk.scale[k].lo) || 1));
  const age = Math.max(0, (now - Date.parse(p.date)) / DAY);
  return [
    (p.ai?.score ?? 3) / 10,               // ai (unscored papers count as a middling 3)
    sim("fav"), sim("saved"), sim("hidden"),
    Math.pow(0.5, age / state.halfLife),   // recency
    Math.min(1, Math.log1p(p.buzz_score || 0) / Math.log1p(20)),  // buzz
    myPis(p).length ? 1 : 0,               // pi
    p.kind === "article" ? 1 : 0,          // journal (vs preprint)
  ];
}

// Your signals: positives with a weight for how strong a sign each is, and negatives — papers you hid,
// plus a fixed sample of papers you saw nothing in (3 per positive, at low weight).
function trainingSet() {
  const pos = new Map(), add = (ids, w) => { for (const id of ids) if (byId.has(id)) pos.set(id, Math.max(pos.get(id) || 0, w)); };
  add(Object.keys(state.favorites), 3);
  add(Object.keys(state.saved), 2);
  add(Object.keys(state.summaries), 2);
  add(Object.keys(state.clicks), 1);
  const neg = new Map(Object.keys(state.hidden)
    .filter((id) => byId.has(id) && !pos.has(id) && hideReason(id) !== "read").map((id) => [id, 2]));
  const unlabeled = DATA.papers.filter((p) => !pos.has(p.id) && !neg.has(p.id) && !has(state.hidden, p))
    .sort((a, b) => hash32(a.id) - hash32(b.id)).slice(0, Math.max(60, 3 * pos.size));
  for (const p of unlabeled) neg.set(p.id, 0.5);
  return { pos, neg };
}

function fitModel(rk) {
  const prior = Object.keys(FEATURES).map((k) => state.weights[k] ?? FEATURES[k].def);
  const { pos, neg } = trainingSet();
  const stats = { positives: pos.size, hides: Object.keys(state.hidden).filter((id) => byId.has(id)).length };
  if (!state.learn || pos.size < MIN_POSITIVES) return { w: prior, prior, learned: false, ...stats };
  const now = Date.now(), X = [], y = [], wt = [];
  for (const [id, w] of pos) { X.push(features(byId.get(id), rk, now)); y.push(1); wt.push(w); }
  for (const [id, w] of neg) { X.push(features(byId.get(id), rk, now)); y.push(0); wt.push(w); }
  const k = prior.length, total = wt.reduce((a, b) => a + b, 0);
  const w = [...prior]; let b = 0;
  for (let it = 0; it < 400; it++) {  // batch gradient descent; tiny problem
    const g = new Array(k).fill(0); let gb = 0;
    for (let i = 0; i < X.length; i++) {
      let z = b; for (let j = 0; j < k; j++) z += w[j] * X[i][j];
      const err = (1 / (1 + Math.exp(-z)) - y[i]) * wt[i] / total;
      for (let j = 0; j < k; j++) g[j] += err * X[i][j];
      gb += err;
    }
    for (let j = 0; j < k; j++) w[j] -= 1.0 * (g[j] + 2 * PRIOR_PULL * (w[j] - prior[j]));
    b -= 1.0 * gb;
  }
  return { w, prior, learned: true, ...stats };
}

let affinityCache = null; // the ranker: { scale, w, … }; rebuilt when the corpus, vectors or your signals change
let nearCache = null; // Map(id -> { fav, sim }) for "Near favorites", rebuilt alongside it

function ranker() {
  if (affinityCache) return affinityCache;
  if (VEC) {
    REF.fav.sync(Object.keys(state.favorites));
    REF.saved.sync(recentIds(state.saved));
    // Only a plain ✕ makes similar papers look worse: "not my area" works through the region instead,
    // and "already read" / "low quality" say nothing about the topic.
    REF.hidden.sync(recentIds(Object.fromEntries(Object.entries(state.hidden).filter(([id]) => !hideReason(id)))));
  }
  const favRegions = favoriteRegions();
  const rk = { scale: VEC ? { fav: REF.fav.scale(), saved: REF.saved.scale(), hidden: REF.hidden.scale() } : {},
               regionDown: regionDownweights(), regionUp: regionUpweights(favRegions), favRegions };
  affinityCache = Object.assign(rk, fitModel(rk));
  return affinityCache;
}

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const rankScore = (p, rk, now) => topicWeight(p) * (rk.regionDown.get(p.region) ?? 1) * (rk.regionUp.get(p.region) ?? 1)
  * dot(rk.w, features(p, rk, now));

const hideReason = (id) => state.hidden[id]?.reason || null;
// "Not my area" turns a whole map region down: x0.75 per such hide, to a floor of 0.4. It's counted
// from the hides themselves, so it keeps working after those papers leave the 30-day window.
function regionDownweights() {
  const n = new Map();
  for (const h of Object.values(state.hidden)) if (h?.reason === "area" && h.r != null) n.set(h.r, (n.get(h.r) || 0) + 1);
  return new Map([...n].map(([r, k]) => [r, Math.max(0.4, 0.75 ** k)]));
}

// Favorites calibrate the map's neighborhoods the other way: the region each ♥ sits in ranks a little
// higher, x1.1 per favorite there, up to x1.3. A favorite outside the window has no region of its own,
// so it takes the commonest one among its 8 nearest papers (weighted by similarity).
const FAV_REGION_UP = 0.1, FAV_REGION_MAX = 1.3;
function favoriteRegions() {
  const out = new Map();
  let withRegion = null;  // [vector row, region] for every paper that has a region
  for (const id of Object.keys(state.favorites)) {
    const own = byId.get(id)?.region;
    if (own != null) { out.set(id, own); continue; }
    const f = VEC ? vecRow({ id, aliases: favPaper(id)?.aliases }) : undefined;
    if (f === undefined) continue;
    withRegion ||= DATA.papers.filter((p) => p.region != null).map((p) => [vecRow(p), p.region]).filter(([r]) => r !== undefined);
    const { dim, data, norms } = VEC, sims = [];
    for (const [r, region] of withRegion) {
      if (r === f) continue;
      let d = 0; for (let j = 0; j < dim; j++) d += data[f * dim + j] * data[r * dim + j];
      sims.push([d / (norms[f] * norms[r]), region]);
    }
    const votes = new Map();
    for (const [sim, region] of sims.sort((a, b) => b[0] - a[0]).slice(0, 8)) votes.set(region, (votes.get(region) || 0) + sim);
    const top = [...votes].sort((a, b) => b[1] - a[1])[0];
    if (top) out.set(id, top[0]);
  }
  return out;
}
function regionUpweights(favRegions) {
  const n = new Map();
  for (const r of favRegions.values()) n.set(r, (n.get(r) || 0) + 1);
  return new Map([...n].map(([r, k]) => [r, Math.min(FAV_REGION_MAX, 1 + FAV_REGION_UP * k)]));
}

// Re-rank the top of the list, in three steps:
// 1. Dither: each paper's rank gets a small random nudge on a log scale (seeded by day and paper),
//    so the top barely moves, the tail reshuffles, and the order is stable within a day.
// 2. Spread: near-duplicates are pushed apart, and no map region takes more than REGION_CAP of the
//    top 20.
// 3. Explore: every 7th slot from the 5th (~15%) goes to a good paper from a region picked by a
//    per-region bandit, so regions you've never been shown still come up. Each region keeps a
//    Beta(1 + engaged, 1 + 0.2·shown + hidden) estimate from papers it explored for you.
const HEAD = 80, DUP_SIM = 0.8, REGION_CAP = 4, DITHER = Math.log(1.3);
let exploring = new Map(); // paper id -> region it was explored from

// Seeded randomness: the same day, user and paper always give the same draw.
const unit = (key) => (hash32(key) + 0.5) / 4294967296;
const gauss = (key) => Math.sqrt(-2 * Math.log(unit(key))) * Math.cos(2 * Math.PI * unit(key + "#"));
function gammaSample(a, key) {  // Marsaglia–Tsang, for a >= 1
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (let i = 0; i < 20; i++) {
    const x = gauss(`${key}:x${i}`), v = (1 + c * x) ** 3;
    if (v > 0 && Math.log(unit(`${key}:u${i}`)) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
  return d;
}
const betaSample = (a, b, key) => { const x = gammaSample(a, key + "a"); return x / (x + gammaSample(b, key + "b")); };
const today = () => new Date().toISOString().slice(0, 10);
const userKey = () => "local"; // per-user once accounts exist

function regionBandit(seed) {
  const stats = new Map(), bump = (r, k, by = 1) => {
    const s = stats.get(r) || { engaged: 0, shown: 0, hidden: 0 }; s[k] += by; stats.set(r, s);
  };
  const positive = (id) => state.favorites[id] || state.saved[id] || state.summaries[id] || state.clicks[id];
  for (const [id, e] of Object.entries(state.explored)) {
    bump(e.r, "shown");
    if (positive(id)) bump(e.r, "engaged");
    else if (state.hidden[id] && hideReason(id) !== "read") bump(e.r, "hidden");
  }
  return (r, slot) => { const s = stats.get(r) || { engaged: 0, shown: 0, hidden: 0 };
    return betaSample(1 + s.engaged, 1 + 0.2 * s.shown + s.hidden, `${seed}:slot${slot}:region${r}`); };
}

function diversify(scored, rk) {
  exploring = new Map();
  const seed = `${userKey()}:${today()}`;
  scored = scored.map(([, p], i) => [-(Math.log(i + 1) - DITHER * gauss(`${seed}:${p.id}`)), p])
    .sort((a, b) => b[0] - a[0]);
  if (!VEC) return scored.map(([, p]) => p);
  const head = scored.slice(0, HEAD), rest = scored.slice(HEAD);
  const span = (head[0]?.[0] ?? 0) - (head[head.length - 1]?.[0] ?? 0) || 1;
  const rowOf = new Map(head.map(([, p]) => [p.id, vecRow(p)]));
  const cos = (a, b) => {
    const ra = rowOf.get(a.id) ?? vecRow(a), rb = rowOf.get(b.id) ?? vecRow(b);
    if (ra === undefined || rb === undefined) return 0;
    const { dim, data, norms } = VEC; let d = 0;
    for (let j = 0; j < dim; j++) d += data[ra * dim + j] * data[rb * dim + j];
    return d / (norms[ra] * norms[rb]);
  };
  // Exploration candidates: the best-ranked decent paper of each region. Decent means Claude rated it
  // 5+ (when scored), or it's a journal paper or has buzz (when not).
  const decent = (p) => typeof p.ai?.score === "number" ? p.ai.score >= 5 : (p.kind === "article" || p.buzz_score > 0);
  const byRegion = new Map();
  for (const [, p] of scored) if (p.region != null && decent(p) && !byRegion.has(p.region)) byRegion.set(p.region, []);
  for (const [, p] of scored) if (p.region != null && decent(p) && byRegion.get(p.region).length < 5) byRegion.get(p.region).push(p);
  const sample = regionBandit(seed);
  const regionCount = new Map(), picked = [], left = [...head], used = new Set();
  const take = (p) => { picked.push(p); used.add(p.id); regionCount.set(p.region, (regionCount.get(p.region) || 0) + 1); };
  while (left.length) {
    if (picked.length >= 4 && (picked.length - 4) % 7 === 0 && byRegion.size) {
      const choices = [...byRegion.entries()].map(([r, ps]) => [r, ps.find((p) => !used.has(p.id))]).filter(([, p]) => p);
      const [r, ex] = choices.map(([r, p]) => [r, p, sample(r, picked.length)]).sort((a, b) => b[2] - a[2])[0] || [];
      if (ex) { take(ex); exploring.set(ex.id, r); continue; }
    }
    const inTop20 = picked.length < 20;
    let best = -1, bestV = -Infinity;
    for (let i = 0; i < left.length; i++) {
      const [s, p] = left[i];
      if (used.has(p.id)) continue;
      if (inTop20 && p.region != null && (regionCount.get(p.region) || 0) >= REGION_CAP) continue;
      const near = picked.slice(-12).reduce((m, q) => Math.max(m, cos(p, q)), 0);  // the recent few is enough
      const v = s - (near > DUP_SIM ? span * 0.5 * (near - DUP_SIM) / (1 - DUP_SIM) : 0);
      if (v > bestV) { bestV = v; best = i; }
    }
    if (best < 0) {  // only capped regions left: relax the cap
      const i = left.findIndex(([, p]) => !used.has(p.id));
      if (i < 0) break;
      take(left.splice(i, 1)[0][1]);
      continue;
    }
    take(left.splice(best, 1)[0][1]);
  }
  logExplored(picked.slice(0, 40));
  return [...picked, ...rest.map(([, p]) => p).filter((p) => !used.has(p.id))];
}

// Remember which explored papers you were actually shown, for the bandit (kept 90 days).
function logExplored(top) {
  let added = false;
  for (const p of top) {
    if (exploring.has(p.id) && !state.explored[p.id]) { state.explored[p.id] = { ts: Date.now(), r: p.region }; added = true; }
  }
  if (added) setTimeout(persist, 0);  // not mid-render
}

// "Why it's here", from the two features that contribute most to this paper's score.
function whyLine(p) {
  if (exploring.has(p.id)) return `🧭 exploring ${DATA.regions?.[exploring.get(p.id)] || "something new"}${typeof p.ai?.score === "number" ? ` · AI ${p.ai.score}` : ""}`;
  const rk = affinityCache;
  if (!rk) return "";
  const x = features(p, rk, Date.now()), r = VEC ? vecRow(p) : undefined;
  const nearest = (k) => (r === undefined || REF[k].arg[r] < 0) ? null : VEC.ids[REF[k].arg[r]];
  const say = {
    0: () => `AI ${p.ai?.score}`,
    1: () => { const f = nearest("fav"); return f ? `near ♥ ${favLabel(f)}` : ""; },
    2: () => { const s = nearest("saved"); return s ? `like “${shortTitle(favPaper(s)?.title || state.saved[s]?.paper?.title, 4)}” you saved` : ""; },
    4: () => { const d = Math.round(Math.max(0, (Date.now() - Date.parse(p.date)) / DAY)); return d < 1 ? "today" : `${d}d old`; },
    5: () => "buzzing",
    6: () => `PI · ${myPis(p)[0]}`,
    7: () => `in ${p.venue}`,
  };
  return x.map((v, j) => [rk.w[j] * v, j]).filter(([c, j]) => c > 0.15 && say[j])
    .sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, j]) => say[j]()).filter(Boolean).join(" · ");
}

// How well the ranking fits you: where your recent saves land in For You, among everything there.
function rankFit(rk) {
  const now = Date.now(), cut = now - 30 * DAY;
  const recent = Object.entries(state.saved).filter(([id, s]) => entryTs(s) > cut && byId.has(id)).map(([id]) => byId.get(id));
  if (recent.length < 3) return null;
  const pool = DATA.papers.filter((p) => (sectionTopics(p).length || myPis(p).length || (p.ai?.score ?? 0) >= 4) && !has(state.hidden, p));
  const scores = pool.map((p) => rankScore(p, rk, now)).sort((a, b) => b - a);
  const top20 = scores[Math.min(19, scores.length - 1)];
  return { n: recent.length, inTop20: recent.filter((p) => rankScore(p, rk, now) >= top20).length };
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

// ---- tabs / filtering -----------------------------------------------------------

function allTabs() {
  if (!account.user) return [PUBLIC_TABS[state.section]];
  const { head, tail } = SECTION_TABS[state.section];
  const topicTabs = DATA.topics.filter((t) => (t.section || "bio") === state.section).map((t) => t.slug);
  const known = [...head, ...topicTabs, ...tail];
  const order = state.tabOrder.filter((t) => known.includes(t));
  return [...order, ...known.filter((t) => !order.includes(t))];
}

function tabInfo(slug) {
  if (!account.user && slug === PUBLIC_TABS[state.section]) return { ...SPECIAL_TABS[slug], label: "Trending" };
  return SPECIAL_TABS[slug] || DATA.topics.find((t) => t.slug === slug) || { label: slug, color: "var(--muted)" };
}

function inTab(p, tab) {
  if (!(p.sections || ["bio"]).includes(state.section)) return false;
  // Journal papers that fit no topic tab still belong in For You when Claude rates them relevant.
  if (tab === "foryou") return (sectionTopics(p).length || myPis(p).length || (p.ai?.score ?? 0) >= 4) && topicWeight(p) > 0;
  if (tab === "mlforyou") return topicWeight(p) > 0;
  if (tab === "pis") return myPis(p).length > 0;
  if (HOT_TABS.has(tab)) return (p.buzz_score || 0) > 0;
  return p.topics?.includes(tab);
}

// Your watched PIs among a paper's authors. Matched like the build does (pipeline/tagging.py): last
// name and first initial, accent-insensitive; the build's own `pis` tags count too, because long author
// lists are cut down before they ship and a PI in the middle survives only as that tag.
function nameParts(name) {
  const parts = String(name).normalize("NFKD").replace(/[^\x00-\x7f]/g, "").toLowerCase().split(/[\s.]+/).filter(Boolean);
  if (!parts.length) return ["", ""];
  if (parts.length >= 2 && parts.at(-1).length === 1) return [parts.at(-1), parts[0]]; // "Uhler C"
  return [parts[0], parts.at(-1)];
}
let piCache = { sig: null };
function myPis(p) {
  const sig = Object.keys(state.pis).join("|");
  if (piCache.sig !== sig) {
    piCache = { sig, hits: new Map(), list: Object.values(state.pis).map((x) => [x.name, ...nameParts(x.name)]) };
  }
  if (!piCache.list.length) return [];
  let hit = piCache.hits.get(p.id);
  if (!hit) {
    const authors = (p.authors || []).map(nameParts);
    hit = piCache.list.filter(([name, first, last]) => (p.pis || []).includes(name)
      || authors.some(([f, l]) => l === last && f.slice(0, 1) === first.slice(0, 1))).map(([name]) => name);
    piCache.hits.set(p.id, hit);
  }
  return hit;
}

// For You's filter chips: Must read, papers near any favorite, or near the favorites with one label.
function forYouChip(chip) {
  if (chip === "must") return (p) => (p.ai?.score ?? 0) >= MUST_READ;
  const near = nearFavorites();
  if (chip === "near") return (p) => near.has(p.id);
  const label = chip.slice("label:".length);
  return (p) => state.favorites[near.get(p.id)?.fav]?.label === label;
}

// For You's source chips: peer-reviewed journal papers, or preprints (bioRxiv, medRxiv, arXiv).
const sourceChip = (chip) => (p) => (p.kind === "article") === (chip === "journals");

function savedPapers() {
  return Object.entries(state.saved)
    .sort((a, b) => b[1].ts - a[1].ts)
    .map(([id, s]) => byId.get(id) || DATA.papers.find((p) => p.aliases?.includes(id)) || s.paper)
    .filter(Boolean);
}

const sortFor = (tab) => {
  const sort = state.sortByTab[tab] || (HOT_TABS.has(tab) ? "buzz" : "foryou");
  return !account.user && sort === "foryou" ? "buzz" : sort;  // signed out there's no For You ranking
};

function visiblePapers() {
  const tab = curTab();
  let list = tab === "saved" ? savedPapers() : DATA.papers.filter((p) => inTab(p, tab) && !has(state.hidden, p));
  if (tab === "pis" && ui.pi) list = list.filter((p) => myPis(p).includes(ui.pi));
  if ((tab === "foryou" || tab === "mlforyou") && ui.forYou) list = list.filter(forYouChip(ui.forYou));
  if ((tab === "foryou" || tab === "mlforyou") && ui.source) list = list.filter(sourceChip(ui.source));
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
  const rk = ranker(), now = Date.now();
  const scored = list.map((p) => [rankScore(p, rk, now), p]).sort((a, b) => b[0] - a[0]);
  return ui.query ? scored.map(([, p]) => p) : diversify(scored, rk);
}

// ---- rendering ---------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function render({ keepPage = false } = {}) {
  if (!keepPage) ui.shown = PAGE;
  const pub = !account.user;  // signed out: just what's trending
  const tabs = allTabs().filter((t) => pub || !state.hiddenTabs.includes(t) || t === curTab());
  if (!tabs.includes(curTab())) setTab(pub ? tabs[0] : SECTION_TABS[state.section].head[0]);
  const tab = curTab();

  const graph = !pub && state.view === "graph";
  document.body.classList.toggle("signed-out", pub);
  $("sections").innerHTML = Object.entries(DATA.sections || {}).map(([k, label]) =>
    `<button data-section="${k}" class="${!graph && k === state.section ? "active" : ""}">${esc(label)}</button>`).join("")
    + (pub ? "" : `<button data-view="graph" class="${graph ? "active" : ""}" title="Every paper laid out by similarity, centered on your ♥ favorites">Graph</button>`);

  const updated = DATA.generated ? new Date(DATA.generated).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";
  const inSection = DATA.papers.filter((p) => (p.sections || ["bio"]).includes(state.section)).length;
  const loading = state.section === "ml" && !inSection ? "loading ML papers…" : `${inSection.toLocaleString()} papers`;
  $("sub").textContent = `${loading} · updated ${updated}`;
  document.body.classList.toggle("graph-mode", graph);
  if (account.pending) {  // a stored session is being restored: don't flash the signed-out page
    for (const id of ["tabs", "sortbar", "chips"]) $(id).innerHTML = "";
    $("feed").innerHTML = `<div class="empty">Signing you in…</div>`;
    $("more").hidden = true;
    return;
  }
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
  const sortLabel = (k, label) => k === "buzz" && (state.section === "ml" || pub) ? "Hot" : label;
  $("sortbar").innerHTML = tab === "saved" ? `<button class="active">Recently saved</button>`
    : Object.entries(SORTS).filter(([k]) => !pub || k !== "foryou").map(([k, label]) =>
        `<button data-sort="${k}" class="${k === sort ? "active" : ""}">${sortLabel(k, label)}</button>`).join("");

  const nNew = DATA.papers.filter((p) => isNew(p) && inTab(p, tab === "saved" ? SECTION_TABS[state.section].head[0] : tab)).length;
  $("new-label").textContent = prevVisitDay ? `new since last visit (${nNew})` : "new since last visit";

  const myPiList = Object.values(state.pis).sort((a, b) => a.name.localeCompare(b.name));
  $("chips").innerHTML = tab === "pis" && myPiList.length
    ? [null, ...myPiList].map((x) =>
        `<button class="chip ${ui.pi === (x?.name ?? null) ? "active" : ""}" data-pi="${esc(x?.name ?? "")}"
          title="${esc(x?.inst || "")}">${esc(x?.name ?? "All PIs")}</button>`).join("")
    : tab === "foryou" || tab === "mlforyou"
    ? (() => {
        const base = DATA.papers.filter((p) => inTab(p, tab) && visible(p));
        const labels = [...new Set(Object.values(state.favorites).map((f) => f.label).filter(Boolean))];
        const chips = [[null, "All"], ["must", "Must read"],
          ...(Object.keys(state.favorites).length ? [["near", "♥ Near favorites"]] : []),
          ...labels.map((l) => [`label:${l}`, `♥ ${l}`])];
        const sources = [[null, "All sources"], ["journals", "Journals"], ["preprints", "Preprints"]];
        return chips.map(([k, label]) => `<button class="chip ${ui.forYou === k ? "active" : ""}" data-foryou="${esc(k ?? "")}">${esc(label)}${
          k ? ` <span class="n">${base.filter(forYouChip(k)).length}</span>` : ""}</button>`).join("")
          + `<span class="chip-sep" aria-hidden="true"></span>`
          + sources.map(([k, label]) => `<button class="chip ${ui.source === k ? "active" : ""}" data-source="${k ?? ""}">${label}${
            k ? ` <span class="n">${base.filter(sourceChip(k)).length}</span>` : ""}</button>`).join("");
      })()
    : "";

  const list = visiblePapers();
  ui.lastCount = list.length;
  $("feed").innerHTML = list.length
    ? list.slice(0, ui.shown).map(card).join("")
    : `<div class="empty">${tab === "saved" ? "Nothing saved yet — hit ☆ on a paper."
      : tab === "pis" && !Object.keys(state.pis).length ? "You aren't watching any PIs yet. Add some in ⚙︎ Settings → PI watchlist."
      : ui.forYou === "must" ? "Nothing in Must read yet. It fills in as the daily AI scoring runs."
      : ui.forYou ? "Nothing near your favorites yet (they need the embeddings, which load a few seconds after the page)."
      : "No papers match."}</div>`;
  $("more").hidden = list.length <= ui.shown;
  $("more").textContent = `Show more (${list.length - ui.shown} left)`;
}

const fmtK = (n) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
const link = (href, text, title = "") =>
  `<a href="${esc(href)}" target="_blank" rel="noopener"${title ? ` title="${esc(title)}"` : ""}>${text}</a>`;

const showWhy = (p) => state.view !== "graph" && curTab() !== "saved" && sortFor(curTab()) === "foryou" && !!whyLine(p);

function card(p) {
  // Current section's topics first, so the card edge matches the tab you're in.
  const topics = (p.topics || []).map((t) => DATA.topics.find((x) => x.slug === t)).filter(Boolean)
    .sort((a, b) => ((b.section || "bio") === state.section) - ((a.section || "bio") === state.section));
  const pis = myPis(p);
  const color = topics[0]?.color || (pis.length ? "#a98fd6" : (p.buzz_score ? "var(--hot)" : "var(--muted)"));
  const b = p.buzz || {}, hf = p.hf || {};
  const arxivId = p.doi?.startsWith("10.48550/arxiv.") ? p.doi.slice("10.48550/arxiv.".length) : null;
  const lastNames = new Set(pis.map((n) => nameParts(n)[1]));
  const authors = (p.authors || []).map((a) =>
    lastNames.has(nameParts(a)[1]) ? `<b>${esc(a)}</b>` : esc(a)).join(", ");
  const pub = !account.user, saved = has(state.saved, p), fav = has(state.favorites, p);
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
    ${pub ? "" : `<div class="actions">
      <button class="${saved ? "on" : ""}" data-act="save" aria-label="Save" title="Save">${saved ? "★" : "☆"}</button>
      <button class="${fav ? "on fav" : ""}" data-act="fav" aria-label="Favorite" title="Favorite: a landmark paper the Graph and Near favorites center on">♥</button>
      <button class="hide" data-act="hide" aria-label="${curTab() === "saved" ? "Unsave" : "Not interested"}" title="${curTab() === "saved" ? "Unsave" : "Not interested: hide it, and rank papers like it lower"}">✕</button>
    </div>`}
    <div class="body">
      <h3><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
      <div class="meta">
        ${isNew(p) ? `<span class="badge new">new</span>` : ""}
        <span class="badge venue">${esc(p.venue)}${p.kind === "preprint" && !/rxiv/i.test(p.venue) ? " · preprint" : ""}</span>
        ${hf.org ? `<span class="badge org">${esc(hf.org)}</span>` : ""}
        <span class="date">${esc(p.date)}</span>
        ${pis.map((n) => `<span class="badge pi">PI · ${esc(n)}</span>`).join("")}
        ${near ? `<span class="badge near" title="Near your favorite: ${esc(favPaper(near.fav)?.title)}">♥ ${esc(favLabel(near.fav))}</span>` : ""}
        ${!pub && typeof p.ai?.score === "number" ? `<span class="badge ai ${p.ai.score >= MUST_READ ? "hi" : ""}" title="Claude's relevance score for you: ${esc(p.ai.why)}">AI ${p.ai.score}</span>` : ""}
        ${hot}
        ${topics.map((t) => `<span class="badge" style="--c:${t.color}"${p.tagged_by ? ` title="Tagged from similar papers (no keyword matched)"` : ""}>${esc(t.label)}</span>`).join("")}
      </div>
      <div class="authors" title="${esc((p.authors || []).join(", "))}">${authors}${p.n_authors ? ` <span>(${p.n_authors})</span>` : ""}</div>
      ${showWhy(p) ? `<div class="why-line">${esc(whyLine(p))}</div>` : ""}
      ${p.ai?.tldr || hf.tldr ? `<div class="tldr"><b>TL;DR</b> ${esc(p.ai?.tldr || hf.tldr)}${!pub && p.ai?.why ? ` <span class="why">· ${esc(p.ai.why)}</span>` : ""}</div>` : ""}
      ${p.abstract ? `<div class="abstract">${esc(p.abstract)}</div>` : ""}
      ${takehomeHTML(p)}
      <div class="links">${links.join("")}${pub ? "" : `<button type="button" class="linkbtn th-btn" data-act="takehome" title="A short take-home summary, written on demand">${state.summaries[p.id] ? (ui.takehomeOpen.has(p.id) ? "hide take-home" : "take-home ✓") : "take-home"}</button><button type="button" class="linkbtn" data-act="map" title="Show it on the Graph with its nearest neighbours">similar papers</button>`}</div>
    </div>
  </article>`;
}

// ---- ✕ toast: why not? ---------------------------------------------------------------
// ✕ hides a paper. The toast that follows lets you say why, which changes what it teaches the ranking:
//   not my area   a "no", and the paper's whole map region turns down (see regionDownweights)
//   already read  just hide it; nothing is learned
//   low quality   a "no" for training, without making similar papers look worse
//   (no answer)   a "no", and similar papers rank a bit lower
let toastTimer = null;
function showHideToast(p) {
  const t = $("toast"), region = DATA.regions?.[p.region];
  t.innerHTML = `<span>Hidden. Why?</span>
    <button type="button" data-why="area" title="${region ? `Show less of ${esc(region)}` : "Show less like this"}">Not my area</button>
    <button type="button" data-why="read">Already read</button>
    <button type="button" data-why="quality">Low quality</button>
    <button type="button" data-why="undo" class="undo">Undo</button>`;
  t.dataset.id = p.id; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 9000);
}
$("toast").addEventListener("click", (e) => {
  const why = e.target.closest("[data-why]")?.dataset.why, id = $("toast").dataset.id;
  if (!why || !id) return;
  if (why === "undo") forget("hidden", [id]);
  else if (state.hidden[id]) state.hidden[id] = { ...(typeof state.hidden[id] === "object" ? state.hidden[id] : {}), ts: Date.now(), reason: why };
  $("toast").hidden = true;
  affinityCache = nearCache = null; persist(); render({ keepPage: true });
});


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
  const opened = e.target.closest(".paper h3 a");
  if (opened) { state.clicks[opened.closest(".paper").dataset.id] = Date.now(); affinityCache = null; persist(); return; }
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
  if (act === "save") toggle("saved", p, { ts: Date.now(), paper: snapshot(p) });
  if (act === "hide") {
    if (curTab() === "saved") toggle("saved", p);
    else { state.hidden[p.id] = { ts: Date.now(), ...(p.region != null ? { r: p.region } : {}) }; showHideToast(p); }
  }
  affinityCache = nearCache = null; // hides count too: "not interested" pushes similar papers down
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
  const f = e.target.closest("[data-foryou]");
  if (f) { ui.forYou = f.dataset.foryou || null; render(); }
  const src = e.target.closest("[data-source]");
  if (src) { ui.source = src.dataset.source || null; render(); }
});
$("more").addEventListener("click", () => { ui.shown += PAGE; render({ keepPage: true }); });
$("new-only").addEventListener("change", (e) => { ui.newOnly = e.target.checked; render(); });
// One search box for both views: it filters the feed (or dims the graph), and from 3 characters on
// also looks the query up across the literature, with ☆ / ♥ on every result.
let searchTimer;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    ui.query = e.target.value.trim();
    if (state.view === "graph" && MAP) { for (const q of mapPts) q.dim = !matchesGraph(q.p); drawMap(); }
    else render();
    if (ui.query.length >= 3) lookup(ui.query);
    else { lookupResults = []; renderLookup(""); }
  }, 250);
});

// ---- graph -------------------------------------------------------------------------
// UMAP layout of the embeddings (pipeline/layout.py), its own view next to Bio | ML / CS. Dots take
// their topic's color; ♥ favorites are labelled hubs, and "around favorites" narrows the graph to
// them plus the papers closest to each. Saved / upvoted papers get a ring. The lookup box finds
// any paper, in the feed or anywhere in the literature (OpenAlex), to ♥ it. Click a dot to see that
// paper and the ones most like it below.

const mapView = { k: 1, tx: 0, ty: 0 };
let mapSel = null, mapHover = null, mapFocus = null, mapPts = [], mapSize = { w: 0, h: 0, s: 0 };
let mapAllLinks = false, mapNeedsFit = true, lookupResults = [], mapRegions = true, regionCounts = [];
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
const xyBase = (x, y) => [(mapSize.w - mapSize.s) / 2 + x * mapSize.s, (mapSize.h - mapSize.s) / 2 + y * mapSize.s];
const mapBase = (i) => xyBase(MAP.xy[i][0], MAP.xy[i][1]);
const toScreen = ([x, y]) => [x * mapView.k + mapView.tx, y * mapView.k + mapView.ty];
const mapScreen = (i) => toScreen(mapBase(i));

function fitMap(indexes) {
  if (!indexes.length) { Object.assign(mapView, { k: 1, tx: 0, ty: 0 }); return; }
  const pts = indexes.map(mapBase), xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const k = Math.min(12, Math.max(1, Math.min(mapSize.w / (x1 - x0 + 80), mapSize.h / (y1 - y0 + 80))));
  Object.assign(mapView, { k, tx: mapSize.w / 2 - ((x0 + x1) / 2) * k, ty: mapSize.h / 2 - ((y0 + y1) / 2) * k });
}

function matchesGraph(p) {
  const q = ui.query.toLowerCase();
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
    mapPts.push({ i, p, fav, near: near.get(id), color: topic?.color || (myPis(p).length ? "#a98fd6" : muted),
                  liked: isLiked(p), dim: !matchesGraph(p) });
  });
  regionCounts = (MAP.regions || []).map(() => 0);
  for (const q of mapPts) if (MAP.c?.[q.i] != null) regionCounts[MAP.c[q.i]]++;
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
  if (mapRegions && !around) {
    // Region names (pipeline/layout.py), skipped where the current filter leaves the region mostly empty.
    // Biggest regions first; a label that would overlap one already drawn is skipped (zoom in to see it).
    const muted = cssVar("--muted"), drawn = [];
    ctx.textAlign = "center";
    (MAP.regions || []).map((reg, c) => [reg, c]).sort((a, b) => regionCounts[b[1]] - regionCounts[a[1]]).forEach(([reg, c]) => {
      if (!reg.label || regionCounts[c] < Math.max(8, reg.n * 0.3)) return;
      const [x, y] = toScreen(xyBase(reg.x, reg.y));
      if (x < 0 || y < 0 || x > w || y > h) return;
      const size = Math.round(10 + Math.min(4, Math.sqrt(regionCounts[c]) / 6));
      ctx.font = `600 ${size}px system-ui, sans-serif`;
      const half = ctx.measureText(reg.label).width / 2 + 4, box = [x - half, y - size, x + half, y + size];
      if (drawn.some((d) => box[0] < d[2] && box[2] > d[0] && box[1] < d[3] && box[3] > d[1])) return;
      drawn.push(box);
      ctx.globalAlpha = 0.95; ctx.lineWidth = 4; ctx.strokeStyle = card; ctx.strokeText(reg.label, x, y);
      ctx.fillStyle = muted; ctx.fillText(reg.label, x, y);
    });
  }
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
  VEC = { ...VEC, data, norms }; VEC.index.set(id, n); VEC.ids.push(id);
  MAP.ids.push(id); MAP.xy.push([x / (w || 1), y / (w || 1)]); MAP.nn.push(top.map(([, i]) => i)); MAP.index.set(id, n);
  MAP.c?.push(MAP.c[top[0][1]]);
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
      if (ob) renderObFavs();
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

const OA_WORK_FIELDS = "id,doi,title,display_name,publication_date,publication_year,type,authorships,primary_location,abstract_inverted_index";
// Papers from OpenAlex: the one a DOI / arXiv ID points to, or a title search.
async function openAlexWorks(q, ref = paperRef(q)) {
  const url = ref
    ? `https://api.openalex.org/works/doi:${encodeURIComponent(ref)}?select=${OA_WORK_FIELDS}`
    : `https://api.openalex.org/works?search=${encodeURIComponent(q)}&per-page=8&select=${OA_WORK_FIELDS}`;
  const res = await fetch(url);
  const body = res.ok ? await res.json() : null;
  return (body?.results || (body?.id ? [body] : [])).map(fromOpenAlex);
}

let lookupSeq = 0;
async function lookup(q) {
  const seq = ++lookupSeq, ref = paperRef(q), lq = q.toLowerCase(), graph = state.view === "graph";
  // On the graph, list matching papers already there too; in the feed they're the filtered list below.
  const local = !graph ? [] : ref && byId.has(ref) ? [byId.get(ref)]
    : [...byId.values()].filter((p) => p.title.toLowerCase().includes(lq)).slice(0, 6);
  lookupResults = local.map((p) => ({ p, local: true }));
  renderLookup(q.length >= 3 ? "searching the literature…" : "");
  if (q.length < 3) return;
  let remote = [];
  try { remote = await openAlexWorks(q, ref); } catch { /* offline: local results only */ }
  if (seq !== lookupSeq) return;
  const seen = new Set(lookupResults.map((r) => r.p.id));
  for (const p of remote) {
    if (seen.has(p.id) || (!graph && byId.has(p.id) && DATA.papers.includes(byId.get(p.id)))) continue; // already in the feed
    seen.add(p.id);
    lookupResults.push(byId.has(p.id) ? { p: byId.get(p.id), local: graph } : { p, local: false });
  }
  renderLookup(lookupResults.length ? "" : "No papers found.");
}

function renderLookup(status) {
  const row = (r, i) => {
    const p = r.p, fav = has(state.favorites, p), saved = has(state.saved, p), on = MAP?.index.has(p.id);
    const meta = [p.venue, (p.date || "").slice(0, 4), (p.authors || []).filter((a) => a !== "…").slice(0, 3).join(", ")].filter(Boolean).join(" · ");
    return `<li data-lk="${i}">
      <button type="button" class="heart save ${saved ? "on" : ""}" data-lkact="save" aria-label="${saved ? "Unsave" : "Save"}" title="${saved ? "Unsave" : "Save for later"}">${saved ? "★" : "☆"}</button>
      <button type="button" class="heart ${fav ? "on" : ""}" data-lkact="fav" aria-label="${fav ? "Remove favorite" : "Add to favorites"}" title="${fav ? "Remove favorite" : "Add to favorites"}">♥</button>
      <div class="lk-body"><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a><div class="lk-meta">${esc(meta)}</div></div>
      ${on && state.view === "graph" ? `<button type="button" class="lk-show" data-lkact="show">show</button>` : ""}
    </li>`;
  };
  const local = lookupResults.map((r, i) => [r, i]).filter(([r]) => r.local);
  const remote = lookupResults.map((r, i) => [r, i]).filter(([r]) => !r.local);
  $("lookup-results").hidden = ui.query.length < 3;
  const inFeed = state.view !== "graph" && ui.query
    ? `<div class="lk-status">${ui.lastCount.toLocaleString()} match${ui.lastCount === 1 ? "" : "es"} in your feed below · Esc to see them</div>` : "";
  $("lookup-results").innerHTML = inFeed + (local.length ? `<div class="lk-sec">In your feed</div><ul>${local.map(([r, i]) => row(r, i)).join("")}</ul>` : "")
    + (remote.length ? `<div class="lk-sec">From the literature · OpenAlex</div><ul>${remote.map(([r, i]) => row(r, i)).join("")}</ul>` : "")
    + (status ? `<div class="lk-status">${esc(status)}</div>` : "");
}

document.addEventListener("click", (e) => {
  // A ♥ in the results re-renders the list, detaching the clicked button: that's not an outside click.
  if (e.target.isConnected && !e.target.closest(".lookup-box")) $("lookup-results").hidden = true;
});
$("search").addEventListener("focus", () => { if (ui.query.length >= 3) $("lookup-results").hidden = false; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lookup-results").hidden = true; });
$("lookup-results").addEventListener("click", (e) => {
  const b = e.target.closest("[data-lkact]"), r = lookupResults[b?.closest("[data-lk]")?.dataset.lk];
  if (!r) return;
  if (b.dataset.lkact === "show") { mapFocus = r.p.id; render({ keepPage: true }); return; }
  if (b.dataset.lkact === "save") toggle("saved", r.p, { ts: Date.now(), paper: snapshot(r.p) });
  else toggleFavorite(r.p);
  affinityCache = nearCache = null;
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
  $("map-regions").addEventListener("change", (e) => { mapRegions = e.target.checked; drawMap(); });
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
    saveSummary(id);
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
  const rk = ranker(), fit = rk.learned ? rankFit(rk) : null;
  $("weights").innerHTML = Object.entries(FEATURES).map(([k, f], j) => `
    <label class="row">${esc(f.label)}
      <input type="range" min="${f.min ?? 0}" max="${f.max ?? 2}" step="0.1" value="${state.weights[k] ?? f.def}" data-weight="${k}">
      <output>${(+(state.weights[k] ?? f.def)).toFixed(1)}</output>
      <span class="learned" title="What it learned from your activity">${rk.learned ? `→ ${rk.w[j].toFixed(1)}` : ""}</span>
    </label>`).join("");
  $("learn-status").innerHTML = `<label><input type="checkbox" id="learn" ${state.learn ? "checked" : ""}> Learn from my activity</label> · ${
    !state.learn ? "using the starting weights as set"
    : rk.learned ? `fitted to ${rk.positives} papers you ♥ / ☆ / opened and ${rk.hides} you hid${
        fit ? ` · ${fit.inTop20} of your ${fit.n} saves from the last month rank in For You's top 20` : ""}`
    : `needs ${MIN_POSITIVES - rk.positives} more ♥ / ☆ / opened papers before it starts adjusting (${rk.positives} so far)`}`;
  $("halflife").value = String(state.halfLife);
  renderAccount();
  renderSync();
  renderTakehomeSettings();
  if (!$("pi-box").firstChild) $("pi-box").innerHTML = piEditorHTML();
  renderPiEditors();

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
  affinityCache = null;
  persist();
});
$("halflife").addEventListener("change", (e) => { state.halfLife = +e.target.value; affinityCache = null; persist(); });
$("learn-status").addEventListener("change", (e) => {
  if (e.target.id === "learn") { state.learn = e.target.checked; affinityCache = null; persist(); renderSettings(); }
});

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

// ---- PI watchlist ------------------------------------------------------------------------
// Everyone watches their own PIs (state.pis, synced with the account). Their papers fill the PI Watch
// tab and get the "watched PI" boost in For You. PIs are found with OpenAlex's author search, so each
// has an OpenAlex ID: the daily build reads everyone's IDs (pipeline/users.py) to fetch their papers from
// journals we have no feed for, and keeps those papers for six months.

const nameKey = (name) => { const [first, last] = nameParts(name); return `${first.slice(0, 1)} ${last}`; };
let piFind = { results: [], status: "" }, piSeq = 0, piTimer = null;

async function findAuthors(q) {
  const seq = ++piSeq;
  piFind = { results: [], status: q.length >= 3 ? "searching OpenAlex…" : "" };
  renderPiEditors();
  if (q.length < 3) return;
  try {
    const res = await fetch(`https://api.openalex.org/authors?search=${encodeURIComponent(q)}&per-page=6&select=id,display_name,last_known_institutions,works_count`);
    const body = res.ok ? await res.json() : null;
    if (seq !== piSeq) return;
    piFind.results = (body?.results || []).map((a) => ({ name: a.display_name, openalex: a.id.split("/").pop(),
      inst: a.last_known_institutions?.[0]?.display_name || "", works: a.works_count || 0 }));
    piFind.status = piFind.results.length ? "" : "No one by that name in OpenAlex.";
  } catch {
    if (seq === piSeq) piFind.status = "Couldn't reach OpenAlex.";
  }
  renderPiEditors();
}

// Worth suggesting: the last authors of your favorites (usually the PI), then the site's own list.
function piSuggestions() {
  const taken = new Set(Object.values(state.pis).map((x) => nameKey(x.name))), counts = new Map();
  for (const id of Object.keys(state.favorites)) {
    const last = (favPaper(id)?.authors || []).filter((a) => a !== "…").at(-1);
    if (last) counts.set(last, (counts.get(last) || 0) + 1);
  }
  const names = [...[...counts].sort((a, b) => b[1] - a[1]).map(([n]) => n), ...(DATA.people || []).map((p) => p.name)];
  return names.filter((n) => !taken.has(nameKey(n)) && taken.add(nameKey(n))).slice(0, 8);
}

// The editor (in ⚙︎ and in onboarding): this static part once, then renderPiEditors fills it in, so the
// search box keeps its focus while results come back.
const piEditorHTML = () => `<div class="pi-editor">
  <ul class="pi-list"></ul>
  <div class="pi-sugg"></div>
  <input type="search" class="pi-search" placeholder="Find a PI by name" aria-label="Find a PI" autocomplete="off">
  <ul class="pi-results"></ul>
</div>`;

function renderPiEditors() {
  const pis = Object.entries(state.pis).sort((a, b) => a[1].name.localeCompare(b[1].name));
  const sugg = piSuggestions(), watching = new Set(Object.keys(state.pis));
  for (const box of document.querySelectorAll(".pi-editor")) {
    box.querySelector(".pi-list").innerHTML = pis.length
      ? pis.map(([k, x]) => `<li data-pikey="${esc(k)}"><span class="pi-name">${esc(x.name)}</span>
          <span class="pi-inst">${esc(x.inst || "")}</span>
          <button type="button" class="rm" data-piact="remove" aria-label="Stop watching ${esc(x.name)}" title="Stop watching">✕</button></li>`).join("")
      : `<li class="hint">Not watching anyone yet.</li>`;
    box.querySelector(".pi-sugg").innerHTML = sugg.length ? `<span class="hint">Suggested:</span> ${sugg.map((n) =>
      `<button type="button" class="chip" data-piact="suggest" data-name="${esc(n)}">+ ${esc(n)}</button>`).join("")}` : "";
    box.querySelector(".pi-results").innerHTML = piFind.results.map((a, i) => `<li>
        <div class="lk-body">${esc(a.name)}<div class="lk-meta">${esc([a.inst, `${a.works.toLocaleString()} works`].filter(Boolean).join(" · "))}</div></div>
        <button type="button" class="lk-show" data-piact="add" data-i="${i}" ${watching.has(a.openalex) ? "disabled" : ""}>${watching.has(a.openalex) ? "watching" : "watch"}</button>
      </li>`).join("") + (piFind.status ? `<li class="lk-status">${esc(piFind.status)}</li>` : "");
  }
}

function watchPi(a) {
  state.pis[a.openalex || nameKey(a.name)] = { ts: Date.now(), name: a.name,
    ...(a.openalex ? { openalex: a.openalex } : {}), ...(a.inst ? { inst: a.inst } : {}) };
  affinityCache = null; persist(); renderPiEditors();
}

document.addEventListener("click", (e) => {
  const b = e.target.closest(".pi-editor [data-piact]");
  if (!b) return;
  const act = b.dataset.piact;
  if (act === "remove") { forget("pis", [b.closest("[data-pikey]").dataset.pikey]); affinityCache = null; persist(); renderPiEditors(); }
  if (act === "add") watchPi(piFind.results[+b.dataset.i]);
  if (act === "suggest") {
    // From the site's list we already know who it is; anyone else could be several people: search.
    const known = DATA.people?.find((p) => p.name === b.dataset.name && p.openalex);
    if (known) { watchPi(known); return; }
    b.closest(".pi-editor").querySelector(".pi-search").value = b.dataset.name;
    findAuthors(b.dataset.name);
  }
});
document.addEventListener("input", (e) => {
  if (!e.target.matches(".pi-search")) return;
  clearTimeout(piTimer);
  piTimer = setTimeout(() => findAuthors(e.target.value.trim()), 300);
});
// Enter in a search box inside a dialog would submit (and close) it.
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches("dialog input[type=search]")) e.preventDefault();
});

// ---- onboarding ------------------------------------------------------------------------
// The first time someone signs in, four short steps set up their feed:
//   1. About you: a few sentences on what they work on. Saved as their interests (what Claude scores
//      papers for), and used right away to suggest topic tabs.
//   2. Tabs: every topic, the suggested ones ticked. Unticked tabs are hidden, and their papers count
//      for a quarter in For You.
//   3. Favorite papers: landmarks the Graph centers on. Papers near them rank higher, and so does the
//      map neighborhood each one sits in (favoriteRegions).
//   4. PIs to watch, suggested from the favorites' last authors.
// Skipping marks it done; ⚙︎ → Account → "Set up my feed again" brings it back.

const OB_STEPS = ["about", "tabs", "favs", "pis"];
const OFF_TAB_WEIGHT = 0.25;
let ob = null, obDismissed = false, obSeq = 0, obTimer = null;

// Topic tabs suggested from a description. It's split into phrases ("CRISPR screens", "T cell
// exhaustion", …), each phrase finds its 40 nearest papers, and a topic is suggested when it's
// over-represented among them (3x its usual share or more): ticked at 40%+ of some phrase's
// neighbours, offered as a maybe at 15%+. A topic named in the text is ticked too.
const SUGGEST_K = 40;
const QUERY_PREFIX = "Represent this sentence for searching relevant passages: "; // bge's prefix for short queries
const GENERIC_LABELS = new Set(["micro", "viral", "systems", "efficiency", "theory", "optimization", "safety",
  "reasoning", "vision", "multimodal", "agents"]);

function descriptionPhrases(text) {
  const out = [];
  for (const raw of text.replace(/^#.*$/gm, "").split(/[.;\n]+|,\s+(?:and\s+)?|\band\b/)) {
    const x = raw.replace(/^[\s*-]+|[\s*-]+$/g, "");
    if (!x) continue;
    if (out.length && x.split(/\s+/).length < 2) out[out.length - 1] += ` ${x}`;  // one word means little alone
    else out.push(x);
  }
  return out.slice(0, 40);
}

function labelHit(text, t) {
  const lower = ` ${text.toLowerCase()} `;
  return t.label.split(/\s*[/&]\s*|\s+and\s+/).map((s) => s.trim())
    .filter((s) => s.length >= 2 && !GENERIC_LABELS.has(s.toLowerCase()))
    .find((s) => new RegExp(`[^a-z0-9]${s.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/s$/, "")}s?[^a-z0-9]`).test(lower));
}

async function suggestTabs(text) {
  const out = new Map();  // slug -> { strong, share, why }
  for (const t of DATA.topics) {
    const hit = labelHit(text, t);
    if (hit) out.set(t.slug, { strong: true, share: 1, why: `you mentioned ${hit}` });
  }
  const phrases = descriptionPhrases(text);
  await Promise.all([loadVectors(), loadML()]);
  if (!phrases.length || !VEC) return out;
  const qs = await embedInBrowser(phrases.map((x) => QUERY_PREFIX + x));
  const rows = DATA.papers.filter((p) => p.topics?.length).map((p) => [vecRow(p), p.topics]).filter(([r]) => r !== undefined);
  const usual = new Map();
  for (const [, ts] of rows) for (const t of ts) usual.set(t, (usual.get(t) || 0) + 1 / rows.length);
  const { dim, data, norms } = VEC;
  phrases.forEach((phrase, k) => {
    const q = qs[k];
    const sims = rows.map(([r], i) => {
      let d = 0; for (let j = 0; j < dim; j++) d += q[j] * data[r * dim + j];
      return [d / norms[r], i];  // the query's own norm doesn't change the order
    }).sort((a, b) => b[0] - a[0]).slice(0, SUGGEST_K);
    const share = new Map();
    for (const [, i] of sims) for (const t of rows[i][1]) share.set(t, (share.get(t) || 0) + 1 / SUGGEST_K);
    for (const [t, sh] of share) {
      if (sh < 0.15 || sh < 3 * usual.get(t)) continue;
      const strong = sh >= 0.4, prev = out.get(t);
      if (!prev || (strong && !prev.strong) || (strong === prev.strong && sh > prev.share)) {
        out.set(t, { strong, share: sh, why: `“${shortTitle(phrase, 6)}”` });
      }
    }
  });
  return out;
}

function maybeOnboard() {
  if (account.user && account.profile && !account.profile.onboarded && DATA.generated && !ob && !obDismissed) startOnboarding();
}

function startOnboarding() {
  ob = { step: 0, interests: account.profile?.interests || "", suggest: null, suggestedFor: null,
    checked: new Set(DATA.topics.map((t) => t.slug).filter((t) => !state.hiddenTabs.includes(t))),
    found: [], status: "", busy: null, error: null };
  loadVectors(); loadML(); loadMap();  // for tab suggestions, and to place favorites on the map
  renderOnboarding();
  if (!$("onboard").open) $("onboard").showModal();
}

function renderOnboarding() {
  const step = OB_STEPS[ob.step], last = ob.step === OB_STEPS.length - 1;
  $("ob-step").textContent = `Step ${ob.step + 1} of ${OB_STEPS.length}`;
  $("ob-back").hidden = ob.step === 0;
  $("ob-back").disabled = !!ob.busy;
  $("ob-next").textContent = last ? "Finish" : "Next";
  $("ob-next").disabled = !!ob.busy;
  $("ob-error").textContent = ob.error || "";
  const body = $("ob-body");
  if (step === "about") {
    body.innerHTML = `<h3>What do you work on?</h3>
      <p class="hint">A few sentences, like a note to a colleague: your field, the questions you care about,
        what makes a paper a must-read for you, and what to skip. It suggests your tabs next, and it's what
        Claude scores new papers against.</p>
      <textarea id="ob-interests" rows="8" placeholder="I study how T cells become exhausted in tumors, mostly with CRISPR screens and single-cell sequencing…">${esc(ob.interests)}</textarea>`;
  }
  if (step === "tabs") {
    if (ob.busy) { body.innerHTML = `<h3>Your tabs</h3><p class="hint">${esc(ob.busy)}</p>`; return; }
    const sug = ob.suggest || new Map();
    const rank = (t) => (sug.get(t.slug)?.strong ? 0 : sug.has(t.slug) ? 1 : 2);
    body.innerHTML = `<h3>Your tabs</h3>
      <p class="hint">${sug.size ? "Ticked from what you wrote; “maybe” ones are close." : "Pick the topics you want as tabs."}
        Unticked tabs are hidden, and their papers count for less in For You. You can change this any time in ⚙︎.</p>
      ${Object.entries(DATA.sections || {}).map(([sec, label]) => `<h4>${esc(label)}</h4><ul class="ob-tabs">${
        DATA.topics.filter((t) => (t.section || "bio") === sec).sort((a, b) => rank(a) - rank(b)).map((t) => {
          const s = sug.get(t.slug);
          return `<li><label><input type="checkbox" data-obtab="${esc(t.slug)}" ${ob.checked.has(t.slug) ? "checked" : ""}>
            <span class="dot" style="--c:${t.color}"></span>${esc(t.label)}</label>
            ${s ? `<span class="ob-why ${s.strong ? "" : "maybe"}">${s.strong ? "suggested" : "maybe"} · ${esc(s.why)}</span>` : ""}</li>`;
        }).join("")}</ul>`).join("")}`;
  }
  if (step === "favs") {
    body.innerHTML = `<h3>Papers you love</h3>
      <p class="hint">Three or more that define your taste: classics, your own, the ones you send everyone. The
        Graph centers on them, papers near them rank higher, and so does the neighborhood of the map each one
        sits in.</p>
      <input type="search" id="ob-find" placeholder="Title, DOI or arXiv ID" aria-label="Find a paper" autocomplete="off">
      <ul class="ob-found lookup-list"></ul>
      <div class="ob-favs"></div>`;
    renderObFavs();
  }
  if (step === "pis") {
    body.innerHTML = `<h3>PIs to watch</h3>
      <p class="hint">Every new paper with one of them as an author lands in your PI Watch tab and ranks higher in
        For You. Suggestions come from your favorites' last authors.</p>${piEditorHTML()}`;
    renderPiEditors();
  }
}

function renderObFavs() {
  if (!ob || OB_STEPS[ob.step] !== "favs") return;
  const row = (p, i) => {
    const fav = has(state.favorites, p);
    const meta = [p.venue, (p.date || "").slice(0, 4), (p.authors || []).filter((a) => a !== "…").slice(0, 3).join(", ")].filter(Boolean).join(" · ");
    return `<li><button type="button" class="heart ${fav ? "on" : ""}" data-obfav="${i}" aria-label="${fav ? "Remove favorite" : "Add to favorites"}">♥</button>
      <div class="lk-body">${esc(p.title)}<div class="lk-meta">${esc(meta)}</div></div></li>`;
  };
  document.querySelector(".ob-found").innerHTML = ob.found.map(row).join("") + (ob.status ? `<li class="lk-status">${esc(ob.status)}</li>` : "");
  const favs = Object.entries(state.favorites).sort((a, b) => b[1].ts - a[1].ts);
  const regions = favoriteRegions(), names = DATA.regions || [];
  const hoods = new Map();
  for (const r of regions.values()) if (names[r]) hoods.set(names[r], (hoods.get(names[r]) || 0) + 1);
  document.querySelector(".ob-favs").innerHTML = `<h4>Your favorites <span class="n">${favs.length}</span></h4>
    ${favs.length ? `<ul class="ob-favlist">${favs.map(([id]) => {
      const r = regions.get(id);
      return `<li><span class="t">${esc(favPaper(id)?.title || id)}</span>
        <span class="ob-why">${names[r] ? esc(names[r]) : placing ? "placing…" : ""}</span>
        <button type="button" class="rm" data-obunfav="${esc(id)}" aria-label="Remove favorite" title="Remove">✕</button></li>`;
    }).join("")}</ul>` : `<p class="hint">None yet. Search above, or ♥ papers in the feed later.</p>`}
    ${hoods.size ? `<p class="hint">They anchor ${hoods.size === 1 ? "this neighborhood" : `these ${hoods.size} neighborhoods`} of the map:
      ${[...hoods].sort((a, b) => b[1] - a[1]).map(([n, k]) => `<b>${esc(n)}</b>${k > 1 ? ` (${k})` : ""}`).join(", ")}.</p>` : ""}
    ${placeStatus && placeStatus !== "loading" ? `<p class="hint">${esc(placeStatus)}</p>` : ""}`;
}

async function obFind(q) {
  const seq = ++obSeq;
  ob.found = []; ob.status = q.length >= 3 ? "searching…" : "";
  renderObFavs();
  if (q.length < 3) return;
  const ref = paperRef(q), lq = q.toLowerCase();
  const local = ref && byId.has(ref) ? [byId.get(ref)] : [...byId.values()].filter((p) => p.title.toLowerCase().includes(lq)).slice(0, 3);
  let remote = [];
  try { remote = await openAlexWorks(q, ref); } catch { /* offline: local results only */ }
  if (seq !== obSeq || !ob) return;
  const seen = new Set();
  ob.found = [...local, ...remote.map((p) => byId.get(p.id) || p)].filter((p) => !seen.has(p.id) && seen.add(p.id)).slice(0, 8);
  ob.status = ob.found.length ? "" : "No papers found.";
  renderObFavs();
}

async function obNext(dir) {
  if (OB_STEPS[ob.step] === "about") ob.interests = $("ob-interests").value;
  ob.error = null;
  if (dir > 0 && ob.step === OB_STEPS.length - 1) { await finishOnboarding(false); return; }
  ob.step = Math.max(0, Math.min(OB_STEPS.length - 1, ob.step + dir));
  const suggest = OB_STEPS[ob.step] === "tabs" && ob.interests.trim() && ob.suggestedFor !== ob.interests;
  if (suggest) ob.busy = `Matching what you wrote to topics…${extractor ? "" : " (the first time downloads a 34 MB model)"}`;
  renderOnboarding();
  if (!suggest) return;
  const o = ob;
  try {
    o.suggest = await suggestTabs(o.interests);
    o.suggestedFor = o.interests;
    const strong = [...o.suggest].filter(([, x]) => x.strong).map(([t]) => t);
    if (strong.length) o.checked = new Set(strong);
  } catch (err) {
    o.error = `Couldn't suggest tabs (${err.message}). Pick them yourself.`;
  }
  o.busy = null;
  if (ob === o) renderOnboarding();  // unless the dialog was closed meanwhile
}

async function finishOnboarding(skip) {
  ob.busy = "Saving…"; renderOnboarding();
  try {
    if (!skip) {
      if (ob.interests.trim() !== (account.profile?.interests || "").trim()) await saveInterests(ob.interests.trim());
      for (const t of DATA.topics) {
        const on = ob.checked.has(t.slug);
        state.hiddenTabs = on ? state.hiddenTabs.filter((x) => x !== t.slug) : [...new Set([...state.hiddenTabs, t.slug])];
        if (!on) state.topicWeights[t.slug] = Math.min(state.topicWeights[t.slug] ?? 1, OFF_TAB_WEIGHT);
        else if (state.topicWeights[t.slug] === OFF_TAB_WEIGHT) delete state.topicWeights[t.slug];
      }
      persist();
    }
    const { error } = await SB.from("profiles").update({ onboarded: true, updated_at: new Date().toISOString() }).eq("user_id", uid());
    if (error) throw new Error(error.message);
    account.profile = { ...account.profile, onboarded: true };
  } catch (err) {
    ob.busy = null; ob.error = `Couldn't save: ${err.message}`; renderOnboarding();
    return;
  }
  ob = null;
  $("onboard").close();
  affinityCache = nearCache = null;
  render();
}

$("ob-next").addEventListener("click", () => obNext(1));
$("ob-back").addEventListener("click", () => obNext(-1));
$("ob-skip").addEventListener("click", () => finishOnboarding(true));
$("onboard").addEventListener("close", () => { if (ob) { ob = null; obDismissed = true; } });  // Esc: ask again next visit
$("ob-body").addEventListener("change", (e) => {
  const t = e.target.dataset.obtab;
  if (t) e.target.checked ? ob.checked.add(t) : ob.checked.delete(t);
});
$("ob-body").addEventListener("input", (e) => {
  if (e.target.id !== "ob-find") return;
  clearTimeout(obTimer);
  obTimer = setTimeout(() => obFind(e.target.value.trim()), 300);
});
$("ob-body").addEventListener("click", (e) => {
  const add = e.target.closest("[data-obfav]"), rm = e.target.closest("[data-obunfav]");
  if (add) toggleFavorite(ob.found[+add.dataset.obfav]);
  if (rm) { forget("favorites", [rm.dataset.obunfav]); affinityCache = nearCache = null; mapNeedsFit = true; }
  if (add || rm) { persist(); renderObFavs(); }
});

// ---- accounts (Supabase) ---------------------------------------------------------------
// Signed in, your state syncs to your own row in Supabase (user_state) instead of a gist, take-homes
// go to their own table, and your interests text (what the daily build has Claude score papers for)
// lives on your profile. Row-level security means each person can only read and write their own
// rows; the key below is the public one. Invite-only: sign-ups are off, so only people you invite
// from the Supabase dashboard can sign in.

const SUPABASE_URL = "https://uywohcjbluxrtqwphouk.supabase.co";
const SUPABASE_KEY = "sb_publishable_hXcTWch1-JJciG-NeLVHEA__3DtAmXC";
const SB = window.supabase?.createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
});
const redirectTo = () => location.origin + location.pathname;
// A session stored in this browser takes a moment to restore; until then show "signing in", not the
// signed-out page (onAuthStateChange settles it either way).
account.pending = !!SB && (() => {
  if (new URLSearchParams(location.search).has("code")) return true;  // back from a sign-in link
  try { return !!localStorage.getItem(`sb-${new URL(SUPABASE_URL).hostname.split(".")[0]}-auth-token`); } catch { return false; }
})();
const settlePending = () => { if (account.pending) { account.pending = false; if (DATA.generated) render({ keepPage: true }); } };
setTimeout(settlePending, 10000);  // never hang on it

async function sbRead() {
  const { data, error } = await SB.from("user_state").select("state, rev").eq("user_id", uid()).maybeSingle();
  if (error) throw new Error(error.message);
  return data || null;
}

// Postgres stores JSON with its keys reordered, so compare with keys sorted.
const canon = (v) => JSON.stringify(v, (k, x) => (x && typeof x === "object" && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b))) : x));

// Write only if nobody else wrote since we read (rev unchanged). False means: re-read, merge, retry.
async function sbWrite(blob, rev, exists) {
  if (!exists) {
    const { error } = await SB.from("user_state").insert({ user_id: uid(), state: blob, rev: 1 });
    if (error && error.code !== "23505") throw new Error(error.message);  // 23505: another device created it first
    return !error;
  }
  const { data, error } = await SB.from("user_state")
    .update({ state: blob, rev: rev + 1, updated_at: new Date().toISOString() })
    .eq("user_id", uid()).eq("rev", rev).select("rev");
  if (error) throw new Error(error.message);
  return data.length === 1;
}

// One sync round with Supabase: read, merge into the state as it is now, write; retried on a conflict.
async function syncAccount() {
  for (let attempt = 0; attempt < 4; attempt++) {
    const row = await sbRead();
    const remote = row?.state || {};
    const merged = mergeSynced(syncedPart(state), remote);
    Object.assign(state, merged);
    const blob = { ...merged };
    delete blob.summaries;  // those live in their own table
    const { summaries, ...remoteBlob } = remote;
    if (row && canon(blob) === canon(remoteBlob)) return;  // nothing new to write
    if (await sbWrite(blob, row?.rev || 0, !!row)) return;
  }
  throw new Error("couldn't save: another device kept writing at the same time");
}

async function loadAccountExtras() {
  const [{ data: profile }, summaries] = await Promise.all([
    SB.from("profiles").select("display_name, interests, onboarded").eq("user_id", uid()).maybeSingle(),
    fetchAll(SB.from("summaries").select("paper_id, data, ts").eq("user_id", uid())),
  ]);
  account.profile = profile || { interests: "" };
  for (const s of summaries) {
    state.summaries[s.paper_id] = { ...(s.data || {}), ts: Date.parse(s.ts) || Date.now() };
  }
}

// Supabase returns at most 1,000 rows per request: page through.
async function fetchAll(query, page = 1000) {
  const out = [];
  for (let from = 0; ; from += page) {
    const { data, error } = await query.range(from, from + page - 1);
    if (error) throw new Error(error.message);
    out.push(...data);
    if (data.length < page) return out;
  }
}

async function saveSummary(id) {
  if (!uid() || !state.summaries[id]) return;
  const { ts, ...data } = state.summaries[id];
  const { error } = await SB.from("summaries").upsert({ user_id: uid(), paper_id: id, data, ts: new Date(ts).toISOString() });
  if (error) console.warn("couldn't save take-home:", error.message);
}

async function saveInterests(text) {
  const { error } = await SB.from("profiles").update({ interests: text, updated_at: new Date().toISOString() }).eq("user_id", uid());
  if (error) throw new Error(error.message);
  account.profile = { ...account.profile, interests: text };
}

// Switching who's signed in swaps the whole local state: each user has their own copy in this browser.
async function onSignedIn(user) {
  if (account.user?.id === user.id) return;
  const loggedOut = syncedPart(state);
  account.user = user;
  const mine = (() => { try { return localStorage.getItem(storeKey()); } catch { return null; } })();
  state = mine ? loadState() : { ...defaults(), lastVisit: Date.now() };
  settingsSeen = settingsJSON(state);
  affinityCache = nearCache = null;
  try {
    await loadAccountExtras();
    await syncAccount();
    // First sign-in on this browser: offer to bring in what was saved here while signed out.
    const n = ["saved", "favorites", "hidden"].reduce((k, m) => k + Object.keys(loggedOut[m] || {}).length, 0);
    if (!mine && n && confirm(`Add this browser's ${n} saved, favorited and hidden papers to your account?`)) {
      Object.assign(state, mergeSynced(syncedPart(state), loggedOut));
      await syncAccount();
      for (const id of Object.keys(state.summaries)) saveSummary(id);
    }
    saveLocal();
    syncedJSON = JSON.stringify(syncedPart(state));
    Object.assign(syncStatus, { at: Date.now(), error: null });
  } catch (err) {
    account.error = err.message;
  }
  account.pending = false;
  renderAccountButton();
  if (DATA.generated) render({ keepPage: true });
  maybeOnboard();
}

function onSignedOut() {
  account.user = null; account.profile = null;
  state = loadState(); settingsSeen = settingsJSON(state); affinityCache = nearCache = null;
  renderAccountButton();
  if (DATA.generated) render({ keepPage: true });
}

function renderAccountButton() {
  const b = $("account-btn");
  if (!SB) { b.hidden = true; return; }
  const name = account.profile?.display_name || account.user?.email || "";
  b.textContent = account.user ? (name[0] || "?").toUpperCase() : "Sign in";
  b.title = account.user ? `Signed in as ${name}` : "Sign in to sync and get your own AI scores";
  b.classList.toggle("signed-in", !!account.user);
}

function renderAccount() {
  if (!SB) { $("account-box").innerHTML = `<p class="hint">Accounts aren't available (couldn't load Supabase).</p>`; return; }
  const err = account.error ? `<p class="hint sync-error">${esc(account.error)}</p>` : "";
  if (!account.user) {
    $("account-box").innerHTML = `
      <p class="hint">Sign in to keep your saves, favorites and settings on every device, and (from the next
        daily build) get Claude's scores written for your interests. Invite-only.</p>
      <div class="btnrow"><button type="button" data-acct="github">Continue with GitHub</button></div>
      <div class="btnrow"><input type="email" id="acct-email" placeholder="you@example.org" autocomplete="email" aria-label="Email">
        <button type="button" data-acct="email">Email me a sign-in link</button></div>${err}`;
    return;
  }
  $("account-box").innerHTML = `
    <p class="hint">Signed in as <b>${esc(account.user.email || account.profile?.display_name || "")}</b>${
      syncStatus.at ? ` · synced ${new Date(syncStatus.at).toLocaleTimeString([], { timeStyle: "short" })}` : ""}.</p>
    <label class="hint" for="acct-interests">What should Claude look for? Plain prose, like a note to a colleague: your
      fields, what makes a paper a must-read for you, what to skip.</label>
    <textarea id="acct-interests" rows="7" placeholder="I work on…">${esc(account.profile?.interests || "")}</textarea>
    <div class="btnrow"><button type="button" data-acct="save-interests">Save interests</button>
      <button type="button" data-acct="onboard">Set up my feed again</button>
      <button type="button" data-acct="signout">Sign out</button></div>${err}`;
}

$("account-box").addEventListener("click", async (e) => {
  const act = e.target.closest("[data-acct]")?.dataset.acct;
  if (!act) return;
  account.error = null;
  try {
    if (act === "github") {
      const { error } = await SB.auth.signInWithOAuth({ provider: "github", options: { redirectTo: redirectTo() } });
      if (error) throw error;
    }
    if (act === "email") {
      const email = $("acct-email").value.trim();
      if (!email) return;
      const { error } = await SB.auth.signInWithOtp({ email, options: { shouldCreateUser: false, emailRedirectTo: redirectTo() } });
      if (error) throw error;
      account.error = null;
      $("account-box").insertAdjacentHTML("beforeend", `<p class="hint">Check ${esc(email)} for a sign-in link.</p>`);
      return;
    }
    if (act === "save-interests") { await saveInterests($("acct-interests").value); e.target.textContent = "Saved ✓"; return; }
    if (act === "onboard") { $("settings").close(); obDismissed = false; startOnboarding(); return; }
    if (act === "signout") { await SB.auth.signOut(); }
  } catch (err) {
    account.error = /signups not allowed|not allowed/i.test(err.message) ? "That account isn't invited yet. Ask the site owner to invite you." : err.message;
  }
  renderAccount();
});
$("account-btn").addEventListener("click", () => { renderSettings(); $("settings").showModal(); $("account-box").scrollIntoView({ block: "start" }); });

// A failed OAuth sign-in comes back with the error in the URL (e.g. someone who isn't invited).
(function oauthError() {
  const q = new URLSearchParams(location.hash.slice(1) || location.search);
  const msg = q.get("error_description");
  if (msg) {
    account.error = /signups not allowed/i.test(msg) ? "That account isn't invited yet. Ask the site owner to invite you." : msg;
    history.replaceState(null, "", location.pathname);
  }
})();

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
  if ((!sync.token && !uid()) || JSON.stringify(syncedPart(state)) === syncedJSON) return;
  clearTimeout(syncTimer);
  syncTimer = setTimeout(syncNow, 2000);
}

async function syncNow() {
  if (!sync.token && !uid()) return;
  if (syncStatus.busy) { syncStatus.again = true; return; }
  syncStatus.busy = true; clearTimeout(syncTimer);
  try {
    if (uid()) {
      const before = JSON.stringify(syncedPart(state));
      await syncAccount();
      settingsSeen = settingsJSON(state);
      saveLocal();
      syncedJSON = JSON.stringify(syncedPart(state));
      Object.assign(syncStatus, { at: Date.now(), error: null });
      if (syncedJSON !== before) { affinityCache = nearCache = null; if (DATA.generated) render({ keepPage: true }); }
      return;
    }
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
    if ($("settings").open) { renderSync(); renderAccount(); }
  }
}

function renderSync() {
  const err = syncStatus.error ? `<p class="hint sync-error">${esc(syncStatus.error)}</p>` : "";
  if (uid()) {
    $("sync-box").innerHTML = `<p class="hint">You're signed in, so everything syncs with your account.</p>${err}`;
    return;
  }
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
  if (!document.hidden && (sync.token || uid()) && Date.now() - (syncStatus.at || 0) > 60000) syncNow();
});

// Sign-in state: the session is restored from this browser (or completed from a sign-in link).
if (SB) {
  SB.auth.onAuthStateChange((event, session) => {
    if (session?.user) setTimeout(() => onSignedIn(session.user), 0);  // not inside Supabase's callback
    else { if (event === "SIGNED_OUT") onSignedOut(); settlePending(); }
  });
}
renderAccountButton();
if (sync.token && !SB) syncNow();

load().catch((err) => {
  $("sub").textContent = "couldn't load papers";
  $("feed").innerHTML = `<div class="empty">No data yet — run <code>python -m pipeline.build</code>.<br><small>${esc(err.message)}</small></div>`;
});
