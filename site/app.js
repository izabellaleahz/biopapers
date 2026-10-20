// biopapers — static client. Loads data/papers.json (+ older.json in the background, and
// ml.json when the ML / CS section is opened), ranks and renders the feed.
// Saves / upvotes / settings live in localStorage, optionally synced to a secret GitHub Gist.

"use strict";

const PAGE = 40;
const STORE_KEY = "biopapers:v1";
const SYNC_KEY = "biopapers:sync"; // { token, gistId } — kept apart so Export JSON never contains the token
const SYNC_MAPS = ["votes", "saved", "hidden"];
const SYNC_SETTINGS = ["weights", "topicWeights", "halfLife", "tabOrder", "hiddenTabs"];
const DAY = 86400000;

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
  ml: { head: ["mlforyou", "trending"], tail: ["saved"] },
};
const HOT_TABS = new Set(["buzz", "trending"]);

const WEIGHTS = {
  recency: { label: "Recency", def: 1 },
  buzz: { label: "Buzz (HF upvotes · HN · Bluesky · press)", def: 0.6 },
  pi: { label: "Watched PIs", def: 1 },
  affinity: { label: "Similar to what I save / upvote", def: 0.8 },
  vote: { label: "My upvotes", def: 0.3 },
};

const SORTS = { foryou: "For You", date: "Newest", buzz: "Buzz" };

// ---- state ---------------------------------------------------------------

const defaults = () => ({
  votes: {}, saved: {}, hidden: {}, removed: {}, settingsAt: 0,
  weights: Object.fromEntries(Object.entries(WEIGHTS).map(([k, w]) => [k, w.def])),
  topicWeights: {}, halfLife: 7, tabOrder: [], hiddenTabs: [],
  section: "bio", tabBy: { bio: "foryou", ml: "mlforyou" }, sortByTab: {},
  lastVisit: null, prevVisit: null,
});

function loadState() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? { ...defaults(), ...JSON.parse(raw) } : defaults();
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

const curTab = () => state.tabBy[state.section];
const setTab = (t) => { state.tabBy[state.section] = t; };
let ui = { shown: PAGE, query: "", newOnly: false, pi: null };

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
  affinityCache = null;
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

function affinityScores() {
  if (affinityCache) return affinityCache;
  const liked = new Set([...Object.keys(state.saved), ...Object.keys(state.votes)]);
  affinityCache = new Map();
  if (!liked.size) return affinityCache;

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
    const p = byId.get(id) || state.saved[id]?.paper;
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
  return topicWeight(p) * (w.recency * recency + w.buzz * buzz + w.pi * pi
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

  $("sections").innerHTML = Object.entries(DATA.sections || {}).map(([k, label]) =>
    `<button data-section="${k}" class="${k === state.section ? "active" : ""}">${esc(label)}</button>`).join("");

  const updated = DATA.generated ? new Date(DATA.generated).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";
  const inSection = DATA.papers.filter((p) => (p.sections || ["bio"]).includes(state.section)).length;
  const loading = state.section === "ml" && !inSection ? "loading ML papers…" : `${inSection.toLocaleString()} papers`;
  $("sub").textContent = `${loading} · updated ${updated}`;

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
    : "";

  const list = visiblePapers();
  $("feed").innerHTML = list.length
    ? list.slice(0, ui.shown).map(card).join("")
    : `<div class="empty">${tab === "saved" ? "Nothing saved yet — hit ☆ on a paper." : "No papers match."}</div>`;
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
  const voted = has(state.votes, p), saved = has(state.saved, p);
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
        ${hot}
        ${topics.map((t) => `<span class="badge" style="--c:${t.color}">${esc(t.label)}</span>`).join("")}
      </div>
      <div class="authors" title="${esc((p.authors || []).join(", "))}">${authors}${p.n_authors ? ` <span>(${p.n_authors})</span>` : ""}</div>
      ${hf.tldr ? `<div class="tldr"><b>TL;DR</b> ${esc(hf.tldr)}</div>` : ""}
      ${p.abstract ? `<div class="abstract">${esc(p.abstract)}</div>` : ""}
      ${links.length ? `<div class="links">${links.join("")}</div>` : ""}
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

function snapshot(p) {
  // Saved papers keep a copy so they survive aging out of the 30-day window.
  const { id, doi, title, authors, venue, date, url, kind, topics, pis, abstract } = p;
  return { id, doi, title, authors, venue, date, url, kind, topics, pis, abstract: (abstract || "").slice(0, 600) };
}

$("feed").addEventListener("click", (e) => {
  const abs = e.target.closest(".abstract");
  if (abs) { abs.classList.toggle("open"); return; }
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.closest(".paper").dataset.id;
  const p = byId.get(id) || state.saved[id]?.paper;
  if (!p) return;
  const act = btn.dataset.act;
  if (act === "vote") toggle("votes", p, Date.now());
  if (act === "save") toggle("saved", p, { ts: Date.now(), paper: snapshot(p) });
  if (act === "hide") {
    if (curTab() === "saved") toggle("saved", p);
    else state.hidden[p.id] = Date.now();
  }
  if (act !== "hide") affinityCache = null;
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
  const b = e.target.closest("[data-section]");
  if (!b || b.dataset.section === state.section) return;
  state.section = b.dataset.section; ui.pi = null; persist(); render();
  if (state.section === "ml") loadML();
  window.scrollTo({ top: 0 });
});
$("chips").addEventListener("click", (e) => {
  const c = e.target.closest("[data-pi]");
  if (c) { ui.pi = c.dataset.pi || null; render(); }
});
$("more").addEventListener("click", () => { ui.shown += PAGE; render({ keepPage: true }); });
$("new-only").addEventListener("change", (e) => { ui.newOnly = e.target.checked; render(); });
let searchTimer;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { ui.query = e.target.value.trim(); render(); }, 150);
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
    persist(); affinityCache = null; renderSettings(); render();
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
  persist(); affinityCache = null; renderSettings(); render();
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
    if (syncedJSON !== before) { affinityCache = null; if (DATA.generated) render({ keepPage: true }); }
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
