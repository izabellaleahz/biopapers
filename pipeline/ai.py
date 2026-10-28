"""AI relevance scores + TL;DRs from Claude Haiku 5.5.

Each unscored paper gets p["ai"] = {"score": 0-10, "tldr": "...", "why": "...", "topics": [...]}, written for the
reader described in config/profile.md (plus their favorites and the papers they saved /
upvoted, if Gist sync is on and BIOPAPERS_GIST_ID is set). Papers go out PER_REQUEST to a request.
"topics" are the Bio tabs the model would file the paper under: the build uses them for journal papers
no keyword rule tagged (broad journals like Nature and Science have no subject categories).

Two ways to reach the model, picked by which key is set:
  ANTHROPIC_API_KEY   the Message Batches API (half price). A build waits up to `wait_min` minutes
                      for its batch; one that hasn't finished is recorded in site/data/ai_pending.json
                      and collected by the next build instead of being resubmitted.
  OPENROUTER_API_KEY  OpenRouter's chat completions, called directly (a few at a time), at most
                      MAX_PER_RUN papers per build, newest first.
With neither, the step is skipped.
"""

import functools
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .common import CONFIG_DIR, load_config

MODEL = "claude-haiku-5-5"
OPENROUTER_MODEL = "anthropic/claude-haiku-5.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_PER_RUN = 3000  # OpenRouter path: bounds a first run's cost (~$0.25) and time; the rest go next build
PER_REQUEST = 20
MAX_ABSTRACT = 1200
MAX_LIKED = 40
# Batch rates for claude-haiku-5-5, $/MTok (half the standard $0.10 in / $0.50 out).
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625}

INSTRUCTIONS = """You triage new research papers for one reader, described below. For each numbered paper return:
- score: an integer 0-10 for how much THIS reader would want to read it. Be stingy. Most papers are 0-3; a solid paper in one of their areas is 4-6; 7-8 means they'd be sorry to miss it; 9-10 is rare (a few a week) and reserved for field-changing work squarely in their interests. Relevance to this reader matters as much as quality.
- tldr: one plain sentence (at most 30 words) stating the main finding or contribution concretely: what was shown or built, not "this paper studies". No hype.
- why: a few words on why it matters to this reader, or "outside your areas".
- topics: which of the reader's tabs (listed below) the paper belongs in, by slug, judged on what the paper is about, not on how much the reader would like it. Only tabs it squarely belongs in: a neuroscience paper that mentions microglia isn't Immunology. Most papers outside these areas get none.
Return every paper exactly once, with its number as n."""


@functools.cache
def bio_topics() -> tuple[tuple[str, str], ...]:
    """(slug, description) for each Bio tab in config/topics.yaml — the choices for "topics"."""
    out = []
    for t in load_config("topics")["topics"]:
        if t.get("section", "bio") == "bio":
            cats = ", ".join(t.get("categories", []))
            out.append((t["slug"], f"{t['label']} — {t['description']}" if t.get("description")
                        else f"{t['label']} ({cats})" if cats else t["label"]))
    return tuple(out)


def instructions() -> str:
    return INSTRUCTIONS + "\n\n<tabs>\n" + "\n".join(f"- {s}: {d}" for s, d in bio_topics()) + "\n</tabs>"


def schema() -> dict:
    return {
        "type": "object",
        "properties": {"papers": {"type": "array", "items": {
            "type": "object",
            "properties": {"n": {"type": "integer"}, "score": {"type": "integer"},
                           "tldr": {"type": "string"}, "why": {"type": "string"},
                           "topics": {"type": "array", "items": {"type": "string", "enum": [s for s, _ in bio_topics()]}}},
            "required": ["n", "score", "tldr", "why", "topics"],
            "additionalProperties": False,
        }}},
        "required": ["papers"],
        "additionalProperties": False,
    }


def load_profile() -> str:
    text = (CONFIG_DIR / "profile.md").read_text()
    return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()


def profile_version() -> str:
    """Stamped on every score. Editing profile.md, the instructions or the Bio tabs changes it, and papers
    scored against the old version get re-scored, newest first."""
    return hashlib.sha1(f"{instructions()}\n{load_profile()}".encode()).hexdigest()[:8]


def needs_score(p: dict, version: str) -> bool:
    return not p.get("ai") or p["ai"].get("v") != version


def titles(entries: dict, by_id: dict[str, dict], limit: int) -> list[str]:
    """Newest-first titles for synced {id: ts | {ts, paper}} entries."""
    rows = [(e if isinstance(e, (int, float)) else e.get("ts", 0),
             (by_id.get(pid) or (e.get("paper") if isinstance(e, dict) else None) or {}).get("title"))
            for pid, e in entries.items()]
    out = []
    for _, title in sorted(rows, key=lambda x: -x[0]):
        if title and title not in out:
            out.append(title)
    return out[:limit]


def system_prompt(favorites: list[str], liked: list[str]) -> str:
    text = f"{instructions()}\n\n<reader_profile>\n{load_profile()}\n</reader_profile>"
    if favorites:
        text += ("\n\n<their_favorite_papers>\nPapers they picked as defining their interests — the strongest "
                 "signal of what they care about:\n" + "\n".join(f"- {t}" for t in favorites)
                 + "\n</their_favorite_papers>")
    if liked:
        text += ("\n\n<papers_they_saved_or_upvoted>\n" + "\n".join(f"- {t}" for t in liked)
                 + "\n</papers_they_saved_or_upvoted>")
    return text


def reader_prompt(synced: dict | None, by_id: dict[str, dict]) -> str:
    synced = synced or {}
    favorites = titles(synced.get("favorites") or {}, by_id, MAX_LIKED)
    liked = [t for t in titles({**(synced.get("votes") or {}), **(synced.get("saved") or {})}, by_id, MAX_LIKED)
             if t not in favorites]
    return system_prompt(favorites, liked)


def describe(n: int, p: dict) -> str:
    authors = [a for a in p.get("authors", []) if a != "…"]
    if len(authors) > 4:
        authors = authors[:2] + ["…"] + authors[-2:]
    meta = " · ".join(x for x in [p.get("venue"), p.get("date"), ", ".join(authors)] if x)
    return f"[{n}] {p['title']}\n{meta}\n{(p.get('abstract') or '(no abstract)')[:MAX_ABSTRACT]}"


def chunked(papers: list[dict]) -> list[list[dict]]:
    return [papers[i:i + PER_REQUEST] for i in range(0, len(papers), PER_REQUEST)]


def user_message(chunk: list[dict]) -> str:
    return "\n\n".join(describe(n, p) for n, p in enumerate(chunk, 1))


def apply_scores(text: str, ids: list[str], by_id: dict[str, dict]) -> int:
    """Write the model's JSON answer onto the papers it covers; returns how many it scored."""
    scored, valid = 0, {s for s, _ in bio_topics()}
    for item in json.loads(text).get("papers", []):
        n = item.get("n", 0)
        if 1 <= n <= len(ids) and (p := by_id.get(ids[n - 1])):
            p["ai"] = {"score": max(0, min(10, int(item["score"]))),
                       "tldr": item["tldr"].strip(), "why": item["why"].strip(),
                       "topics": [t for t in item.get("topics", []) if t in valid], "v": profile_version()}
            scored += 1
    return scored


def submit(client, papers: list[dict], system: str) -> dict:
    chunks = chunked(papers)
    requests = [{
        "custom_id": f"r{i}",
        "params": {
            "model": MODEL,
            "max_tokens": 8000,  # ~1,500 tokens of JSON for 20 papers, plus room for thinking
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message(chunk)}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema()}},
        },
    } for i, chunk in enumerate(chunks)]
    batch = client.messages.batches.create(requests=requests)
    return {"id": batch.id, "requests": {f"r{i}": [p["id"] for p in chunk] for i, chunk in enumerate(chunks)}}


def collect(client, batch: dict, by_id: dict[str, dict], log) -> None:
    scored = failed = 0
    usage = dict.fromkeys(PRICE, 0)
    for r in client.messages.batches.results(batch["id"]):
        ids = batch["requests"].get(r.custom_id, [])
        if r.result.type != "succeeded":
            failed += len(ids)
            continue
        msg = r.result.message
        u = msg.usage
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
        if msg.stop_reason != "end_turn":  # max_tokens / refusal: no usable JSON, retried next build
            failed += len(ids)
            continue
        scored += apply_scores(next((b.text for b in msg.content if b.type == "text"), ""), ids, by_id)
    cost = sum(usage[k] * PRICE[k] for k in PRICE) / 1e6
    log(f"  ai: batch {batch['id']}: {scored} scored, {failed} failed · "
        f"{usage['input'] + usage['cache_read'] + usage['cache_write']:,} in / {usage['output']:,} out tokens ≈ ${cost:.2f}")


def openrouter_json(c: httpx.Client, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict:
    """One OpenRouter chat completion constrained to `schema`; returns the raw response JSON."""
    body = {
        "model": OPENROUTER_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": user},
        ],
        "reasoning": {"effort": "low"},
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        "usage": {"include": True},  # adds the request's cost in $ to `usage`
    }
    for attempt in range(4):
        resp = c.post(OPENROUTER_URL, json=body)
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < 3:
            time.sleep(5 * 2 ** attempt)
            continue
        if resp.is_error:
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def openrouter_call(c: httpx.Client, system: str, chunk: list[dict]) -> dict:
    return openrouter_json(c, system, user_message(chunk), schema())


def complete_json(system: str, user: str, schema: dict) -> dict | None:
    """A single structured answer from Haiku 5.5 through whichever key is set (None with neither,
    or if the call fails). For small jobs like naming map regions."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic

            msg = anthropic.Anthropic().messages.create(
                model=MODEL, max_tokens=8000, system=system, messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
            return json.loads(next(b.text for b in msg.content if b.type == "text")) if msg.stop_reason == "end_turn" else None
        if os.environ.get("OPENROUTER_API_KEY"):
            headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "biopapers"}
            with httpx.Client(timeout=180, headers=headers) as c:
                choice = openrouter_json(c, system, user, schema)["choices"][0]
            return json.loads(choice["message"]["content"]) if choice.get("finish_reason") == "stop" else None
    except Exception as ex:
        print(f"  ai: complete_json failed: {str(ex)[:200]}", flush=True)
    return None


def score_chunk(c: httpx.Client, system: str, chunk: list[dict], by_id: dict[str, dict]) -> tuple[int, int, float, Exception | None]:
    """Score one request's worth of papers -> (scored, failed, cost, first error). If the safety filter
    declines it, a single sensitive paper can sink all 20, so split in half and retry; a paper declined
    on its own is marked so it isn't retried every day."""
    ids = [p["id"] for p in chunk]
    try:
        result = openrouter_call(c, system, chunk)
    except Exception as ex:  # one bad request shouldn't sink the rest; its papers retry next build
        return 0, len(ids), 0.0, ex
    cost = (result.get("usage") or {}).get("cost") or 0
    choice = (result.get("choices") or [{}])[0]
    reason = choice.get("finish_reason")
    if reason == "content_filter":
        if len(chunk) == 1:
            by_id[ids[0]]["ai"] = {"score": None, "declined": True, "v": profile_version()}
            return 0, 0, cost, None
        half = len(chunk) // 2
        parts = [score_chunk(c, system, part, by_id) for part in (chunk[:half], chunk[half:])]
        return (sum(x[0] for x in parts), sum(x[1] for x in parts), cost + sum(x[2] for x in parts),
                next((x[3] for x in parts if x[3]), None))
    try:
        if reason != "stop":  # cut off: retried next build
            raise ValueError(f"finish_reason {reason}")
        return apply_scores(choice["message"]["content"], ids, by_id), 0, cost, None
    except (ValueError, KeyError, TypeError) as ex:
        return 0, len(ids), cost, ex


def run_openrouter(todo: list[dict], system: str, by_id: dict[str, dict], log) -> None:
    # Newest first, but journal papers still waiting for a topic go ahead of everything else.
    needs_topic = lambda p: p.get("kind") == "article" and not p.get("topics")
    todo = sorted(todo, key=lambda p: (needs_topic(p), p.get("date", "")), reverse=True)[:MAX_PER_RUN]
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "biopapers"}
    with httpx.Client(timeout=180, headers=headers) as c, ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda ch: score_chunk(c, system, ch, by_id), chunked(todo)))
    scored, failed, cost = (sum(r[i] for r in results) for i in range(3))
    declined = sum(1 for p in todo if (by_id[p["id"]].get("ai") or {}).get("declined"))
    if first_error := next((r[3] for r in results if r[3]), None):
        log(f"  ai: first failure: {str(first_error)[:300]}")
    log(f"  ai: openrouter {OPENROUTER_MODEL}: {scored} scored, {failed} failed, {declined} declined by the "
        f"safety filter · ${cost:.3f}")


def run(papers: list[dict], pending: list[dict], wait_min: float, log, synced: dict | None = None) -> list[dict]:
    """Score `papers` in place. `synced` is the site's synced state (favorites, saves, votes).
    Returns the batches still running (for ai_pending.json)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            log("  ai: skipped (no ANTHROPIC_API_KEY or OPENROUTER_API_KEY)")
            return pending
        by_id = {p["id"]: p for p in papers}
        todo = [p for p in papers if needs_score(p, profile_version())]
        if todo:
            run_openrouter(todo, reader_prompt(synced, by_id), by_id, log)
        else:
            log("  ai: nothing new to score")
        return pending
    import anthropic

    client = anthropic.Anthropic()
    by_id = {p["id"]: p for p in papers}
    still = []
    for b in pending:
        try:
            if client.messages.batches.retrieve(b["id"]).processing_status == "ended":
                collect(client, b, by_id, log)
            else:
                still.append(b)
        except anthropic.NotFoundError:
            log(f"  ai: batch {b['id']} is gone — its papers will be resubmitted")
    waiting = {pid for b in still for ids in b["requests"].values() for pid in ids}
    todo = [p for p in papers if needs_score(p, profile_version()) and p["id"] not in waiting]
    if not todo:
        log(f"  ai: nothing new to score ({len(still)} batch(es) still running)")
        return still

    batch = submit(client, todo, reader_prompt(synced, by_id))
    log(f"  ai: submitted {len(todo)} papers as batch {batch['id']} ({len(batch['requests'])} requests)")
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        time.sleep(30)
        if client.messages.batches.retrieve(batch["id"]).processing_status == "ended":
            collect(client, batch, by_id, log)
            return still
    log(f"  ai: batch {batch['id']} still running after {wait_min:.0f} min — the next build collects it")
    return [*still, batch]
