"""AI relevance scores + TL;DRs from Claude, via the Message Batches API (half price).

Each unscored paper gets p["ai"] = {"score": 0-10, "tldr": "...", "why": "..."}, written for the
reader described in config/profile.md (plus their favorites and the papers they saved /
upvoted, if Gist sync is on and BIOPAPERS_GIST_ID is set). Papers go out PER_REQUEST to a request. A build waits up to
`wait_min` minutes for its batch; one that hasn't finished by then is recorded in
site/data/ai_pending.json and collected by the next build instead of being resubmitted.

Needs ANTHROPIC_API_KEY; without it the step is skipped.
"""

import json
import os
import time

from .common import CONFIG_DIR

MODEL = "claude-haiku-5-5"
PER_REQUEST = 20
MAX_ABSTRACT = 1200
MAX_LIKED = 40
# Batch rates for claude-haiku-5-5, $/MTok (half the standard $0.10 in / $0.50 out).
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625}

INSTRUCTIONS = """You triage new research papers for one reader, described below. For each numbered paper return:
- score: an integer 0-10 for how much THIS reader would want to read it. Be stingy. Most papers are 0-3; a solid paper in one of their areas is 4-6; 7-8 means they'd be sorry to miss it; 9-10 is rare (a few a week) and reserved for field-changing work squarely in their interests. Relevance to this reader matters as much as quality.
- tldr: one plain sentence (at most 30 words) stating the main finding or contribution concretely: what was shown or built, not "this paper studies". No hype.
- why: a few words on why it matters to this reader, or "outside your areas".
Return every paper exactly once, with its number as n."""

SCHEMA = {
    "type": "object",
    "properties": {"papers": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "score": {"type": "integer"},
                       "tldr": {"type": "string"}, "why": {"type": "string"}},
        "required": ["n", "score", "tldr", "why"],
        "additionalProperties": False,
    }}},
    "required": ["papers"],
    "additionalProperties": False,
}


def load_profile() -> str:
    text = (CONFIG_DIR / "profile.md").read_text()
    return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()


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
    text = f"{INSTRUCTIONS}\n\n<reader_profile>\n{load_profile()}\n</reader_profile>"
    if favorites:
        text += ("\n\n<their_favorite_papers>\nPapers they picked as defining their interests — the strongest "
                 "signal of what they care about:\n" + "\n".join(f"- {t}" for t in favorites)
                 + "\n</their_favorite_papers>")
    if liked:
        text += ("\n\n<papers_they_saved_or_upvoted>\n" + "\n".join(f"- {t}" for t in liked)
                 + "\n</papers_they_saved_or_upvoted>")
    return text


def describe(n: int, p: dict) -> str:
    authors = [a for a in p.get("authors", []) if a != "…"]
    if len(authors) > 4:
        authors = authors[:2] + ["…"] + authors[-2:]
    meta = " · ".join(x for x in [p.get("venue"), p.get("date"), ", ".join(authors)] if x)
    return f"[{n}] {p['title']}\n{meta}\n{(p.get('abstract') or '(no abstract)')[:MAX_ABSTRACT]}"


def submit(client, papers: list[dict], system: str) -> dict:
    chunks = [papers[i:i + PER_REQUEST] for i in range(0, len(papers), PER_REQUEST)]
    requests = [{
        "custom_id": f"r{i}",
        "params": {
            "model": MODEL,
            "max_tokens": 8000,  # ~1,500 tokens of JSON for 20 papers, plus room for thinking
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "\n\n".join(describe(n, p) for n, p in enumerate(chunk, 1))}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
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
        text = next((b.text for b in msg.content if b.type == "text"), "")
        for item in json.loads(text).get("papers", []):
            n = item.get("n", 0)
            if 1 <= n <= len(ids) and (p := by_id.get(ids[n - 1])):
                p["ai"] = {"score": max(0, min(10, int(item["score"]))),
                           "tldr": item["tldr"].strip(), "why": item["why"].strip()}
                scored += 1
    cost = sum(usage[k] * PRICE[k] for k in PRICE) / 1e6
    log(f"  ai: batch {batch['id']}: {scored} scored, {failed} failed · "
        f"{usage['input'] + usage['cache_read'] + usage['cache_write']:,} in / {usage['output']:,} out tokens ≈ ${cost:.2f}")


def run(papers: list[dict], pending: list[dict], wait_min: float, log, synced: dict | None = None) -> list[dict]:
    """Score `papers` in place. `synced` is the site's synced state (favorites, saves, votes).
    Returns the batches still running (for ai_pending.json)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("  ai: skipped (no ANTHROPIC_API_KEY)")
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
    todo = [p for p in papers if not p.get("ai") and p["id"] not in waiting]
    if not todo:
        log(f"  ai: nothing new to score ({len(still)} batch(es) still running)")
        return still

    synced = synced or {}
    favorites = titles(synced.get("favorites") or {}, by_id, MAX_LIKED)
    liked = [t for t in titles({**(synced.get("votes") or {}), **(synced.get("saved") or {})}, by_id, MAX_LIKED)
             if t not in favorites]
    system = system_prompt(favorites, liked)
    batch = submit(client, todo, system)
    log(f"  ai: submitted {len(todo)} papers as batch {batch['id']} ({len(batch['requests'])} requests)")
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        time.sleep(30)
        if client.messages.batches.retrieve(batch["id"]).processing_status == "ended":
            collect(client, batch, by_id, log)
            return still
    log(f"  ai: batch {batch['id']} still running after {wait_min:.0f} min — the next build collects it")
    return [*still, batch]
