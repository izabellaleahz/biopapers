"""biopapers MVP v0 — FastAPI web app.

Server-rendered feed of recent bioRxiv preprints in the lab's topics, with a
one-click upvote per paper. Anonymous votes for v0 (no auth yet).

Run:
    uvicorn main:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

import config
import db

app = FastAPI(title="biopapers")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/", response_class=HTMLResponse)
def feed(request: Request, topic: str | None = None, sort: str = "votes"):
    if topic not in config.TOPICS:
        topic = None  # None = "All"
    if sort not in ("votes", "date"):
        sort = "votes"
    papers = db.fetch_feed(topic=topic, sort=sort)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "papers": papers,
            "topics": config.TOPICS,
            "counts": db.counts_by_topic(),
            "active_topic": topic,
            "sort": sort,
        },
    )


@app.post("/upvote/{paper_id}", response_class=PlainTextResponse)
def upvote(paper_id: int):
    """Increment a paper's votes; returns the new count (consumed by the page JS)."""
    new_total = db.add_vote(paper_id)
    if new_total is None:
        return PlainTextResponse("not found", status_code=404)
    return PlainTextResponse(str(new_total))
