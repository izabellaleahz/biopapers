"""Signed-in users' data the daily build needs, read from Supabase with the service key.

For now that's their PI watchlists. Each user's `pis` (in their synced state, user_state.state) are
fetched from OpenAlex like config/people.yaml's, tagged on papers and kept as long, so their papers are
in the feed and the site can find them. The site matches each user's own PIs itself; the build only
makes sure the papers are there. Their names end up in the public data as papers' `pis` tags (as
people.yaml's do), but who watches whom is never written out or logged.

Needs SUPABASE_SERVICE_KEY (an Actions secret); without it this returns nothing.
"""

import os

import httpx

from .common import client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://uywohcjbluxrtqwphouk.supabase.co")


def _headers(key: str) -> dict:
    # New-style secret keys (sb_secret_…) go in `apikey` only; a legacy service_role JWT also needs
    # to be the bearer token for PostgREST to use its role.
    # And not the shared client's "Mozilla/…" User-Agent: Supabase refuses a secret key from anything
    # that looks like a browser.
    auth = {"apikey": key} if key.startswith("sb_") else {"apikey": key, "Authorization": f"Bearer {key}"}
    return {**auth, "User-Agent": "biopapers-build (python-httpx)"}


def watched_pis(known: list[dict], log) -> list[dict]:
    """Users' PIs not already in `known` (people.yaml), as people.yaml-style {"name", "openalex"}."""
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()  # a pasted secret can carry a newline
    if not key:
        return []
    rows, page = [], 1000
    try:
        with client() as c:
            while True:  # PostgREST caps a response at 1,000 rows
                resp = c.get(f"{SUPABASE_URL}/rest/v1/user_state", params={"select": "pis:state->pis"},
                             headers={**_headers(key), "Range": f"{len(rows)}-{len(rows) + page - 1}"})
                resp.raise_for_status()
                rows += resp.json()
                if len(resp.json()) < page:
                    break
    except httpx.HTTPStatusError as ex:  # Supabase's message names the problem ("Invalid API key"), never the key
        log(f"  user PIs: couldn't read from Supabase ({ex.response.status_code}: {ex.response.text[:200]})")
        return []
    except Exception as ex:
        log(f"  user PIs: couldn't read from Supabase ({ex.__class__.__name__})")
        return []
    seen = {p.get("openalex") or p["name"].lower() for p in known}
    out = []
    for row in rows:
        for pi in (row.get("pis") or {}).values():
            name, aid = (pi or {}).get("name"), (pi or {}).get("openalex")
            if not name or (aid or name.lower()) in seen:
                continue
            seen.add(aid or name.lower())
            out.append({"name": name, **({"openalex": aid} if aid else {})})
    log(f"  user PIs: {len(out)} beyond people.yaml, from {len(rows)} users")
    return out
