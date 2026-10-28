"""One-off: why do `from:` queries for curated accounts return almost nothing? Counts only (7 days)."""
import os
import time

from .common import client

import yaml
from .sources import buzz
GROUPS = [" OR ".join(v["terms"]) for v in buzz.people_queries(yaml.safe_load(open("config/sources.yaml"))["buzz"]["x"]["people"]).values()]
FILTERS = ["", "-is:reply", "-is:reply -is:retweet", "is:retweet", "is:quote", "has:links", "is:reply has:links",
           "-is:reply (has:links OR is:quote OR is:retweet)"]
QUERIES = [f"(from:anshulkundaje) {f}".strip() for f in FILTERS] + [f"({g}) {f}".strip() for g in GROUPS for f in FILTERS]

with client() as c:
    c.headers["Authorization"] = f"Bearer {os.environ['X_BEARER_TOKEN']}"
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 167 * 3600))
    for q in QUERIES:
        r = c.get("https://api.x.com/2/tweets/counts/recent", params={"query": q, "granularity": "day", "start_time": start})
        total = r.json().get("meta", {}).get("total_tweet_count") if r.is_success else f"HTTP {r.status_code} {r.text[:200]}"
        print(f"{str(total):>8}  {q[:60]} … {q[-50:]}", flush=True)
