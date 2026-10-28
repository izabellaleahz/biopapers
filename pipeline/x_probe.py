"""One-off: why do `from:` queries for curated accounts return almost nothing? Counts only (7 days)."""
import os
import time

from .common import client

QUERIES = [
    "from:EricTopol", "from:EricTopol has:links", "from:EricTopol has:links -is:retweet",
    "from:EricTopol is:retweet", "from:EricTopol is:quote",
    "from:karpathy", "from:karpathy has:links", "from:anshulkundaje", "from:anshulkundaje has:links",
    "from:genophoria", "from:lpachter", "from:omarsar0", "from:omarsar0 has:links", "from:emollick has:links",
    "(from:EricTopol OR from:karpathy OR from:omarsar0) has:links",
    "(from:EricTopol OR from:karpathy OR from:omarsar0)",
    "from:EricTopol url:\"doi.org\"", "from:omarsar0 url:\"arxiv.org\"",
]

with client() as c:
    c.headers["Authorization"] = f"Bearer {os.environ['X_BEARER_TOKEN']}"
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 167 * 3600))
    for q in QUERIES:
        r = c.get("https://api.x.com/2/tweets/counts/recent", params={"query": q, "granularity": "day", "start_time": start})
        total = r.json().get("meta", {}).get("total_tweet_count") if r.is_success else f"HTTP {r.status_code} {r.text[:200]}"
        print(f"{str(total):>8}  {q}", flush=True)
