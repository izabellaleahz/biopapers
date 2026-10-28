"""One-off X diagnostic: why does the buzz search return so few posts? (~$1-2 per run)

Counts (cheap, no posts read) for variants of the build's queries, then one small page each of
relevancy- and recency-ordered search, to see which part of the query truncates the results.

    X_BEARER_TOKEN=... python -m pipeline.x_probe
"""

import os
import time

from .common import client, load_config
from .sources.buzz import X_SEARCH, _x_query, doi_from_url

COUNTS = "https://api.x.com/2/tweets/counts/recent"


def main() -> None:
    queries = load_config("sources")["buzz"]["x"]["queries"]
    bio, cs = queries["bio"]["domains"], queries["cs"]["domains"]
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 166 * 3600))
    variants = {f"{q:42}": q for q in [
        'url:"biorxiv.org" -is:retweet', 'url:"www.biorxiv.org" -is:retweet', 'url:biorxiv -is:retweet',
        'biorxiv -is:retweet', 'url:"nature.com" -is:retweet', 'url:"www.nature.com" -is:retweet',
        'url:nature -is:retweet', 'url:"science.org" -is:retweet', 'url:"www.science.org" -is:retweet',
        'url:"cell.com" -is:retweet', 'url:"doi.org" -is:retweet', 'url:"nejm.org" -is:retweet',
        'url:"www.nejm.org" -is:retweet', 'url:"arxiv.org" -is:retweet', 'url:"medrxiv.org" -is:retweet',
        'url:"www.medrxiv.org" -is:retweet', 'url:"elifesciences.org" -is:retweet', 'url:"pnas.org" -is:retweet',
        'url:"huggingface.co/papers" -is:retweet', 'url:"huggingface.co" -is:retweet', 'url:"alphaxiv.org" -is:retweet',
        'url:"www.alphaxiv.org" -is:retweet',
    ]}
    with client(timeout=30) as c:
        c.headers["Authorization"] = f"Bearer {os.environ['X_BEARER_TOKEN']}"
        print("== counts, last ~7 days")
        for label, q in variants.items():
            r = c.get(COUNTS, params={"query": q, "start_time": start, "granularity": "day"})
            if r.is_error:
                print(f"{label:34} HTTP {r.status_code} {r.text[:300]}")
                continue
            days = [d["tweet_count"] for d in r.json().get("data", [])]
            print(f"{label:34} total {sum(days):>8,}  per day {days}")

        print("\n== raw link fields, 5 posts linking to biorxiv / nejm")
        import json
        for q in ("biorxiv -is:retweet has:links", 'url:"nejm.org" -is:retweet'):
            r = c.get(X_SEARCH, params={"query": q, "start_time": start, "max_results": 10,
                                        "tweet.fields": "entities,note_tweet"})
            print(q, "->", r.status_code)
            for t in (r.json().get("data") or [])[:5] if not r.is_error else []:
                print("  ", json.dumps((t.get("entities") or {}).get("urls", []))[:600])
            if r.is_error:
                print("  ", r.text[:300])

if __name__ == "__main__":
    main()
