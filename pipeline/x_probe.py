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
    variants = {
        "bio, min_likes:5 (build)": _x_query(bio, 5),
        "bio, no min_likes": _x_query(bio, 0),
        "cs, min_likes:10 (build)": _x_query(cs, 10),
        "cs, no min_likes": _x_query(cs, 0),
        'url:"nature.com" only': 'url:"nature.com" -is:retweet',
        '"nature.com" keyword, no url:': '"nature.com" -is:retweet',
        'url:"arxiv.org" only': 'url:"arxiv.org" -is:retweet',
        'url:"biorxiv.org" only': 'url:"biorxiv.org" -is:retweet',
    }
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

        print("\n== one page of 50 posts each, bio query as built")
        for order in ("relevancy", "recency"):
            r = c.get(X_SEARCH, params={"query": _x_query(bio, 5), "start_time": start, "max_results": 50,
                                        "sort_order": order, "tweet.fields": "public_metrics,entities,created_at"})
            if r.is_error:
                print(f"{order:10} HTTP {r.status_code} {r.text[:300]}")
                continue
            data = r.json()
            posts = data.get("data") or []
            likes = sorted(((t.get("public_metrics") or {}).get("like_count", 0) for t in posts), reverse=True)
            urls = [u.get("unwound_url") or u.get("expanded_url") or ""
                    for t in posts for u in (t.get("entities") or {}).get("urls", [])]
            mapped = sum(bool(doi_from_url(u)) for u in urls)
            print(f"{order:10} {len(posts)} posts · next_token: {bool((data.get('meta') or {}).get('next_token'))}"
                  f" · likes top/median {likes[:5]}/{likes[len(likes) // 2] if likes else '-'}"
                  f" · {len(urls)} links, {mapped} resolved to a paper"
                  f" · oldest {min((t.get('created_at', '') for t in posts), default='-')}")
            print("   sample unmatched links:", [u for u in urls if not doi_from_url(u)][:8])


if __name__ == "__main__":
    main()
