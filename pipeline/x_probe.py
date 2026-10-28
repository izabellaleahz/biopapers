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
    BIO = ('(url:"doi.org" OR url:nature OR url:science OR url:cell OR url:pnas OR url:elifesciences OR url:plos'
           ' OR url:nejm OR biorxiv OR medrxiv) has:links -is:retweet')
    CS = '(url:"arxiv.org" OR url:"huggingface.co/papers" OR alphaxiv) has:links -is:retweet'
    qs = [BIO + (f" min_likes:{n}" if n else "") for n in (0, 5, 10, 20, 50)]
    qs += [CS + (f" min_likes:{n}" if n else "") for n in (0, 10, 20, 50)]
    qs += [f"{t} has:links -is:retweet" for t in ("url:science", "url:cell", "url:plos", "url:pnas", "url:elifesciences",
                                                  "url:nejm", "medrxiv", "alphaxiv", "url:biorxiv", 'url:"doi.org"')]
    qs += [f"{t} has:links -is:retweet min_likes:5" for t in ("url:science", "url:cell", 'url:"doi.org"', "url:nature", "biorxiv")]
    variants = {(q.replace(BIO, "BIO").replace(CS, "CS"))[:60]: q for q in qs}
    with client(timeout=30) as c:
        c.headers["Authorization"] = f"Bearer {os.environ['X_BEARER_TOKEN']}"
        print("== counts, last ~7 days")
        for label, q in variants.items():
            r = c.get(COUNTS, params={"query": q, "start_time": start, "granularity": "day"})
            if r.is_error:
                print(f"{label:60} HTTP {r.status_code} {r.text[:300]}")
                continue
            days = [d["tweet_count"] for d in r.json().get("data", [])]
            print(f"{label:60} total {sum(days):>8,}  per day {days}")



if __name__ == "__main__":
    main()
