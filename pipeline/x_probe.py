"""One-off X diagnostic: which publisher terms match on X (counts only, ~$0.01 each)."""

import os
import time

from .common import client

COUNTS = "https://api.x.com/2/tweets/counts/recent"
TERMS = ["url:pubmed", "url:ncbi", "url:sciencedirect", "url:springer", "url:link.springer", "url:wiley", "url:oup",
         "url:academic", "url:thelancet", "url:lancet", "url:jamanetwork", "url:jama", "url:bmj", "url:plos", "url:frontiersin",
         "url:mdpi", "url:tandfonline", "url:sagepub", "url:acs", "url:pubs", "url:rsc", "url:aps", "url:iop", "url:ieee",
         "url:ieeexplore", "url:acm", "url:aclanthology", "url:openreview", "url:ssrn", "url:researchsquare", "url:chemrxiv",
         "url:osf", "url:psyarxiv", "url:embopress", "url:jci", "url:aacrjournals", "url:ahajournals", "url:annualreviews",
         "url:royalsocietypublishing", "url:cambridge", "url:jneurosci", "url:asm", "url:biologists", "url:elife",
         "url:nejm", "url:pnas", "url:researchgate", "url:semanticscholar", "url:europepmc", "url:proceedings",
         "url:mlr", "url:neurips", "url:jmlr", "url:aaai", "url:cvf", "url:thecvf", "url:emerald", "url:karger",
         "url:thieme", "url:degruyter", "url:hindawi", "url:cureus", "url:peerj", "url:f1000research",
         "url:scientificamerican", "url:quantamagazine"]


def main() -> None:
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 166 * 3600))
    with client(timeout=30) as c:
        c.headers["Authorization"] = f"Bearer {os.environ['X_BEARER_TOKEN']}"
        for t in TERMS:
            r = c.get(COUNTS, params={"query": f"{t} has:links -is:retweet min_likes:5", "start_time": start,
                                      "granularity": "day"})
            n = r.json().get("meta", {}).get("total_tweet_count") if r.is_success else f"HTTP {r.status_code} {r.text[:120]}"
            print(f"PROBE {t:28} {n}")


if __name__ == "__main__":
    main()
