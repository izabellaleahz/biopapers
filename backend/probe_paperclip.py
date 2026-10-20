"""One-off: see what Paperclip's MCP tools return for a few papers (run with `modal run`)."""

import modal

app = modal.App("biopapers-probe")
image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx")


@app.function(image=image, secrets=[modal.Secret.from_name("biopapers")], timeout=300)
def probe(dois: list[str]) -> None:
    import json, os, re
    import httpx

    url = "https://paperclip.gxl.ai/mcp"
    c = httpx.Client(timeout=60, headers={"X-API-Key": os.environ["PAPERCLIP_API_KEY"], "Content-Type": "application/json",
                                          "Accept": "application/json, text/event-stream"})
    r = c.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "biopapers", "version": "1"}}})
    if sid := r.headers.get("mcp-session-id"):
        c.headers["Mcp-Session-Id"] = sid
    c.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    n = [1]

    def call(tool, **args):
        n[0] += 1
        t = c.post(url, json={"jsonrpc": "2.0", "id": n[0], "method": "tools/call", "params": {"name": tool, "arguments": args}}).text
        m = re.findall(r"^data: (.*)$", t, re.M)
        d = json.loads(m[-1] if m else t)
        if "error" in d:
            return "ERROR " + json.dumps(d["error"])[:300]
        return "\n".join(x.get("text", "") for x in d["result"]["content"])

    print("STATUS:", call("status")[:400])
    for doi in dois:
        print(f"\n===== {doi}")
        out = call("lookup", field="doi", value=doi, as_json=True)
        print("LOOKUP:", out[:900])
        m = re.search(r"/papers/([A-Za-z0-9_]+)|\"id\"\s*:\s*\"([A-Za-z0-9_]+)\"", out)
        pid = m and (m.group(1) or m.group(2))
        if not pid:
            continue
        print("LS:", call("ls", path=f"/papers/{pid}/")[:400])
        print("SECTIONS:", call("ls", path=f"/papers/{pid}/sections/")[:400])
        body = call("cat", path=f"/papers/{pid}/content.lines", lines="1-40")
        print(f"CAT content.lines (first 40 lines, {len(body)} chars):", body[:1200])


@app.local_entrypoint()
def main():
    probe.remote([
        "10.1101/2025.06.10.658933",       # bioRxiv preprint (June 2025)
        "10.1002/advs.77518",              # open-access journal paper from the feed (Sept 2026)
        "10.1038/s41592-018-0305-7",       # Nature Methods 2019 (PMC author manuscript)
        "10.48550/arxiv.2305.10601",       # arXiv
    ])
