"""Reproduce the sidecar's memory growth in a minute instead of a tournament.

    python tools/leak_probe.py runs/bc-long.pt --n 3000

A serve.py left running across one 160-match tournament reached 28-35 GB resident and starved the
machine: the game died in oInit with "Memory allocation failed", and because Snitch's crash dialog
is modal the process then sat there and the tournament looked like a slow match for 43 minutes.

A first fix (`block_on_close = False`) was applied on a theory and did NOT work -- the next run
leaked just as fast. So this measures instead: it drives a real sidecar with real payloads from the
corpus and reports RSS and live thread count as they grow, which distinguishes the two candidates.

    threads climbing with RSS   -> handler threads are not exiting (a connection is never closed,
                                   and each stuck thread holds its stack and its request buffers)
    RSS climbing, threads flat  -> something per-request is retained: tensors, payload dicts, or
                                   an unbounded structure inside the policy
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def rss_and_threads(pid):
    """Resident bytes and thread count for a pid, via wmic -- no psutil dependency here."""
    rss = out = 0
    try:
        r = subprocess.run(["wmic", "process", "where", "ProcessId=%d" % pid,
                            "get", "WorkingSetSize,ThreadCount"],
                           capture_output=True, text=True, timeout=30)
        nums = [int(x) for x in r.stdout.replace("\r", "").split() if x.isdigit()]
        if len(nums) >= 2:
            # wmic prints ThreadCount then WorkingSetSize (alphabetical)
            out, rss = min(nums), max(nums)
    except Exception:                                                # noqa: BLE001
        pass
    return rss, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--port", type=int, default=8977)
    ap.add_argument("--traces", default="../RaifuWars/data/cpu-traces-v3-ffa.jsonl")
    args = ap.parse_args()

    rows = []
    with open(args.traces, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("available_actions"):
                rows.append({"state": d["state"], "available_actions": d["available_actions"]})
            if len(rows) >= 200:
                break
    print("loaded %d real payloads to replay" % len(rows))

    srv = subprocess.Popen([sys.executable, "tools/serve.py", args.checkpoint,
                            "--port", str(args.port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    url = "http://127.0.0.1:%d/v1/act" % args.port
    base_rss, base_thr = rss_and_threads(srv.pid)
    print("start: rss %.0f MB, threads %d" % (base_rss / 1048576, base_thr))

    try:
        for i in range(args.n):
            body = json.dumps(rows[i % len(rows)]).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            if (i + 1) % 500 == 0:
                rss, thr = rss_and_threads(srv.pid)
                print("  %5d requests | rss %7.0f MB (+%.0f) | threads %4d (+%d)"
                      % (i + 1, rss / 1048576, (rss - base_rss) / 1048576, thr, thr - base_thr))
    finally:
        rss, thr = rss_and_threads(srv.pid)
        print("\nend: rss %.0f MB, threads %d" % (rss / 1048576, thr))
        grow = (rss - base_rss) / max(1, args.n)
        print("growth: %.1f KB per request" % (grow / 1024))
        print("  a tournament is ~10,000 decisions, so that projects to %.1f GB"
              % (grow * 10000 / 1073741824))
        if thr - base_thr > args.n * 0.5:
            print("  VERDICT: handler threads are not exiting -- one per request survives")
        elif grow > 100 * 1024:
            print("  VERDICT: per-request retention, threads are fine -- look at what `choose`"
                  " keeps a reference to")
        else:
            print("  VERDICT: no meaningful growth reproduced at this scale")
        srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
