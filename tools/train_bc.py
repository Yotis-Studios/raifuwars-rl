"""Behaviour cloning from traces, optionally weighted by how the seat's match turned out.

    python tools/train_bc.py <traces.jsonl> --out runs/bc.pt [--weight none|win|tier]

PLAIN BC CAPS AT THE TEACHER, and the teacher wins ~32% of its own seats. Every decision is treated
as equally good, including every decision from the three seats that lost. That is the ceiling an
LLM fine-tune already hit on this game: 86% agreement with the built-in AI, 2 wins in 40.

So the weight matters more than the architecture here:

    none   every decision equal -- the baseline, and what SFT does
    win    only the seat that won that match
    tier   weighted by the seat's final tier, so a seat that reached 3 teaches more than one at 1

`win` is the strongest signal and throws away three quarters of the data. `tier` keeps everything
and still prefers the seats that got somewhere. Which is better is an empirical question and the
point of having both.

THE SPLIT IS BY MATCH, NEVER BY ROW. Decisions inside one match share a board, a roster and a map,
so a random row split puts near-identical positions on both sides and reports memorisation as
validation. Keyed on (map, match_id) because match_id alone collided across maps in corpora
collected before that was fixed.
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.features import encode_actions, encode_state    # noqa: E402
from raifuwars_rl.policy import ActionScorer, masked_log_probs    # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))
from outcomes import outcomes                                     # noqa: E402


def load(traces, weight_mode, limit=0):
    finals = outcomes(traces) if weight_mode != "none" else {}
    rows = []
    for path in traces:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                acts = d.get("available_actions") or []
                chosen = str(d.get("chosen_action", ""))
                if len(acts) < 2:
                    continue
                idx = next((i for i, a in enumerate(acts)
                            if str(a.get("action_id")) == chosen), None)
                if idx is None:
                    continue

                key = (d.get("map", "?"), d.get("match_id", "?"))
                seat = int(float(d.get("seat", -1)))
                w = 1.0
                if weight_mode != "none":
                    seats = finals.get(key) or {}
                    fin = seats.get(seat)
                    if fin is None:
                        continue
                    if weight_mode == "win":
                        w = 1.0 if fin["won"] else 0.0
                    else:
                        # Tier 0..4 -> 0.25..1.25. Never zero: a seat that reached tier 1 still
                        # played legal, sensible Raifu Wars and its decisions are not noise.
                        w = 0.25 + 0.25 * float(fin["tier"])
                    if w <= 0.0:
                        continue

                st = d.get("state") or {}
                rows.append((key, encode_state(st), encode_actions(st, acts), idx, w))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def batches(rows, size, rng, device):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [rows[j] for j in order[i:i + size]]
        n = max(r[2].shape[0] for r in chunk)
        S = torch.tensor(np.stack([r[1] for r in chunk]), device=device)
        A = torch.zeros(len(chunk), n, chunk[0][2].shape[1], device=device)
        for b, r in enumerate(chunk):
            A[b, :r[2].shape[0]] = torch.tensor(r[2], device=device)
        L = torch.tensor([r[2].shape[0] for r in chunk], device=device)
        Y = torch.tensor([r[3] for r in chunk], device=device)
        W = torch.tensor([r[4] for r in chunk], dtype=torch.float32, device=device)
        yield S, A, L, Y, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="runs/bc.pt")
    ap.add_argument("--weight", choices=["none", "win", "tier"], default="none")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    rows = load(args.traces, args.weight, args.limit)
    if not rows:
        print("no usable rows")
        return 1

    keys = sorted({r[0] for r in rows})
    rng = random.Random(7)
    rng.shuffle(keys)
    val_keys = set(keys[:max(1, len(keys) // 10)])
    train = [r for r in rows if r[0] not in val_keys]
    val = [r for r in rows if r[0] in val_keys]

    print("rows %d (%d train / %d val) from %d matches, weight=%s, loaded in %.0fs"
          % (len(rows), len(train), len(val), len(keys), args.weight, time.time() - t0))

    device = torch.device(args.device)
    net = ActionScorer().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        net.train()
        tot = seen = 0.0
        for S, A, L, Y, W in batches(train, args.batch, rng, device):
            lp = masked_log_probs(net(S, A), L)
            loss = -(lp.gather(1, Y[:, None]).squeeze(1) * W).sum() / W.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(Y); seen += len(Y)

        net.eval()
        hit = n = 0
        with torch.no_grad():
            for S, A, L, Y, W in batches(val, args.batch, rng, device):
                pred = masked_log_probs(net(S, A), L).argmax(1)
                hit += int((pred == Y).sum()); n += len(Y)
        # Always-first is the trivial policy for a scorer: it is what a model that has learned
        # nothing but the ordering will do, and the action list is emitted in a fixed order.
        base = sum(1 for r in val if r[3] == 0) / max(1, len(val))
        print("  epoch %d: loss %.4f | val top-1 %.1f%% (always-first %.1f%%)"
              % (ep, tot / max(1, seen), 100.0 * hit / max(1, n), 100.0 * base))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "weight_mode": args.weight}, args.out)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
