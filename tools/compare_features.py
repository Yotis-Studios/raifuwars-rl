"""Encode the SAME real state from both payloads and subtract.

    hemlock roundtrip.hml EMIT=pairs.jsonl        # in the raifusim tree
    python tools/compare_features.py pairs.jsonl

THE TIGHTEST CHECK THE BRIDGE ADMITS. `check_features.py` run over sim states and over real ones
compares two DISTRIBUTIONS, so it catches a dead column but not a column that is merely wrong --
`dist_own_base` computed from the wrong point is alive, plausible, and useless. Here the state is
the same real state in both files, so the feature vectors must agree NUMBER FOR NUMBER, and any
disagreement is the serializer.

The action side is matched by `action_id` rather than by position: the two lists are the same set
in different orders, and comparing row i to row i would report every column as wrong.

Card rows are dropped from the game's side. They are out of scope by construction, and counting a
deliberate omission as a serializer fault measures the wrong thing.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.features import (ACTION_FIELDS, STATE_FIELDS,   # noqa: E402
                                   encode_actions, encode_state)

CARD_TYPES = ("play_card", "discard_card", "tier_choice", "select_tile", "cancel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    ds, da = [], []
    ns = na = 0
    id_missing = id_extra = 0
    with open(args.pairs, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            sim, game = p["sim"], p["game"]

            ds.append(encode_state(sim["state"]) - encode_state(game["state"]))
            ns += 1

            g_rows = [a for a in game["available_actions"] if a.get("type") not in CARD_TYPES]
            s_rows = sim["available_actions"]
            g_enc = {a["action_id"]: r for a, r in
                     zip(g_rows, encode_actions(game["state"], g_rows))}
            s_enc = {a["action_id"]: r for a, r in
                     zip(s_rows, encode_actions(sim["state"], s_rows))}
            for aid, srow in s_enc.items():
                if aid not in g_enc:
                    id_extra += 1
                    continue
                da.append(srow - g_enc[aid])
                na += 1
            id_missing += sum(1 for aid in g_enc if aid not in s_enc)

            if ns >= args.n:
                break

    if not ds:
        print("no pairs")
        return 1

    def report(name, mat, fields):
        print("\n=== %s: %d rows, |sim - game| per feature ===" % (name, mat.shape[0]))
        print("  %-28s %12s %12s %10s" % ("feature", "max |diff|", "mean |diff|", "rows off"))
        worst = 0.0
        for i, f in enumerate(fields):
            col = np.abs(mat[:, i])
            hi = float(np.nanmax(col)) if col.size else 0.0
            n_off = int(np.sum(col > args.tol))
            worst = max(worst, hi)
            flag = "" if n_off == 0 else "   <-- DIFFERS"
            print("  %-28s %12.6f %12.6f %10d%s"
                  % (f, hi, float(np.nanmean(col)), n_off, flag))
        return worst

    w1 = report("state", np.stack(ds), STATE_FIELDS)
    w2 = report("action", np.stack(da), ACTION_FIELDS)

    print("\nstates compared      : %d" % ns)
    print("action rows matched  : %d" % na)
    print("action ids sim-only  : %d" % id_extra)
    print("action ids game-only : %d" % id_missing)
    print("worst absolute difference on any feature: %.6g" % max(w1, w2))
    if max(w1, w2) <= args.tol:
        print("\nOK: the serializer reproduces every feature the game's own payload produces.")
        return 0
    print("\nFAIL: a feature differs. The columns marked DIFFERS above are the ones to read.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
