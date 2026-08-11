"""Recover per-seat match outcomes from traces, with no extra instrumentation.

    python tools/outcomes.py <traces.jsonl>

WHY THIS IS POSSIBLE AT ALL. Every trace carries `state.players` -- seat, tier, stars, kills,
status -- for EVERY seat, not just the one deciding. So the last trace of a match is a final
scoreboard, and a corpus collected for imitation learning turns out to already contain the outcome
labels that reinforcement learning wants. Nothing needed re-collecting.

WHY IT IS KEYED ON (map, match_id). `match_id` was "rw-<seed>" and every map replays the same seed
sequence, so 360 matches produced 80 ids and 40 of those spanned up to four boards. Taking "the
last trace of a match" by id alone returns the last trace of four unrelated games on four different
maps. Fixed at the source, but a corpus already collected still needs the pair.

WHAT "WON" MEANS HERE. Reaching tier 4 ends the match and wins it, so the winner is the seat at the
highest tier, stars breaking ties. That is a reconstruction, not a recorded fact: the final trace
is emitted BEFORE the winning action is applied, so a winner often appears at tier 3 with `tier`
as its chosen action. Both are handled -- a seat whose last decision was to tier is credited with
the tier it was about to reach.
"""

import collections
import json
import sys


def load(paths):
    """-> {(map, match_id): {seat: {...final...}}}, plus each match's last chosen action."""
    last_trace = {}
    last_action = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (d.get("map", "?"), d.get("match_id", "?"))
                last_trace[key] = d
                last_action[key] = (int(float(d.get("seat", -1))),
                                    str(d.get("chosen_action", "")))
    return last_trace, last_action


def outcomes(traces):
    last_trace, last_action = load(traces)
    out = {}
    for key, d in last_trace.items():
        players = (d.get("state") or {}).get("players") or []
        if not players:
            continue
        seats = {}
        for p in players:
            seats[int(float(p.get("seat", -1)))] = {
                "tier": float(p.get("tier", 0)),
                "stars": float(p.get("stars", 0)),
                "kills": float(p.get("kills", 0)),
            }
        # The final trace precedes the final action. A seat whose last decision was `tier` is
        # credited with the tier it was about to reach, or the reconstruction misses every winner.
        acting_seat, action = last_action.get(key, (-1, ""))
        if action == "tier" and acting_seat in seats:
            seats[acting_seat]["tier"] += 1
        best = max(seats, key=lambda s: (seats[s]["tier"], seats[s]["stars"]))
        for s, v in seats.items():
            v["won"] = 1 if s == best else 0
        out[key] = seats
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    o = outcomes(sys.argv[1:])
    if not o:
        print("no matches found")
        return 1

    winners = collections.Counter()
    tiers = collections.Counter()
    for key, seats in o.items():
        for s, v in seats.items():
            if v["won"]:
                winners[s] += 1
            tiers[int(v["tier"])] += 1

    print("=== outcomes recovered from %d matches ===" % len(o))
    print("  wins by seat: %s" % dict(sorted(winners.items())))
    print("  final tier distribution: %s" % dict(sorted(tiers.items())))

    # ONE WINNER PER MATCH IS THE INVARIANT. Reaching tier 4 ends the match, so anything else means
    # the reconstruction is wrong -- and a silently wrong outcome label would poison every weight
    # derived from it.
    bad = [k for k, seats in o.items() if sum(v["won"] for v in seats.values()) != 1]
    print("  matches without exactly one winner: %d" % len(bad))
    at4 = sum(1 for seats in o.values()
              for v in seats.values() if v["won"] and v["tier"] >= 4)
    print("  winners actually at tier 4        : %d / %d" % (at4, len(o)))
    if at4 < len(o) * 0.9:
        print("\n  WARNING: most winners are not at tier 4. Either matches are ending some other")
        print("  way, or the reconstruction is wrong. Do not weight a dataset on this yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
