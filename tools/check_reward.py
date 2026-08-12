"""Replay real matches through the reward function and ask whether it ranks what it should.

    python tools/check_reward.py <traces.jsonl> [--limit N]

THE ASSERTION THAT MATTERS. A reward function is a claim about what is worth doing, and the claim
is testable BEFORE any training: run it over trajectories whose outcomes are already known, and
check that the seat which won collected more than the seats which lost. If it does not rank the
teacher's own games correctly, no amount of PPO will fix it -- the agent will faithfully maximise
the wrong thing and the run will look healthy the entire time.

This is worth more than every other check here combined, because a reward bug is the one class of
defect that cannot be found by watching training. Loss falls, entropy behaves, the return curve
climbs -- and the policy is learning to do something nobody wanted. Days, at a day per run.

360 matches already sit in the corpus with their outcomes recoverable (tools/outcomes.py), so the
test costs nothing to run and needs no new data.

WHAT IT CHECKS, in descending order of what a failure would cost:

    RANKS       the winner's return beats the mean loser's, per match
    ORDERS      return rises with final tier across all seats (rank correlation)
    BOUNDED     no infinity, no NaN, and the shaping term stays inside its stated range
    ACCOUNTS    the tier bonuses actually paid match the tiers actually gained

ACCOUNTS is the subtle one and it is why the progress term is suppressed on a tier-up. `have`
resets against a larger `need` the moment a tier is gained, so the shaping term goes sharply
NEGATIVE exactly when the bonus fires. Left unsuppressed it claws back a large part of the reward
for the only event that wins the game -- and it would never show up as anything but slightly slow
learning.

NOT A TEST OF WHETHER THE REWARD IS GOOD. The teacher is a decent heuristic, not an optimal
player, so "the winner scored highest" is a floor and not a ceiling. A reward that passes here can
still be badly shaped. A reward that FAILS here is definitely broken.
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.env import _progress, default_reward          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outcomes import outcomes                                   # noqa: E402


def replay(paths, limit=0):
    """-> {(map, match, seat): [rows in order]}. Whole matches, in trace order."""
    seats = collections.defaultdict(list)
    n = 0
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
                n += 1
                key = (str(d.get("map", "?")), str(d.get("match_id", "?")),
                       int(float(d.get("seat", -1))))
                seats[key].append(d.get("state") or {})
                if limit and n >= limit:
                    return seats
    return seats


def spearman(pairs):
    """Rank correlation, written out rather than imported -- scipy is not a dependency here and
    this is twelve lines. Ties get their average rank, which matters: final tiers are integers
    0..4 across hundreds of seats, so ties are the common case rather than an edge one."""
    if len(pairs) < 3:
        return float("nan")

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    xs, ys = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    finals = outcomes(args.traces)
    seats = replay(args.traces, args.limit)
    if not seats:
        print("no rows read")
        return 1

    failures = []
    returns = {}                       # (map, match, seat) -> episode return
    tier_paid = {}                     # (map, match, seat) -> reward attributable to tier-ups
    n_steps = 0
    unseen_tiers = [0]

    for key, states in seats.items():
        total = shaped = tiers = 0.0
        for a, b in zip(states, states[1:]):
            r = default_reward(a, b, None)
            n_steps += 1
            if not math.isfinite(r):
                failures.append(("reward is not finite", "%s -> %r" % (key, r)))
                continue
            total += r

            ta = float((a.get("self") or {}).get("tier", 0))
            tb = float((b.get("self") or {}).get("tier", 0))
            if tb > ta:
                tiers += r
            else:
                shaped += r
                # Stated range of the shaping term: a difference of a function bounded in 0..1,
                # scaled by 1.0. Anything outside means _progress escaped its own bounds, which
                # would put an unbounded quantity into the return.
                if abs(r) > 1.0 + 1e-9:
                    failures.append(("shaping term outside [-1, 1]", "%s: %.3f" % (key, r)))

        fin = (finals.get(key[:2]) or {}).get(key[2])
        if fin is not None and fin.get("won"):
            total += 10.0
        returns[key] = total
        tier_paid[key] = tiers

        # ACCOUNTS. The ladder is 1/2/3/5 for tiers 1..4, so the bonuses paid across a trajectory
        # must equal the sum of the rungs it actually climbed.
        #
        # AGAINST THE OBSERVED TIERS, NOT THE FINAL ONE, and the difference is not pedantry:
        # TIER IS A TEAM PROPERTY. On a 3v3 all three seats read the same tier and the same star
        # total, so a seat's tier rises when a TEAMMATE tiers -- between its own decisions, or
        # after its last one. A seat knocked out at turn 9 of a 32-turn match finishes at tier 3
        # having personally climbed nothing, and comparing to the final tier calls that a bug in
        # the reward. It is not: the seat still sees each rise as a jump between its consecutive
        # observations and is paid for it, which is the correct treatment of a shared objective.
        first = float((states[0].get("self") or {}).get("tier", 0)) if states else 0.0
        last = float((states[-1].get("self") or {}).get("tier", 0)) if states else 0.0
        want = sum({1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}.get(t, 1.0)
                   for t in range(int(first) + 1, int(last) + 1))
        if abs(tiers - want) > 1e-6:
            failures.append(("tier bonuses paid do not match tiers climbed",
                             "%s: paid %.1f, observed tier %g -> %g implies %.1f"
                             % (key, tiers, first, last, want)))
        if fin is not None and float(fin.get("tier", 0)) > last:
            # Not a failure -- credit that arrives after the seat stops acting. Counted because
            # it bounds how much of the objective the per-step reward can ever see: for these
            # seats the terminal win bonus is the only signal that the team got there.
            unseen_tiers[0] += 1

    # RANKS -- the headline.
    by_match = collections.defaultdict(dict)
    for (mp, mid, seat), tot in returns.items():
        by_match[(mp, mid)][seat] = tot

    ranked = mis = 0
    tier_pairs = []
    for mkey, seat_totals in by_match.items():
        fins = finals.get(mkey) or {}
        for seat, f in fins.items():
            if seat in seat_totals:
                tier_pairs.append((float(f.get("tier", 0)), seat_totals[seat]))
        winners = [s for s, f in fins.items() if f.get("won")]
        losers = [s for s in seat_totals if s not in winners]
        if len(winners) != 1 or not losers:
            continue
        w = seat_totals.get(winners[0])
        if w is None:
            continue
        mean_loser = sum(seat_totals[s] for s in losers) / len(losers)
        if w > mean_loser:
            ranked += 1
        else:
            mis += 1
            if mis <= 3:
                failures.append(("winner did not out-earn the losers",
                                 "%s: winner %.2f vs mean loser %.2f" % (mkey, w, mean_loser)))

    total_matches = ranked + mis
    rho = spearman(tier_pairs)

    print("=== reward check: %d seats over %d matches, %d transitions ==="
          % (len(returns), len(by_match), n_steps))
    print("  RANKS   winner out-earned the losers in %d of %d matches (%.1f%%)"
          % (ranked, total_matches, 100.0 * ranked / max(1, total_matches)))
    print("  ORDERS  return vs final tier, Spearman rho = %.3f" % rho)
    print("  CREDIT  %d seats finished above the last tier they saw -- a team-mate tiered after"
          " they stopped acting, so only the terminal bonus can carry it" % unseen_tiers[0])
    print("  returns: min %.2f  mean %.2f  max %.2f"
          % (min(returns.values()), sum(returns.values()) / len(returns), max(returns.values())))

    if rho < 0.5:
        failures.append(("return barely tracks final tier",
                         "rho = %.3f -- the reward does not order outcomes it is meant to" % rho))
    if total_matches and ranked / total_matches < 0.9:
        failures.append(("winner loses on return too often",
                         "%.1f%% ranked correctly" % (100.0 * ranked / total_matches)))

    if failures:
        kinds = collections.Counter(k for k, _ in failures)
        print("\n  FAIL: %d kind(s)" % len(kinds))
        seen = set()
        for kind, detail in failures:
            if kind in seen:
                continue
            seen.add(kind)
            print("    %-46s %6d" % (kind, kinds[kind]))
            print("      e.g. %s" % detail)
        return 1

    print("\n  all reward assertions held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
