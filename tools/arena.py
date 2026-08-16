"""Four policies on one board, seat rotation included. The measurement self-play needs.

    python3 tools/arena.py --sim ~/Projects/raifusim/driver.hml --matches 400 \
        --agents runs/ppo-selfplay/best.pt,runs/ppo-sim/best.pt,runs/bc-long.pt,greedy

A SELF-PLAY RETURN IS NOT COMPARABLE TO A GREEDY-OPPONENT RETURN, which is the whole reason this
file exists. Return is measured against whoever else is on the board, so an agent that improves
while its opponents improve exactly as fast holds its return flat -- and an agent whose opponent
got harder can improve while its return FALLS. The training curve cannot distinguish "learned
nothing" from "learned as fast as the opposition". Putting the checkpoints on one board can.

SEAT ROTATION IS NOT OPTIONAL. Turn order is fixed and seat 0 decides first every turn, which on a
race to tier is a real advantage worth several points of win rate -- measured at greedy-vs-greedy
before this was written, not assumed. Four agents parked on four seats would therefore measure the
seats as much as the agents. So the tournament is run as n rotations of matches/n, one process
each, with every agent occupying every seat for an equal share.

`greedy` AND `random` ARE PLAYED BY THE DRIVER, not from here. They are the simulator's own
built-ins; reimplementing either in Python to make it a peer would create a second copy of a
policy that has to stay identical to the one the training runs used. So a rotation asks the
learner only for the seats holding real checkpoints (`learners=`) and the built-in fills the rest.

WHY SAMPLED AND NOT ARGMAX by default: it is how these policies were trained and how their
returns were measured, and argmax makes a deterministic policy that can sit in the Rush livelock
the driver has a backstop for. `--argmax` is there for a best-play read.
"""

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.features import D_ACTION, encode_actions, encode_state   # noqa: E402
from raifuwars_rl.features import D_ACTION, D_STATE                       # noqa: E402
from raifuwars_rl.policy import ActionScorer                               # noqa: E402
from raifuwars_rl.simenv import SelfPlaySimEnv                             # noqa: E402

BUILTIN = ("greedy", "random")


def load(path, device):
    """Build the net the checkpoint was TRAINED as, not the one the defaults describe.

    The arms are not one architecture -- ppo-bignet is 256/128 -- and calling ActionScorer() at
    its defaults throws a shape error on every layer. Worse is the case that does not throw: a
    checkpoint trained with RW_FEAT_COVER on is 35/28 wide, and in a process whose encoder is
    33/27 it cannot be run at all. Checked here rather than left to produce numbers.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    d_state = sd["state_tower.0.weight"].shape[1]
    d_action = sd["action_tower.0.weight"].shape[1]
    hidden = sd["state_tower.0.weight"].shape[0]
    embed = sd["state_tower.2.weight"].shape[0]
    if (d_state, d_action) != (D_STATE, D_ACTION):
        raise SystemExit(
            "%s was trained at %d/%d features but this process encodes %d/%d. It needs "
            "RW_FEAT_COVER=%s, and it cannot share an arena with checkpoints of the other width."
            % (path, d_state, d_action, D_STATE, D_ACTION, "1" if d_state > 33 else "0"))
    net = ActionScorer(d_state=d_state, d_action=d_action, hidden=hidden, embed=embed).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def blank():
    return {"played": 0, "won": 0, "tier": 0, "stars": 0, "kills": 0, "ret": 0.0, "eps": 0,
            "seat_played": collections.Counter(), "seat_won": collections.Counter()}


def run_rotation(rot, agents, nets, args, device):
    """One process, one seat->agent assignment, matches/n matches.

    Tallies into a LOCAL dict and returns it. `d[k] += 1` from four threads is not atomic in
    CPython and a lost update here would be a wrong win rate -- which is the one number this whole
    file is for, and the one nobody would think to doubt.
    """
    out = collections.defaultdict(blank)
    out["_timeouts"] = 0
    n = len(agents)
    # SEATS MAP TO SLOTS, NOT TO AGENT NAMES. `--agents ckpt,greedy,greedy,greedy` -- one
    # checkpoint against three built-ins, which is the single-seat training configuration -- is
    # a thing worth measuring, and keying the tally by name would merge those three into one
    # bucket with three times the matches.
    assign = {s: (s + rot) % n for s in range(n)}
    learners = [s for s in range(n) if agents[assign[s]] not in BUILTIN]
    builtin = next((a for a in agents if a in BUILTIN), "greedy")
    maps = [(m.rsplit(":", 1)[0], int(m.rsplit(":", 1)[1])) for m in args.maps.split(",")]

    env = SelfPlaySimEnv(driver=args.sim, hemlock=args.hemlock, maps=maps,
                         seats=list(range(n)), policy=builtin,
                         learners=",".join(str(s) for s in learners),
                         seed=args.seed + rot * 104729)
    want = args.matches // n
    rets = {}
    ob = env.reset()
    while env.matches < want:
        seat = ob.info["seat"]
        net = nets[agents[assign[seat]]]
        sv = np.nan_to_num(encode_state(ob.state), nan=0.0, posinf=0.0, neginf=0.0)
        am = (encode_actions(ob.state, ob.actions) if ob.actions
              else np.zeros((1, D_ACTION), dtype=np.float32))
        am = np.nan_to_num(am, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            logits = net(torch.tensor(sv, device=device), torch.tensor(am, device=device))
            if not torch.all(torch.isfinite(logits)):
                logits = torch.zeros_like(logits)
            if args.argmax:
                j = int(torch.argmax(logits).item())
            else:
                j = int(torch.multinomial(torch.softmax(logits / args.temp, dim=0), 1).item())
        step = env.step(j)
        # EPISODE RETURN PER AGENT, on the same `default_reward` the training loop sums. It is
        # what makes a training curve and an arena result the same kind of number -- and it is
        # how the claim "17.7 is unreachable under self-play" gets tested rather than argued:
        # run a checkpoint here against three built-ins, which is the single-seat training
        # configuration, and see whether its return comes back at the value that run reported.
        for (s, r, d) in step.info["credits"]:
            rets[s] = rets.get(s, 0.0) + r
            if d:
                slot = assign[s]
                out[slot]["ret"] += rets.pop(s, 0.0)
                out[slot]["eps"] += 1
        end = step.info.get("end")
        if end is not None:
            for row in end["seats"]:
                a = assign[row["seat"]]
                out[a]["played"] += 1
                out[a]["won"] += 1 if row["won"] else 0
                out[a]["tier"] += row["tier"]
                out[a]["stars"] += row["stars"]
                out[a]["kills"] += row["kills"]
                out[a]["seat_played"][row["seat"]] += 1
                out[a]["seat_won"][row["seat"]] += 1 if row["won"] else 0
            out["_timeouts"] += 1 if end.get("timeout") else 0
        ob = step
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--hemlock", default="/usr/local/bin/hemlock")
    ap.add_argument("--agents", required=True,
                    help="comma-separated, one per seat before rotation: checkpoint paths, or "
                         "the built-in names 'greedy' / 'random'")
    ap.add_argument("--names", default="", help="display names, comma-separated")
    ap.add_argument("--maps", default="Arboretum:4,Crossroads:4,Dustbowl:4,Glacier:4,"
                                      "Cornfield:4,Trench Warfare:4")
    ap.add_argument("--matches", type=int, default=400)
    # A SEED NO TRAINING RUN HAS SEEN. The board rotation, the character roster and every die roll
    # come from it, so evaluating on a training seed measures memorisation of specific matches.
    ap.add_argument("--seed", type=int, default=6060842)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--argmax", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    # `policy=` is one driver-wide setting, so the built-in seats all play the SAME thing.
    # Several seats of the same built-in are fine; two different ones are not representable.
    if len({a for a in agents if a in BUILTIN}) > 1:
        ap.error("greedy and random cannot share a board: `policy=` is one driver-wide setting")
    names = [x.strip() for x in args.names.split(",")] if args.names else agents
    nets = {a: (None if a in BUILTIN else load(a, device)) for a in set(agents)}

    n = len(agents)
    print("[arena] %d agents, %d matches, %d rotations, seed %d, %s"
          % (n, args.matches, n, args.seed, "argmax" if args.argmax else "sampled"), flush=True)
    with ThreadPoolExecutor(max_workers=n) as pool:
        parts = list(pool.map(lambda r: run_rotation(r, agents, nets, args, device), range(n)))

    out = collections.defaultdict(blank)
    out["_timeouts"] = 0
    for part in parts:
        out["_timeouts"] += part["_timeouts"]
        for a, d in part.items():
            if a == "_timeouts":
                continue
            for k in ("played", "won", "tier", "stars", "kills", "ret", "eps"):
                out[a][k] += d[k]
            out[a]["seat_played"].update(d["seat_played"])
            out[a]["seat_won"].update(d["seat_won"])

    rows = []
    print("\n%-24s %8s %8s %8s %7s %8s %7s"
          % ("agent", "matches", "win%", "return", "tier", "stars", "kills"))
    for slot, nm in enumerate(names):
        d = out[slot]
        p = max(1, d["played"])
        print("%-24s %8d %7.1f%% %8.3f %7.2f %8.0f %7.2f"
              % (nm, d["played"], 100 * d["won"] / p, d["ret"] / max(1, d["eps"]),
                 d["tier"] / p, d["stars"] / p, d["kills"] / p))
        rows.append({"agent": agents[slot], "name": nm, "matches": d["played"],
                     "winrate": d["won"] / p, "return": d["ret"] / max(1, d["eps"]),
                     "tier": d["tier"] / p,
                     "by_seat": {str(s): round(d["seat_won"][s] / max(1, d["seat_played"][s]), 3)
                                 for s in sorted(d["seat_played"])}})
    print("\nwin rate by seat (each agent plays every seat an equal share):")
    for r in rows:
        print("  %-32s %s" % (r["name"], r["by_seat"]))
    print("\ntimeouts (no winner inside maxturns): %d" % out["_timeouts"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
