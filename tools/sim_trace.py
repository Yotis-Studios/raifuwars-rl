"""Drive SimEnv with a policy and write the states it produced, in corpus shape.

    python tools/sim_trace.py --driver …/raifusim/driver.hml --n 5000 --out runs/sim-traces.jsonl

WHY THIS EXISTS: `tools/check_features.py` is the only tool that can tell you the serializer is
wrong, and it reads a .jsonl of `{"state": …, "available_actions": […]}` rows. Running it over SIM
states and over REAL ones and putting the two tables side by side is the check -- a column that is
constant or non-finite on sim states and alive on real ones is a field the serializer is not
writing, which `features.py` turns into 0.0 with no error and no crash.

It also reports throughput and episode returns under `default_reward`, because both are free here
and both are things you want to know before starting a run rather than after.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.simenv import SimEnv                          # noqa: E402


def parse_maps(s):
    out = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, n = entry.rsplit(":", 1)
            out.append((name.strip(), int(n)))
        else:
            out.append((entry, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--hemlock", default="/usr/local/bin/hemlock")
    ap.add_argument("--maps", default="Dustbowl:4,Glacier:4,Crossroads:4,Arboretum:4")
    ap.add_argument("--seats", default="0,1,2,3")
    ap.add_argument("--policy", default="random", help="what the OTHER seats play")
    ap.add_argument("--n", type=int, default=5000, help="agent decisions to collect")
    ap.add_argument("--seed", type=int, default=20250813)
    ap.add_argument("--out", default="")
    ap.add_argument("--checkpoint", default="", help="score with this policy instead of uniform")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    env = SimEnv(driver=args.driver, hemlock=args.hemlock,
                 maps=parse_maps(args.maps),
                 seats=[int(s) for s in args.seats.split(",") if s.strip()],
                 seed=args.seed, policy=args.policy)

    net = None
    if args.checkpoint:
        import torch
        from raifuwars_rl.features import D_ACTION, encode_actions, encode_state
        from raifuwars_rl.policy import ActionScorer
        net = ActionScorer()
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["model"] if "model" in blob else blob)
        net.eval()

    rng = np.random.default_rng(args.seed)
    fh = open(args.out, "w", encoding="utf-8") if args.out else None

    obs = env.reset()
    ep_return = 0.0
    returns, wins, tiers, lengths = [], [], [], []
    ep_len = 0
    n_offered = []
    t0 = time.time()
    for _ in range(args.n):
        if fh is not None:
            fh.write(json.dumps({"state": obs.state,
                                 "available_actions": obs.actions}) + "\n")
        n_offered.append(len(obs.actions))

        if net is not None:
            import torch
            with torch.no_grad():
                sv = torch.tensor(encode_state(obs.state))
                am = torch.tensor(encode_actions(obs.state, obs.actions)
                                  if obs.actions else np.zeros((1, D_ACTION), dtype=np.float32))
                probs = torch.softmax(net(sv, am), dim=0)
                j = int(torch.multinomial(probs, 1).item())
        else:
            j = int(rng.integers(len(obs.actions)))

        step = env.step(j)
        ep_return += step.reward
        ep_len += 1
        if step.done:
            returns.append(ep_return)
            lengths.append(ep_len)
            info = (step.info or {}).get("payload") or {}
            wins.append(1.0 if info.get("won") else 0.0)
            tiers.append(float(info.get("tier", 0)))
            ep_return = 0.0
            ep_len = 0
            obs = env.reset()
        else:
            obs = step
    dt = time.time() - t0
    env.close()
    if fh is not None:
        fh.close()

    print("[sim] %d agent decisions in %.1fs = %.1f steps/sec" % (args.n, dt, args.n / dt))
    print("[sim] offered actions per decision: mean %.1f  min %d  max %d"
          % (float(np.mean(n_offered)), int(np.min(n_offered)), int(np.max(n_offered))))
    if returns:
        print("[sim] %d finished episodes: mean return %.3f (sd %.3f, min %.3f, max %.3f)"
              % (len(returns), float(np.mean(returns)), float(np.std(returns)),
                 float(np.min(returns)), float(np.max(returns))))
        print("[sim] win rate %.3f   mean final tier %.2f   mean episode length %.0f decisions"
              % (float(np.mean(wins)), float(np.mean(tiers)), float(np.mean(lengths))))
    else:
        print("[sim] no episode finished within %d decisions" % args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
