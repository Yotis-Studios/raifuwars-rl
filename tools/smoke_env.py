"""Play one real match through WarriorEnv with a random policy.

    python tools/smoke_env.py --runner <Runner.exe> --game <RaifuWars.win>

The point is not the policy -- it is that the environment steps a real GameMaker match end to end,
with control inverted, and terminates. Every RL result downstream rests on this working, and a
random legal walk is the cheapest thing that exercises it.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.env import WarriorEnv                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--map", default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    env = WarriorEnv(runner=args.runner, game=args.game, port=args.port, map_name=args.map)
    rng = random.Random(0)
    try:
        for ep in range(args.episodes):
            t0 = time.time()
            obs = env.reset()
            steps, total = 0, 0.0
            sizes = []
            while not obs.done and steps < args.max_steps:
                sizes.append(len(obs.actions))
                obs = env.step(rng.randrange(len(obs.actions)))
                total += obs.reward
                steps += 1
            dt = time.time() - t0
            me = (obs.state or {}).get("self") or {}
            print("episode %d: %d steps in %.0fs (%.1f steps/s), reward %.2f, "
                  "final tier %s stars %s, legal set %d-%d, done=%s (%s)"
                  % (ep, steps, dt, steps / max(dt, 1e-9), total,
                     me.get("tier"), me.get("stars"),
                     min(sizes) if sizes else 0, max(sizes) if sizes else 0,
                     obs.done, obs.info.get("reason")))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
