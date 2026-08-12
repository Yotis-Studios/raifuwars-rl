"""PPO over real Raifu Wars matches, warm-started from behaviour cloning.

    python tools/ppo.py --init runs/bc-long.pt --envs 4 --hours 6 --out runs/ppo

WARM START IS NOT OPTIONAL HERE. The simulator gives ~1.6 agent decisions per second per
instance, so a night is on the order of 10^5 steps -- three orders short of what it takes to
discover, from scratch, that the way to win is to walk to a square you cannot see the importance
of and press tier. BC already reaches 68.7% agreement with a competent teacher; PPO's job is to
improve a working policy, not to find one.

THE ACTION SPACE IS THE OFFERED LIST, which shapes every part of this. The legal set runs from 2
to ~670 and changes every step, so:

  - the policy SCORES actions and softmaxes over exactly those offered (raifuwars_rl.policy)
  - rollouts store the action MATRIX per step, not an index into a fixed head
  - batching pads to the longest offer in the batch and masks with -inf, because padding that is
    merely zeroed still takes probability mass out of the softmax
  - "illegal action" is not a failure mode that exists, so no penalty term models it

ENVS ARE PROCESSES, AND THE STEP BLOCKS. Each env owns a real game, runs a sidecar thread, and
blocks the game's HTTP handler until the learner answers -- safe because Raifu Wars has no
per-action clock. So stepping N envs means N blocking calls in parallel: a thread pool, not a
loop. Threads are right despite the GIL because every one of them is waiting on IO.

WHAT IT CHECKPOINTS AND WHY SO OFTEN. An overnight run that dies at hour 5 with nothing on disk
has produced nothing. Every update writes `last.pt`, and the best mean-return-so-far writes
`best.pt`, so a crash costs minutes rather than a night.
"""

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.env import WarriorEnv                            # noqa: E402
from raifuwars_rl.features import D_ACTION, encode_actions, encode_state  # noqa: E402
from raifuwars_rl.policy import ActionScorer, masked_log_probs     # noqa: E402


class Rollout:
    """Flat storage. Offers differ in length per step, so action matrices stay a ragged list and
    are padded only at batch time -- padding 670 columns for every step would be mostly zeros."""

    def __init__(self):
        self.s, self.a, self.idx, self.logp, self.val, self.rew, self.done = ([] for _ in range(7))

    def add(self, s, a, idx, logp, val, rew, done):
        self.s.append(s); self.a.append(a); self.idx.append(idx)
        self.logp.append(logp); self.val.append(val); self.rew.append(rew); self.done.append(done)

    def __len__(self):
        return len(self.s)


def pad(mats, device):
    n = max(m.shape[0] for m in mats)
    out = torch.zeros(len(mats), n, mats[0].shape[1], device=device)
    lens = torch.tensor([m.shape[0] for m in mats], device=device)
    for i, m in enumerate(mats):
        out[i, :m.shape[0]] = torch.tensor(m, device=device)
    return out, lens


def gae(rews, vals, dones, last_val, gamma, lam):
    """Advantages per env-trajectory. `dones` cuts the bootstrap: a finished match carries no
    value into the next one, and an episode boundary treated as a transition is the classic way
    to teach a policy that dying is a way to reach a fresh board."""
    adv = np.zeros(len(rews), dtype=np.float32)
    run = 0.0
    nextval = last_val
    for t in reversed(range(len(rews))):
        nonterm = 0.0 if dones[t] else 1.0
        delta = rews[t] + gamma * nextval * nonterm - vals[t]
        run = delta + gamma * lam * nonterm * run
        adv[t] = run
        nextval = vals[t]
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="", help="BC checkpoint to warm-start from")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--port", type=int, default=8940)
    ap.add_argument("--steps", type=int, default=256, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--kl", type=float, default=0.03,
                    help="stop a policy epoch early past this KL from the old policy")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--map", default="")
    ap.add_argument("--runner", default=os.environ.get("RW_RUNNER", ""))
    ap.add_argument("--game", default=os.environ.get("RW_GAME", ""))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    net = ActionScorer().to(device)
    if args.init:
        blob = torch.load(args.init, map_location=device, weights_only=False)
        net.load_state_dict(blob["model"] if "model" in blob else blob)
        print("[ppo] warm-started from %s" % args.init, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    envs = [WarriorEnv(runner=args.runner, game=args.game, port=args.port + i,
                       seat=0, map_name=args.map or None, seed_base=900000 + i * 100000)
            for i in range(args.envs)]
    pool = ThreadPoolExecutor(max_workers=args.envs)

    print("[ppo] resetting %d envs (each launches a real match)" % args.envs, flush=True)
    obs = list(pool.map(lambda e: e.reset(), envs))

    ep_returns = [0.0] * args.envs
    finished = collections.deque(maxlen=100)
    best = -1e9
    t_end = time.time() + args.hours * 3600.0
    update = 0
    total_steps = 0
    logf = open(os.path.join(args.out, "train.jsonl"), "a", encoding="utf-8")

    while time.time() < t_end:
        update += 1
        rolls = [Rollout() for _ in range(args.envs)]

        for _ in range(args.steps):
            svecs = [encode_state(o.state) for o in obs]
            # A decision with no offered actions cannot happen -- the protocol forbids asking
            # with an empty legal set -- but a one-row zero matrix keeps a malformed payload from
            # taking the whole run down at hour four.
            amats = [encode_actions(o.state, o.actions) if o.actions
                     else np.zeros((1, D_ACTION), dtype=np.float32) for o in obs]
            with torch.no_grad():
                picks = []
                for i, (sv, am) in enumerate(zip(svecs, amats)):
                    s = torch.tensor(sv, device=device)
                    a = torch.tensor(am, device=device)
                    logits = net(s, a)
                    probs = torch.softmax(logits, dim=0)
                    j = int(torch.multinomial(probs, 1).item())
                    picks.append((j, float(torch.log(probs[j] + 1e-9)),
                                  float(net.value_of(s))))

            results = list(pool.map(lambda p: p[0].step(p[1]),
                                    [(envs[i], picks[i][0]) for i in range(args.envs)]))

            for i, step in enumerate(results):
                j, lp, v = picks[i]
                rolls[i].add(svecs[i], amats[i], j, lp, v, step.reward, step.done)
                ep_returns[i] += step.reward
                if step.done:
                    finished.append(ep_returns[i])
                    ep_returns[i] = 0.0
                    # A finished match must be replaced or the env has nothing to step into.
                    # Done inline rather than lazily: a stale Step would be silently re-stepped.
                    obs[i] = envs[i].reset()
                else:
                    obs[i] = step
            total_steps += args.envs

        # -- advantages, per env so trajectories are not mixed across boundaries -------------
        with torch.no_grad():
            last_vals = [float(net.value_of(torch.tensor(encode_state(o.state), device=device)))
                         for o in obs]

        S, A, IDX, LOGP, ADV, RET = [], [], [], [], [], []
        for i, r in enumerate(rolls):
            adv = gae(r.rew, r.val, r.done, last_vals[i], args.gamma, args.lam)
            S += r.s; A += r.a; IDX += r.idx; LOGP += r.logp
            ADV += list(adv); RET += list(adv + np.array(r.val, dtype=np.float32))

        S = torch.tensor(np.stack(S), device=device)
        IDX = torch.tensor(IDX, device=device)
        LOGP = torch.tensor(LOGP, dtype=torch.float32, device=device)
        ADV = torch.tensor(ADV, dtype=torch.float32, device=device)
        RET = torch.tensor(RET, dtype=torch.float32, device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        n = len(IDX)
        order = np.arange(n)
        stop_kl = 0.0
        for _ in range(args.epochs):
            np.random.shuffle(order)
            for k in range(0, n, args.batch):
                sl = order[k:k + args.batch]
                Ab, L = pad([A[t] for t in sl], device)
                lp = masked_log_probs(net(S[sl], Ab), L)
                newlp = lp.gather(1, IDX[sl][:, None]).squeeze(1)
                ratio = torch.exp(newlp - LOGP[sl])
                a1 = ratio * ADV[sl]
                a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ADV[sl]
                pg = -torch.min(a1, a2).mean()

                v = net.value_of(S[sl])
                vloss = F.mse_loss(v, RET[sl])
                ent = -(lp.exp() * lp).nan_to_num(0.0).sum(1).mean()

                loss = pg + args.vf * vloss - args.ent * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

                stop_kl = float((LOGP[sl] - newlp).mean())
            # Early stop on KL. Without it a few large batches can move the policy far enough
            # that the warm start is thrown away in the first hour, which reads as "PPO made it
            # worse" and is really "the step size was wrong".
            if abs(stop_kl) > args.kl:
                break

        mean_ret = float(np.mean(finished)) if finished else float("nan")
        rec = {"update": update, "steps": total_steps, "episodes": len(finished),
               "mean_return": mean_ret, "kl": stop_kl,
               "elapsed_min": round((time.time() - (t_end - args.hours * 3600)) / 60, 1)}
        print("[ppo] %s" % json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()

        torch.save({"model": net.state_dict(), "weight_mode": "ppo"},
                   os.path.join(args.out, "last.pt"))
        if finished and mean_ret > best:
            best = mean_ret
            torch.save({"model": net.state_dict(), "weight_mode": "ppo"},
                       os.path.join(args.out, "best.pt"))

    for e in envs:
        e.close()
    logf.close()
    print("[ppo] done: %d updates, %d steps, best mean return %.3f" % (update, total_steps, best),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
