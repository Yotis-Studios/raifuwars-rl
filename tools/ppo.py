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
from raifuwars_rl.features import (ACTION_FIELDS, D_ACTION, STATE_FIELDS,  # noqa: E402
                                   encode_actions, encode_state)
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
    ap.add_argument("--seats", default="0,1,2,3",
                    help="seat rotation -- which start base the agent occupies, varied per "
                         "episode. Turn order is a fixed rotation and spawns differ, so a policy "
                         "parked on one seat is being trained on that seat.")
    ap.add_argument("--maps", default="",
                    help="comma-separated boards, cycled across envs -- training on one map is "
                         "how the 4B ended up losing 20pp of move accuracy off its trained board")
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

    # ONE MAP PER ENV, CYCLED. Every spatial failure measured on this game came from training on
    # a single board: the 4B held 100% on every categorical family off-map and lost move 78->58%
    # and rush 50->24%. An RL policy fed one board learns that board's geometry for free and has
    # no reason not to.
    # "Name:seats" -- the roster size is a property of the BOARD (its start-base count), so it
    # belongs beside the name rather than in a global seat list that cannot know which map it is
    # being applied to.
    maps = []
    for entry in (args.maps.split(",") if args.maps else []):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, n = entry.rsplit(":", 1)
            maps.append((name.strip(), int(n)))
        else:
            maps.append((entry, 4))
    if not maps and args.map:
        maps = [(args.map, 4)]

    combos = [(name, seat) for name, n in maps for seat in range(n)]
    if not combos:
        combos = [(None, 0)]
    seats = sorted({c[1] for c in combos})
    # EVERY env gets the WHOLE pool and rotates through it per episode, offset so they are on
    # different boards at any given moment. Handing env i only map i would fix each env to one
    # board for the whole run -- four conditions rather than coverage of four.
    envs = [WarriorEnv(runner=args.runner, game=args.game, port=args.port + i,
                       combos=combos, rotate_offset=(i * 3) % max(1, len(combos)),
                       seed_base=900000 + i * 100000)
            for i in range(args.envs)]
    print("[ppo] %d envs rotating over %d (map, seat) pairs: %s"
          % (args.envs, len(combos),
             ", ".join("%s#%d" % c for c in combos[:6]) + (" ..." if len(combos) > 6 else "")),
          flush=True)
    pool = ThreadPoolExecutor(max_workers=args.envs)

    print("[ppo] resetting %d envs (each launches a real match)" % args.envs, flush=True)
    obs = list(pool.map(lambda e: e.reset(), envs))

    ep_returns = [0.0] * args.envs
    finished = collections.deque(maxlen=100)
    best = -1e9
    t_end = time.time() + args.hours * 3600.0
    update = 0
    total_steps = 0
    nonfinite = collections.Counter()
    logf = open(os.path.join(args.out, "train.jsonl"), "a", encoding="utf-8")

    while time.time() < t_end:
        update += 1
        rolls = [Rollout() for _ in range(args.envs)]

        # EACH ENV COLLECTS ITS OWN ROLLOUT, IN ITS OWN THREAD. This was a barrier: every step
        # ran `pool.map` over all envs, so each decision waited on the SLOWEST env -- and worse,
        # when a match ended the replacement game was launched INLINE, several seconds of process
        # start-up during which the other three envs sat blocked at the barrier doing nothing.
        #
        # Stepping independently means a reset stalls only the env that is resetting. Nothing here
        # needs synchronising: the rollouts are per-env by construction (GAE is computed per
        # trajectory precisely so they are never mixed), and the network is read-only during
        # collection -- no gradients, no mutation, and torch releases the GIL for the forward.
        def collect(i):
            ob = obs[i]
            roll = rolls[i]
            for _ in range(args.steps):
                sv = encode_state(ob.state)
                am = (encode_actions(ob.state, ob.actions) if ob.actions
                      else np.zeros((1, D_ACTION), dtype=np.float32))

                # Reported by name the first few times rather than quietly repaired -- a column
                # that goes non-finite on live states but not on the corpus is a real difference
                # between the two distributions.
                if not np.all(np.isfinite(sv)):
                    bad = [STATE_FIELDS[k] for k in np.where(~np.isfinite(sv))[0]]
                    nonfinite["state"] += 1
                    if nonfinite["state"] <= 3:
                        print("[ppo] non-finite STATE %s -- sanitising" % bad, flush=True)
                    sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(am)):
                    cols = sorted({int(c) for c in np.where(~np.isfinite(am))[1]})
                    nonfinite["action"] += 1
                    if nonfinite["action"] <= 3:
                        print("[ppo] non-finite ACTION %s -- sanitising"
                              % [ACTION_FIELDS[c] for c in cols if c < len(ACTION_FIELDS)],
                              flush=True)
                    am = np.nan_to_num(am, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    st = torch.tensor(sv, device=device)
                    at = torch.tensor(am, device=device)
                    logits = net(st, at)
                    if not torch.all(torch.isfinite(logits)):
                        nonfinite["logits"] += 1
                        logits = torch.zeros_like(logits)
                    probs = torch.softmax(logits, dim=0)
                    j = int(torch.multinomial(probs, 1).item())
                    lp = float(torch.log(probs[j] + 1e-9))
                    v = float(net.value_of(st))

                step = envs[i].step(j)
                roll.add(sv, am, j, lp, v, step.reward, step.done)
                ep_returns[i] += step.reward
                if step.done:
                    finished.append(ep_returns[i])
                    ep_returns[i] = 0.0
                    ob = envs[i].reset()
                else:
                    ob = step
            obs[i] = ob

        list(pool.map(collect, range(args.envs)))
        total_steps += args.envs * args.steps

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
                # ENTROPY OVER PADDED ROWS IS A NaN FACTORY, and `nan_to_num` on the result does
                # not save you: padding is -inf, so exp(-inf) * -inf is 0 * -inf = nan, and
                # zeroing the VALUE in the forward pass leaves the nan to arrive through the
                # BACKWARD pass anyway. One update was enough to turn every weight to nan, after
                # which every logit was nan, every action was sampled uniformly, and the run
                # carried on looking like it was training -- returns still printed, checkpoints
                # still written, kl quietly NaN.
                #
                # Clamping BEFORE the product keeps it finite end to end: exp(-1e9) underflows to
                # 0 and 0 * -1e9 is 0, with no nan for autograd to propagate.
                lp_safe = torch.nan_to_num(lp, neginf=-1e9)
                ent = -(lp_safe.exp() * lp_safe).sum(1).mean()

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
               "nonfinite": dict(nonfinite),
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
