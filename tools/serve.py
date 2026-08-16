"""Serve a trained checkpoint as a Warrior sidecar, so the game can play against it.

    python tools/serve.py runs/bc-none.pt --port 8899
    RW_WARRIOR_URL=http://127.0.0.1:8899 tools/warrior-tournament.sh 40

WHY THIS IS THE POINT. A validation top-1 of 65% says how often the policy agrees with the
built-in AI on held-out matches. It does not say whether it can play: agreement is measured on
states the TEACHER reached, and a policy that is 65% right and 35% wrong reaches states the
teacher never visited, where its agreement number says nothing at all. The only honest measure is
a match.

And it is measured on exactly the footing every LLM seat was measured on -- same harness, same
maps, same built-in opponent, same n. The scoreboard to beat is the 4B fine-tune's 2 wins in 40,
838 average stars, 0% illegal.

SPEAKS THE PROTOCOL, NOT A SHORTCUT. The game is the HTTP client; this is the server. `/v1/act`
carries the state and `available_actions`, and the reply is one `action_id` from that list. The
policy SCORES the offered actions and takes the best, so an illegal action is unrepresentable
rather than merely unlikely -- the same property that made the action-scoring architecture the
right choice in the first place.

TEMPERATURE IS OFF BY DEFAULT. A tournament is measuring the policy, and sampling adds variance
that has to be paid for in matches. `--temperature` exists because a deterministic policy against
a deterministic opponent on a seeded map can produce identical matches, which looks like a working
run and is one data point repeated.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.features import D_ACTION, D_STATE               # noqa: E402
from raifuwars_rl.features import encode_actions, encode_state    # noqa: E402
from raifuwars_rl.policy import ActionScorer                      # noqa: E402


class Server(HTTPServer):
    """SINGLE-THREADED, AND THAT IS THE FIX.

    This was a ThreadingHTTPServer, and its handler threads never exited: measured at ~7 surviving
    threads and 3.8 MB per request, which projects to 36 GB over one 160-match tournament. That is
    exactly what happened -- a sidecar reached 28-35 GB, the machine ran out, the game died in
    oInit with "Memory allocation failed", and because Snitch's crash dialog is modal the process
    then sat there while the tournament looked like a slow match for 43 minutes.

    An earlier fix set `block_on_close = False` on a theory about ThreadingMixIn retaining Thread
    objects. It was applied without measuring and it did not work; the next run leaked identically.
    tools/leak_probe.py exists so the next such claim is checked in a minute rather than a
    tournament.

    THREADS BOUGHT NOTHING HERE. The game is a sequential client: it asks for one decision and
    blocks until it gets it, so there is never a second request in flight for this server to
    overlap. Concurrency was answering a question nobody asked, and its failure mode cost a run.
    Multiple games mean multiple sidecars on multiple ports, which is what PPO already does.
    """


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0, the default. Keep-alive was tried and reverted: GameMaker's client does not hold
    # the connection, so every episode end reset a socket the server was still holding and threw
    # ConnectionResetError [WinError 10054] in a flood. It bought ~75ms a decision and cost a run.
    #
    # TCP_NODELAY stays -- it is the half that was actually helping. Nagle holds a small write
    # waiting for more data while the peer's delayed ACK waits for a response: the classic ~40ms
    # stall on request/response pairs this size.
    disable_nagle_algorithm = True

    def handle_one_request(self):
        # A game exiting mid-connection is NORMAL -- an episode ending is a process ending.
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {
            "protocol_version": "0.1",
            "name": self.server.tag,
            "policy": "action-scorer",
            "capabilities": {"vision": False, "chat": False, "commentary": False,
                             "max_deadline_ms": 60000},
        })

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n)) if n else {}
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return

        path = self.path.rstrip("/")
        if path != "/v1/act":
            # match/start, match/end and anything else: acknowledge and record nothing. The
            # tournament script reads its results from the game, not from here.
            self._send(200, {"ok": True})
            return

        state = req.get("state") or {}
        actions = req.get("available_actions") or []
        if not actions:
            # The protocol forbids asking with an empty legal set, so this is the game's bug and
            # not something to paper over with a guess. Reported rather than answered.
            self.server.stats["empty_offers"] += 1
            self._send(400, {"error": "no available_actions"})
            return

        t0 = time.time()
        try:
            aid, why = self.server.policy.choose(state, actions)
        except Exception as e:                                        # noqa: BLE001
            # A crash here stalls the match rather than ending it -- the game waits on a reply
            # with no clock. Falling back to the first legal action keeps the tournament moving
            # and the count makes the failure visible instead of silent.
            self.server.stats["errors"] += 1
            if self.server.stats["errors"] <= 3:
                print("[serve] choose() failed: %r -- falling back to first legal action" % e,
                      flush=True)
            aid, why = str(actions[0].get("action_id")), "fallback after error"

        self.server.stats["acts"] += 1
        self.server.stats["ms"] += (time.time() - t0) * 1000.0
        self._send(200, {"action_id": aid, "args": {}, "why": why})


class Policy:
    def __init__(self, ckpt_path, temperature=0.0, device="cpu"):
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

        # BUILD THE NET THE CHECKPOINT WAS TRAINED AS, read from its own first layer. The runs are
        # not one architecture -- `ppo-bignet` is 256/128 -- so the constructor defaults load some
        # checkpoints and throw on others.
        d_state = sd["state_tower.0.weight"].shape[1]
        d_action = sd["action_tower.0.weight"].shape[1]
        hidden = sd["state_tower.0.weight"].shape[0]
        embed = sd["state_tower.2.weight"].shape[0]

        # THE FEATURE WIDTH IS CHECKED BECAUSE IT FAILS LATE. A wrong hidden size raises here, at
        # load. A wrong RW_FEAT_COVER does not: the towers are built from the checkpoint, so
        # load_state_dict succeeds, and the mismatch first appears one layer into the first
        # decision as "mat1 and mat2 shapes cannot be multiplied". A host that catches policy
        # errors and falls back to a legal action -- which the reference sidecar does by design, so
        # a crash cannot hang a game -- will then play an entire match on fallbacks and still
        # produce a results table. The flag is fixed at the first import of `features` and cannot
        # be changed afterwards, so this is checked at startup and named in the message.
        if (d_state, d_action) != (D_STATE, D_ACTION):
            raise SystemExit(
                "%s was trained on %d/%d features but this process encodes %d/%d.\n"
                "Set RW_FEAT_COVER=%s before starting serve.py."
                % (ckpt_path, d_state, d_action, D_STATE, D_ACTION,
                   "1" if d_state > 33 else "0"))

        self.net = ActionScorer(d_state=d_state, d_action=d_action,
                                hidden=hidden, embed=embed).to(device)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.arch = "%d/%d feat, %d/%d wide, %s params" % (
            d_state, d_action, hidden, embed, "{:,}".format(sum(v.numel() for v in sd.values())))
        self.device = torch.device(device)
        self.temperature = temperature
        self.weight_mode = blob.get("weight_mode", "?") if isinstance(blob, dict) else "?"

    @torch.no_grad()
    def choose(self, state, actions):
        s = torch.tensor(encode_state(state), device=self.device)
        a = torch.tensor(encode_actions(state, actions), device=self.device)
        logits = self.net(s, a)
        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=0)
            i = int(torch.multinomial(probs, 1).item())
        else:
            i = int(torch.argmax(logits).item())

        # Belt and braces: the index came from a tensor whose length is len(actions) by
        # construction, but returning an id that was not offered is the one failure mode that
        # would be scored as an illegal action against the policy.
        i = max(0, min(i, len(actions) - 1))
        chosen = actions[i]
        return str(chosen.get("action_id")), "scored %d offered, took %s" % (
            len(actions), chosen.get("type", "?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="raifuwars-rl")
    args = ap.parse_args()

    policy = Policy(args.checkpoint, args.temperature, args.device)
    httpd = Server(("127.0.0.1", args.port), Handler)
    httpd.policy = policy
    httpd.tag = args.tag
    httpd.stats = {"acts": 0, "ms": 0.0, "errors": 0, "empty_offers": 0}

    # The architecture is printed because it is inferred rather than declared: if you loaded the
    # wrong checkpoint for the RW_FEAT_COVER you set, this line is where you see it.
    print("[serve] %s (%s, weight=%s, temperature=%g) on http://127.0.0.1:%d"
          % (os.path.basename(args.checkpoint), policy.arch, policy.weight_mode,
             args.temperature, args.port),
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        s = httpd.stats
        print("\n[serve] %d actions, %.1f ms avg, %d errors, %d empty offers"
              % (s["acts"], s["ms"] / max(1, s["acts"]), s["errors"], s["empty_offers"]),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
