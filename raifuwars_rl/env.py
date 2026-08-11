"""A gym-shaped environment over a real Raifu Wars match.

    env = WarriorEnv(runner=..., game=..., port=8931)
    obs = env.reset()
    while not obs.done:
        obs = env.step(action_index)

CONTROL IS INVERTED, AND THAT IS THE WHOLE DESIGN PROBLEM. In the Warrior protocol the GAME is the
HTTP client and the sidecar is the server: the match runs on its own clock and calls `/v1/act` when
it wants a decision. A learner wants the opposite -- to call `step(a)` and be given the next state.

So this runs the sidecar in a background thread and hands the request across two queues. The HTTP
handler blocks holding the game's turn open until the learner answers, which is safe precisely
because Raifu Wars has no per-action clock: `nextTurn` is called explicitly, so a seat that has not
answered is simply a seat that has not moved yet. That is the same property that lets an LLM take
30 seconds to think.

ONE PROCESS PER ENV. The game is a real GameMaker match, launched per episode. Throughput measured
on this machine: ~8 steps/sec headless for one instance, ~35 across six. So a vectorised setup is
N of these, not one clever process.

THE ACTION SPACE IS THE OFFERED LIST. `step` takes an INDEX into `obs.actions`, never an action_id
string and never a fixed-size head. The legal set varies from 2 to ~670 between decisions, the game
has already computed it, and indexing it means an illegal action is unrepresentable rather than
merely penalised. This is also why the policy must score actions rather than classify over them.
"""

import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Step:
    """One decision point, or the end of an episode."""

    __slots__ = ("state", "actions", "reward", "done", "info")

    def __init__(self, state=None, actions=None, reward=0.0, done=False, info=None):
        self.state = state or {}
        self.actions = actions or []
        self.reward = reward
        self.done = done
        self.info = info or {}


class _Handler(BaseHTTPRequestHandler):
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
        self._send(200, {"protocol_version": "0.1", "name": "raifuwars-rl",
                         "policy": "rl", "capabilities": {"vision": False, "chat": False,
                                                          "commentary": False,
                                                          "max_deadline_ms": 3600000}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        path = self.path.rstrip("/")
        env = self.server.env

        if path == "/v1/act":
            # Hand the decision to the learner and block until it answers. The turn stays open
            # meanwhile, which is a property of the game rather than a trick: there is no
            # per-action clock, so an unanswered seat has simply not moved yet.
            env._to_agent.put(("act", req))
            action_id = env._from_agent.get()
            self._send(200, {"action_id": action_id, "args": {}})
        elif path == "/v1/match/end":
            env._to_agent.put(("end", req))
            self._send(200, {"ok": True})
        else:
            self._send(200, {"ok": True})


class WarriorEnv:
    def __init__(self, runner, game, port=8931, seat=0, map_name=None, seed_base=500000,
                 reward=None, launch_timeout=180.0):
        self.runner = runner
        self.game = game
        self.port = port
        self.seat = seat
        self.map_name = map_name
        self.seed_base = seed_base
        self.launch_timeout = launch_timeout
        self.reward_fn = reward or default_reward

        self._to_agent = queue.Queue()
        self._from_agent = queue.Queue()
        self._proc = None
        self._episode = 0
        self._prev = None
        self._acts = []

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._httpd.env = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    # -- lifecycle ---------------------------------------------------------

    def _launch(self):
        env = dict(os.environ)
        env.update({
            "RW_UI_CAPTURE": "1",
            "RW_UI_CAPTURE_FLOW": "ai",
            "RW_UI_CAPTURE_SIZE": "640x360",
            "RW_SEED": str(self.seed_base + self._episode * 7919),
            "RW_WARRIOR_URL": "http://127.0.0.1:%d" % self.port,
            "RW_WARRIOR_SEATS": str(self.seat),
        })
        if self.map_name:
            # RW_UI_CAPTURE_MAP is the name the game reads; RW_MAP is the human-facing alias that
            # only ui-capture.sh translates. Getting this wrong silently plays the default map
            # while reporting the one you asked for.
            env["RW_UI_CAPTURE_MAP"] = self.map_name
        self._proc = subprocess.Popen(
            [self.runner, "-game", self.game],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        self._kill()
        try:
            self._httpd.shutdown()
        except Exception:                                       # noqa: BLE001
            pass

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except Exception:                                   # noqa: BLE001
                pass
        self._proc = None

    def _drain(self):
        while True:
            try:
                self._to_agent.get_nowait()
            except queue.Empty:
                return

    # -- gym surface -------------------------------------------------------

    def reset(self):
        self._kill()
        self._drain()
        self._prev = None
        self._episode += 1
        self._launch()

        deadline = time.time() + self.launch_timeout
        while True:
            try:
                kind, req = self._to_agent.get(timeout=1.0)
            except queue.Empty:
                if time.time() > deadline:
                    self._kill()
                    raise TimeoutError("game never asked for a decision within %.0fs"
                                       % self.launch_timeout)
                if self._proc.poll() is not None:
                    raise RuntimeError("game exited before asking for a decision")
                continue
            if kind == "act":
                self._prev = req.get("state") or {}
                # Held on the env, because `step` needs the list it is indexing into and the
                # caller is trusted to pass an index rather than an id. Not storing it here was
                # the first bug: reset() handed out actions and step() had none to index.
                self._acts = req.get("available_actions") or []
                return Step(state=self._prev, actions=self._acts)

    def step(self, index):
        actions = self._last_actions()
        if not 0 <= index < len(actions):
            raise IndexError("action %d out of range for %d offered" % (index, len(actions)))
        self._from_agent.put(actions[index]["action_id"])

        while True:
            try:
                kind, req = self._to_agent.get(timeout=5.0)
            except queue.Empty:
                if self._proc.poll() is not None:
                    # The match finished and the process exited without a match/end. Terminal
                    # either way -- and the reward for the final transition is unknown, so it is
                    # zero rather than invented.
                    return Step(state=self._prev, done=True, info={"reason": "process exited"})
                continue
            if kind == "end":
                return Step(state=self._prev, reward=self.reward_fn(self._prev, None, req),
                            done=True, info={"reason": "match end", "payload": req})
            state = req.get("state") or {}
            r = self.reward_fn(self._prev, state, None)
            self._prev = state
            self._acts = req.get("available_actions") or []
            return Step(state=state, actions=self._acts, reward=r)

    def _last_actions(self):
        return getattr(self, "_acts", [])


def default_reward(prev, cur, end_payload):
    """Shaped on TIER, lightly on stars, because tiering is the only thing that wins.

    The failure mode this is arranged against has already been observed in a trained LLM on this
    exact game: it accumulated 838 stars a match -- half the built-in AI's rate -- reached tier
    2.05, and won 2 matches in 40. Stars are a means; a policy rewarded on them will farm points
    and never climb. So tier gains dominate and stars are worth little.
    """
    if end_payload is not None:
        return 10.0 if end_payload.get("won") else 0.0
    if not prev or not cur:
        return 0.0
    a, b = prev.get("self") or {}, cur.get("self") or {}
    d_tier = float(b.get("tier", 0)) - float(a.get("tier", 0))
    d_stars = float(b.get("stars", 0)) - float(a.get("stars", 0))
    return 3.0 * d_tier + 0.001 * d_stars
