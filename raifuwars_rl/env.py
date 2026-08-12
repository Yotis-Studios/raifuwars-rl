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
    # KEEP-ALIVE AND NO NAGLE. BaseHTTPRequestHandler speaks HTTP/1.0 by default, so the socket
    # is torn down and rebuilt for EVERY decision -- a full TCP handshake per action, thousands of
    # times a match. HTTP/1.1 reuses the connection (Content-Length is already sent on every
    # reply, which is what makes that safe).
    #
    # disable_nagle_algorithm sets TCP_NODELAY. Nagle holds a small write back waiting for more
    # data while the peer's delayed-ACK holds the acknowledgement waiting for a response -- the
    # classic ~40ms stall, and a request/response pair of a few hundred bytes is exactly the
    # shape that triggers it.
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

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
                 reward=None, launch_timeout=180.0, maps=None, seats=None, rotate_offset=0,
                 combos=None, reset_retries=3):
        """`maps` and `seats` VARY PER EPISODE, which is not the same as varying per env.

        Fixing a board and a seat at construction means env 0 plays Arboretum from seat 0 for the
        entire run: four envs give four fixed conditions, not coverage of them. The policy then
        has every opportunity to learn one board's geometry and one seat's turn position, and
        nothing in the return would reveal it -- this is the same failure that cost the 4B twenty
        points of move accuracy off its trained board.

        Rotated by episode and offset per env, so the four envs are on different boards at any
        moment (no thundering herd on one map) and each sees all of them over time. Deterministic
        rather than random: a surprising episode can be replayed.
        """
        self.runner = runner
        self.game = game
        self.port = port
        # (map, seat) PAIRS, NOT TWO INDEPENDENT LISTS. A seat only exists if the map seats it:
        # Twin Rivers has two start bases, so rotating a seat index 0..3 across it asks for seat 2
        # of a two-player board. Nothing rejects that -- the warrior is simply never installed, the
        # match plays itself out with the built-in AI, and the env sees a game that exited without
        # ever asking for a decision. It killed a 9-hour run 25 minutes in.
        #
        # Pairing them makes the invalid combination unrepresentable rather than merely unlikely,
        # which is the same argument as indexing the offered action list instead of naming an id.
        if combos:
            self.combos = [(m, int(t)) for m, t in combos]
        else:
            ms = list(maps) if maps else ([map_name] if map_name else [None])
            ts = list(seats) if seats else [seat]
            self.combos = [(m, t) for t in ts for m in ms]
        self.rotate_offset = rotate_offset
        self.map_name, self.seat = self.combos[0]
        self.seed_base = seed_base
        self.launch_timeout = launch_timeout
        self.reset_retries = reset_retries
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
        # Chosen here rather than in reset() so the values used are exactly the ones handed to the
        # process, with no window in which self.seat says one thing and the game was told another.
        # One step through the pre-paired list per episode, offset per env so four envs are on
        # four different boards at any moment rather than all reaching Twin Rivers together.
        k = self._episode + self.rotate_offset
        self.map_name, self.seat = self.combos[k % len(self.combos)]

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
        # CWD IS THE GAME'S DIRECTORY, NOT THE LEARNER'S. A GameMaker build resolves its included
        # files relative to the working directory, so launching it from anywhere else gives a
        # runner that starts, finds no datafiles, and exits before it ever asks for a decision --
        # which arrives here as "game exited before asking for a decision" and reads like a
        # protocol fault. Verified by running the same Runner by hand from the build directory,
        # where it plays a full match.
        self._proc = subprocess.Popen(
            [self.runner, "-game", self.game],
            cwd=os.path.dirname(os.path.abspath(self.game)) or None,
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
        """Retries before giving up. A single failed launch used to end the run: PPO calls reset()
        inline whenever a match finishes, so one bad episode at hour four takes the other eight
        hours with it. Retrying costs a minute; not retrying cost a night."""
        last = None
        for attempt in range(max(1, self.reset_retries)):
            try:
                return self._reset_once()
            except (RuntimeError, TimeoutError) as e:                # noqa: PERF203
                last = e
                print("[env] reset failed on %s seat %s (attempt %d/%d): %s"
                      % (self.map_name, self.seat, attempt + 1, self.reset_retries, e), flush=True)
                self._kill()
                # Advance the rotation rather than retrying the identical launch -- if this
                # combination is the problem, repeating it just fails three times.
                self._episode += 1
        raise last

    def _reset_once(self):
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


def _progress(me):
    """How far this seat is toward its NEXT tier, in 0..1. Saturating, on purpose."""
    tp = (me or {}).get("tier_progress") or {}
    try:
        need = float(tp.get("need", 0))
        have = float(tp.get("have", 0))
    except (TypeError, ValueError):
        return 1.0
    if need <= 0:                      # tier 4, or the game could not name a threshold
        return 1.0
    return min(1.0, max(0.0, have / need))


def default_reward(prev, cur, end_payload):
    """TIERS ARE THE REWARD. Everything else is shaped toward reaching the next one.

    Reaching tier 4 ends the match and wins it; nothing else does. So a tier gained is the event
    worth paying for, and the last one is worth most because it is the one that wins -- hence the
    escalating scale rather than a flat rate.

    THE PROGRESS TERM IS WHY STARS ARE NOT REWARDED DIRECTLY. `tier_progress` is have/need against
    this seat's own track -- stars or KOs, whichever it is on -- and it SATURATES at 1.0. Earning
    stars you already have enough of is worth exactly zero, which is true: at the threshold the
    only useful move is to walk to your base and tier. A raw stars term cannot express that, and a
    policy given one farms points forever.

    That is not hypothetical here. A fine-tuned LLM on this game accumulated 838 stars a match --
    against the built-in AI's 1683 -- reached tier 2.05, and won 2 matches in 40. It had found the
    means and stopped. An RL agent rewarded the same way will find it faster and hold it harder.

    Potential-based in shape: the progress term is a difference of a bounded function of state, so
    it moves the agent toward thresholds without changing which policy is optimal.
    """
    if end_payload is not None:
        return 10.0 if end_payload.get("won") else 0.0
    if not prev or not cur:
        return 0.0
    a, b = prev.get("self") or {}, cur.get("self") or {}

    t0, t1 = float(a.get("tier", 0)), float(b.get("tier", 0))
    r = 0.0
    if t1 > t0:
        # 1 -> 2 -> 3 -> 5 for tiers 1..4. The fourth is the win, and a flat rate would price the
        # decisive tier the same as the first routine one.
        for t in range(int(t0) + 1, int(t1) + 1):
            r += {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}.get(t, 1.0)

    # Progress toward the next threshold. Small: it is a hint about direction, not a goal. On a
    # tier-up `have` resets against a larger `need`, so this term goes sharply negative exactly
    # when the tier bonus fires -- suppressed, or the agent is punished for the thing it is for.
    if t1 == t0:
        r += 1.0 * (_progress(b) - _progress(a))
    return r


# ---------------------------------------------------------------------------
# KNOWN BLOCKER (Aug 12): the bare-Runner path does not install the warrior seat.
#
# Launching `Runner.exe -game RaifuWars.win` with RW_WARRIOR_URL / RW_WARRIOR_SEATS set plays a
# complete match -- `[aitrial] RESULT ... policy=classic turns=86` -- and never contacts the
# sidecar. `warriorSeatUrl` logs "[warrior] sidecar at <url> for seats <n>" the first time it
# resolves, and that line does not appear, so RW_WARRIOR_URL is reaching the process as undefined.
# Confirmed from the other side too: a sidecar left listening on the port logs zero requests.
#
# The tournament harness works because it goes through `igor_run Run`, which sets up the
# environment differently. So this is about how the variables reach a bare Runner, not about the
# protocol, the port, or this file's threading.
#
# WHERE TO LOOK FIRST. `RW_MAP` failed twice before for the same class of reason -- a bash prefix
# that was not an assignment, then the wrong variable name -- so check what the process actually
# receives before changing anything here:
#
#   1. print `environment_get_variable("RW_WARRIOR_URL")` at boot, or run the Runner from a shell
#      that has the variables EXPORTED rather than prefixed, and see whether the [warrior] line
#      appears
#   2. compare against `igor_run`'s environment (tools/gm-env.sh) for anything else it sets
#   3. only then suspect this file
#
# Until it is resolved, PPO cannot collect. `tools/serve.py` measures a checkpoint through the
# igor_run tournament harness and is unaffected.
