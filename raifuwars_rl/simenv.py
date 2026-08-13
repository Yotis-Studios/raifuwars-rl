"""The same environment surface as WarriorEnv, over the Hemlock simulator instead of the game.

    env = SimEnv(driver="/home/…/raifusim/driver.hml", maps=[("Dustbowl", 4)], seats=[0, 1, 2, 3])
    obs = env.reset()
    while not obs.done:
        obs = env.step(action_index)

IT IS A DROP-IN FOR `WarriorEnv` ON PURPOSE. Same `Step`, same `reset()`, same `step(index)`, and
`default_reward` is IMPORTED from env.py rather than reimplemented -- a reward that differs by so
much as a constant makes the two sets of numbers incomparable, which would defeat the point of
having a fast environment to compare against a slow one.

CONTROL IS NOT INVERTED HERE, and that is the entire reason this file is short. The Warrior
protocol makes the GAME the HTTP client: the match runs on its own clock, calls `/v1/act` when it
wants a decision, and `WarriorEnv` has to run a server in a background thread and pass the request
across two queues to turn that back into `step()`. The simulator is a child process on a pipe, so
the learner already drives it -- write an index, read the next state. No threads, no server, no
port to collide on.

WHY NOT HTTP ANYWAY. At ~1.4 decisions/sec the transport is free and HTTP is worth it for the
isolation. At the ~1,500/sec this thing runs at, a connect/headers/teardown per decision is most
of the budget. A newline is not.

WHAT THIS ENVIRONMENT IS NOT. Cards are absent -- all 49 of them, about 9% of real decisions --
and the other seats are played by the simulator's own policy rather than the game's `classic` AI.
So a policy trained here has never seen a card played and has never met the opponent it will be
evaluated against. It is a pre-training environment; the real game is still the judge.
"""

import json
import os
import subprocess
import time

from .env import Step, default_reward


class SimEnvError(RuntimeError):
    pass


class SimEnv:
    def __init__(self, driver, hemlock="/usr/local/bin/hemlock", cwd=None,
                 maps=None, seats=None, seed=20250813, policy="random", length=0,
                 maxturns=4000, reward=None, rotate_offset=0, timeout=120.0):
        """`maps` is a list of (name, seat_count) PAIRS, matching WarriorEnv's `combos` argument.

        The pairing is not decoration: a seat only exists if the board seats it, so "Twin Rivers"
        with seat 2 is not a configuration but a match nobody plays. The driver drops impossible
        pairs rather than starting one.
        """
        self.driver = driver
        self.hemlock = hemlock
        # CWD IS THE SIM'S DIRECTORY. `driver.hml` resolves `data/map/*.rwm` and `src/*.hml`
        # relative to the working directory, so launching from the learner's tree gives a process
        # that starts, fails to find a map, and exits before it ever offers a decision.
        self.cwd = cwd or os.path.dirname(os.path.abspath(driver))
        self.maps = list(maps) if maps else [("Dustbowl", 4)]
        self.seats = list(seats) if seats else [0]
        self.seed = int(seed) + int(rotate_offset) * 1000003
        self.policy = policy
        self.length = length
        self.maxturns = maxturns
        self.timeout = timeout
        self.reward_fn = reward or default_reward

        self._proc = None
        self._prev = None
        self._acts = []
        # After an "end" the driver starts the next match on its own and its first "act" is
        # already on the wire. `reset()` consumes that instead of asking for a match it has
        # already been given -- which would throw one away per episode.
        self._pending_act = None
        self._ended = False
        self.episodes = 0
        self.steps = 0

    # -- process ------------------------------------------------------------

    def _argv(self):
        return [
            self.hemlock, self.driver,
            "maps=" + ",".join("%s:%d" % (m, n) for m, n in self.maps),
            "agent=" + ",".join(str(s) for s in self.seats),
            "seed=%d" % self.seed,
            "policy=%s" % self.policy,
            "length=%d" % self.length,
            "maxturns=%d" % self.maxturns,
        ]

    def _launch(self):
        self._kill()
        self._proc = subprocess.Popen(
            self._argv(), cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # stderr is the driver's log channel, not an error channel -- it announces the board
            # rotation and complains about an out-of-range index. Inherited so it lands in the
            # run log beside PPO's own output rather than filling a pipe nobody drains.
            stderr=None, bufsize=1024 * 1024, text=True, encoding="utf-8")
        self._pending_act = None
        self._ended = False

    def _kill(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.kill()
                self._proc.wait(timeout=10)
        except Exception:                                       # noqa: BLE001
            pass
        self._proc = None

    def close(self):
        self._kill()

    # -- wire ---------------------------------------------------------------

    def _send(self, obj):
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            raise SimEnvError("driver closed its input: %s" % e)

    def _recv(self):
        """One message, or None if the driver exited.

        Reads a WHOLE LINE. A payload with 600 offered actions is tens of kilobytes and arrives in
        several pipe reads; `readline` on a text-mode pipe already handles that, which is the one
        good reason to let Python own the buffering here and not do it by hand.
        """
        line = self._proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise SimEnvError("driver wrote a line that is not JSON (%s): %.200s" % (e, line))

    # -- gym surface --------------------------------------------------------

    def reset(self):
        if self._proc is None or self._proc.poll() is not None:
            self._launch()
        elif not self._ended:
            # Abandoning a match part-way is the only case that needs telling. After an "end" the
            # driver has already started the next one.
            self._send({"t": "reset"})
            self._pending_act = None

        deadline = time.time() + self.timeout
        while True:
            msg = self._pending_act
            self._pending_act = None
            if msg is None:
                msg = self._recv()
            if msg is None:
                if time.time() > deadline:
                    raise SimEnvError("driver produced no decision within %.0fs" % self.timeout)
                self._launch()
                continue
            if msg.get("t") == "act":
                self._ended = False
                self.episodes += 1
                self._prev = msg.get("payload") or {}
                self._acts = msg.get("actions") or []
                return Step(state=self._prev, actions=self._acts)
            # An "end" here belongs to the match being abandoned; keep reading for the "act".

    def step(self, index):
        actions = self._acts
        if not 0 <= index < len(actions):
            raise IndexError("action %d out of range for %d offered" % (index, len(actions)))
        self._send({"i": int(index)})
        self.steps += 1

        msg = self._recv()
        if msg is None:
            # The driver exited mid-match. Terminal either way, and the reward for the final
            # transition is unknown, so it is zero rather than invented -- the same choice
            # WarriorEnv makes when a game process disappears.
            self._ended = True
            return Step(state=self._prev, done=True, info={"reason": "driver exited"})

        if msg.get("t") == "end":
            self._ended = True
            r = self.reward_fn(self._prev, None, msg)
            # The next match's first "act" follows immediately; hold it for reset().
            nxt = self._recv()
            self._pending_act = nxt if (nxt or {}).get("t") == "act" else None
            return Step(state=self._prev, reward=r, done=True,
                        info={"reason": "match end", "payload": msg})

        state = msg.get("payload") or {}
        r = self.reward_fn(self._prev, state, None)
        self._prev = state
        self._acts = msg.get("actions") or []
        return Step(state=state, actions=self._acts, reward=r)
