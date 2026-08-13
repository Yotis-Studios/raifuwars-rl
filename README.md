# raifusim — the Raifu Wars core loop in Hemlock

A headless port of Raifu Wars' turn loop, for pre-training an RL policy before fine-tuning it in
the real game. No rendering, no frame loop, no animation.

**~1,540 agent decisions/sec per process, ~11,000/sec across 8 workers** on sabre, against
**~1.4/sec/instance** for the GameMaker client. About 1,100× per process.

It is a **second implementation of a rulebook**, which is only safe because something compares the
two. `conform.hml` replays 24,200 real CPU decisions from `data/cpu-traces-v3-ffa.jsonl` and
reports where the two disagree. The numbers are below, including the places where the sim is
**known to be wrong**. An unfaithful sim that is known to be unfaithful in specific ways is
useful; one believed faithful and isn't will silently poison the run.

## Layout

```
src/rwmap.hml    .rwm decoding: tile encoding, objects, building footprints, points and bases
src/sim.hml      the rules: movement, combat, tiering, star income, KO/respawn, win condition
src/payload.hml  the decision payload, BOTH ways: parser and serializer in one file
conform.hml      the conformance harness -- the deliverable that makes the port trustworthy
roundtrip.hml    the SERIALIZER's harness: parse a real row, write it back, diff every field
driver.hml       the sim as a subprocess speaking line-delimited JSON -- what an RL loop drives
bench.hml        throughput, driven by a uniform-random policy
data/map/*.rwm   the 30 shipped maps, copied from the game repo
data/traces_sample.jsonl   85 whole matches (24,200 decisions) sampled from the corpus
cmp_board.hml    prints the decoded board beside the one the game drew, for eyeballing
debug_diff.hml   dumps individual legal-set disagreements with the state that produced them
debug_tally.hml  dumps individual star-tally disagreements, per point and per seat
debug_hitchance.hml  tabulates hit-probability error by what the target is standing on
```

Run:

```
hemlock bench.hml                     # MAP=Dustbowl MATCHES=200 PLAYERS=4
hemlock conform.hml                   # ONLYMAP=… LIMIT=… DUMP_FP=1 ASCII_OBJECTS=1
hemlock roundtrip.hml                 # LIMIT=… DUMP=8 EMIT=pairs.jsonl
hemlock driver.hml maps=Dustbowl:4 agent=0,1,2,3 seed=1 policy=greedy
```

## What is implemented

Everything the core loop needs, read out of the GML rather than guessed at. Each rule names the
function it came from, in the source.

- **Turn rotation** — `turnNumber % playerCount`, a fixed rotation. A knocked-out seat does not get
  a turn: it rolls to respawn and the turn passes straight on, tallying as it goes.
- **Movement** — Dijkstra, **eight-connected**, over the square `[cx±budget, cy±budget]` clipped to
  the map. A diagonal costs `sqrt(2)`, not 1. The bounding box is a hard cap, which matters because
  path tiles cost 0.5 and would otherwise buy more steps than the budget. Budget is
  `moveRoll + moveMod`, and `moveRoll` is the **constant 4**, not a die — only a rush rolls.
- **Terrain** — the full `.rwm` decode: one integer carrying material and variant with divisor 18,
  the 1-based point value (`18v-1`), water as the only impassable material, marsh and path move
  costs, obstacle passability and penalties, and **multi-tile building footprints** (a building
  stamps the same instance over a `width × height` block extending up and left, and fills water
  underneath).
- **Attack** — range is a **chebyshev square** (`xDist <= range && yDist <= range`) while the
  distance fed to the probability curve is **euclidean**. Two metrics in one function. Cover comes
  from a Bresenham line of fire, with the game's two quirks reproduced deliberately: the `-0.05`
  inside the obstacle loop applies per obstacle unconditionally despite its comment, and an
  obstacle on the target's own tile is charged twice.
- **Damage** — the attack roll plus `attackMod` and `damageMod`, minus 1 if fortified, floored at 1
  **before** Lucznik's doubling and Ishapore's +5, which is why no amount of armour absorbs a hit.
- **Reload / Fortify** — one action with two meanings, told apart by ammo, not by a gate.
- **Rush** — spends the combat action on a second movement. `rushedThisTurn` is set when that
  second move *lands*, not when Rush is chosen, which is why Rush and Attack are both still offered
  in between.
- **Tiering** — legal inside the **chebyshev square of half-width `point.value`** of any base
  belonging to your own team; a base is always value 1, so a 3×3. Puska tiers at any base. Tier 4
  wins and is the only win condition. Tiering raises the whole team.
- **Stars** — `tallyStarsFromPoints`, once per turn, whole-board, including the capture rules it
  runs first, the base heal, the `turnNumber >= numPlayers` embargo, and the **double multiplier**
  (see "Reproduced quirks").
- **Knockouts** — a kill transfers `round(victim.stars / 4)` to the attacker and costs the victim
  `round(victim.stars / 2)`; the difference leaves the game. Kills credit the whole team. Respawn
  rolls against a threshold that drops by one per missed attempt, so it is certain eventually, and
  returns the raifu to its base with spawn protection.
- **Environmental death** at end of turn — lava kills, lava rock deals 1, water drowns anyone who
  is not Krag.
- **Characters** — all thirteen, including the three abilities that are *not* in
  `setCharacterPlayerValues`: Lucznik's double damage at range ≤ 4, Ishapore's +5 at range < 2, and
  Krag's water immunity.

## What is NOT implemented

- **Cards. All 49 of them.** About 9% of real decisions are card plays. Every place a card would
  enter is marked `CARD SEAM` in `src/sim.hml` — the movement budget (`muck`, `move_roll_2`,
  `net`), the attack gates (`festival`, `hastyshot`), the probability (`accuracy_100`, `bazooka`,
  `pocketsand`, `range_unlimited`), the tier gates (`lemongang`, `blockade`, `bribe`), the tally
  (`blessings`, `dishonor`), and the action list itself. The card draw at turn start is also absent.
- **Ice breaking** under whoever stands on it. The tile mutation is the only board change the core
  loop makes and it needs a per-match tile overlay to hold it. Ice is walkable and permanent here.
- **Pontoon bridges** are decoded from the map (they make a water tile walkable and shootable from,
  with a 10% accuracy penalty) but cannot be placed or destroyed, since both are card effects.
- **Coins**, achievements, chat, the online protocol, teams beyond equal blocks.
- **Procedural vegetation is approximated, not reproduced.** See below.

## Conformance

24,200 decisions, 85 whole matches, 7 maps. Whole matches rather than sampled lines, because a
decision sequence has to stay intact: the payload cannot distinguish "has moved, may not again"
from "has rushed and may move again", and only the previous decision in the same turn can say so.

Excluded, and counted separately: 953 rows sitting in a card-driven modal state, and 2,864 rows in
a turn where a card had already resolved. Card actions are dropped from both sides of the
comparison — counting a deliberate omission as a rules failure would be measuring the wrong thing.

### A. Legal-set agreement

| family | offered by both | sim only (FP) | game only (FN) | precision | recall |
|---|---|---|---|---|---|
| move | 276,230 | 19,527 | 0 | 93.40% | **100.00%** |
| attack | 16,697 | 441 | 120 | 97.43% | 99.29% |
| rush | 8,356 | 0 | 0 | **100.00%** | **100.00%** |
| reload | 15,301 | 0 | 0 | **100.00%** | **100.00%** |
| tier_up | 890 | 0 | 0 | **100.00%** | **100.00%** |
| end_turn | 20,383 | 0 | 0 | **100.00%** | **100.00%** |
| **total** | **337,857** | 19,968 | 120 | **94.42%** | **99.96%** |

Exact set match (every action, both directions): **90.69%** of rows.

Per map:

| map | exact set match | rows |
|---|---|---|
| Dustbowl | **100.00%** | 3,375 |
| Twin Rivers | 99.91% | 1,157 |
| Cornfield | 99.53% | 5,286 |
| Crossroads | 99.52% | 2,492 |
| Glacier | 98.63% | 2,635 |
| Trench Warfare | 97.67% | 1,204 |
| **Arboretum** | **57.58%** | 4,234 |

**Every one of the 19,527 move false positives is on Arboretum**, and they are all the same thing:
Arboretum is the only board in the corpus that uses tile material 0, *climate-random*, on 270 of
its tiles. `initTileAt` scatters trees, bushes and rocks over those with `perlinNoise`, which
reseeds `global.seed` on every sample — so the scatter is a property of the **match**, not of the
map file, and is not recoverable from the `.rwm`. The sim knows the terrain and not the scenery,
and offers moves onto tiles a tree is standing on.

Running the harness with `ASCII_OBJECTS=1` reads those obstacles back out of the ASCII board the
game itself drew in each payload, which measures the rest of the rules on that map instead:

| | exact match | move precision | move recall | attack precision |
|---|---|---|---|---|
| map file only | 90.69% | 93.40% | 100.00% | 97.43% |
| + obstacles from the payload's own board | **94.71%** | **99.99%** | 99.06% | **99.71%** |

Arboretum goes 57.58% → 76.92%. The residual there is that the game's ASCII draws points and raifu
*over* objects, so anything standing on one is invisible, and it cannot say whether an `o` is a
tree (blocking) or a bush (walkable, cost 2) — so the hydrated run over-blocks and produces false
*negatives* instead. Neither direction is a rules disagreement.

**The 441 attack false positives** (2.6% of attack offers) are dominated by one reconstruction gap.
`spawnProtection` is not in the payload at all, and it decides whether a raifu may be shot
(`playerIsAttackable`). The harness reconstructs it from the trace — set at match start and on
every respawn, cleared the moment that seat moves or fires — which took the count from 15% down to
4%, but one case still escapes: **a raifu that dies standing on its own base respawns exactly where
it fell.** It never appears as "knocked out" in any row (killPlayer ends the killer's turn on the
spot, and a knocked-out seat records no decisions of its own) and it does not move, so neither the
status flip nor the warp-home test can see it. A health-jump test for it was tried and traded 10
false positives for 16 false negatives, so it was left out. **This is a limitation of the harness,
not of the sim** — the sim models spawn protection properly; it is the replay that cannot always
tell.

### B. Transition agreement

Given a state and the chosen action, does the sim produce the next row's state?

| family | pairs | matched | agreement |
|---|---|---|---|
| move | 7,708 | 7,708 | **100.00%** |
| attack | 1,427 | 1,427 | **100.00%** |
| rush | 2,749 | 2,749 | **100.00%** |
| reload | 642 | 642 | **100.00%** |

Zero mismatches on tile, ammo, stars, tier, health or fortification across all 12,526 pairs.

Attack is compared on its **deterministic** fields only — the attack roll and the luck sample are
not in the corpus, so a shot's outcome cannot be predicted. The damage *formula* is checked
separately: for every shot that visibly reduced the target's health, is the observed drop one of
the six values the formula can produce? **1,064 / 1,065 = 99.91%.**

### C. End-of-turn tally

The whole-board half of the rules — star income and base healing for every seat, over a turn
ending. 1,066 turn ends were excluded because a card was live within the previous round
(`blessings` doubles a seat's income, `dishonor` zeroes it).

| | |
|---|---|
| turn ends compared | 2,625 |
| stars, all seats | **93.83%** |
| health, all seats | **99.66%** |

Per map: **100.00% on all six four-seat boards** (Dustbowl, Crossroads, Arboretum, Cornfield,
Trench Warfare, Glacier — 2,463 turn ends). Twin Rivers: **0.00%** across all 162 of its turn ends,
and in **every single one** the game paid **exactly twice** what the sim did, to every seat.

Twin Rivers is the only **two-seat** map in the corpus. The doubling is uniform, exact, present
from the first turn stars are awarded, and independent of tier, blessings and point count — so it
is not a rule the sim has wrong. The likeliest reading is the final loop of `tallyStarsFromPoints`:

```gml
var teamPlrs = global.numPlayers / global.numTeams;
for (var j = 0; j < global.numPlayers; j += teamPlrs) {
    changePlayerStars(global.players[| j], teamStars[player.team-1] * multiplier);
}
```

`teamPlrs` is a **float** divide. With `numPlayers = 2` and a `numTeams` still at its
`initAtGameStart` default of 4, it is `0.5`, the loop runs four times at `j = 0, 0.5, 1, 1.5`, GML
truncates the list index, and every seat is credited twice. That is a match-configuration artifact
of the corpus rather than a rule, and it is **not** reproduced here — the sim implements the source
as written. If two-seat matches matter for training, this is the thing to check first.

### Reproduced quirks

Deliberately, because they are in the shipped game and the corpus shows them:

- **The star multiplier is applied twice, and the second one is the last seat's.** `multiplier` is
  left over from the loop above it — read from one player and applied to everybody. A tier-2 raifu
  in the last seat doubles every other seat's income. Confirmed against the corpus.
- **A knocked-out raifu inside its own base still heals.** The base-heal test reads the *unfiltered*
  occupancy array while the capture logic above it filters the knocked out, so a downed raifu heals
  to 1 and stays down.
- **`-0.05` per obstacle, unconditionally**, despite the comment above it saying "if target is
  fortified".
- **An obstacle on the target's own tile is charged twice** — once through the line of fire at 1.5×
  for being adjacent to the target, and again at 2× by the tile check after it.
- **Damage floors at 1 before the character multipliers**, so Lucznik doubles a number that has
  already bottomed out.
- **A start base gives its occupant sandbag cover**, −0.35 to anyone shooting at them. The `.rwm`
  encodes a point tile as a point and loses whatever material was under it, so this is invisible
  to the map decode and had to be recovered from the corpus — see below.

### The base-cover rule, and how it hid

`conform.hml` asks whether a shot was **offered**; it never asked **at what odds**. `roundtrip.hml`
does, because `hit_chance` is a feature the policy reads, and the answer was that every attack at a
target standing on a base disagreed:

| target standing on | attacks | sim = game |
|---|---|---|
| open ground | 9,539 | 95.4% |
| a capture point | 6,297 | 91.7% |
| **a start base** | **923** | **0.0%** |

880 of those 923 were **exactly 0.35** too high — the sandbags penalty in `attackProbability`, to
the digit. So a base is sandbagged and the map file cannot say so. Adding it took the whole-corpus
`hit_chance` agreement from 88.78% to **95.75%**, attack precision from 95.84% to **97.43%**, and
exact-set match from 89.38% to **90.69%**. Capture points deliberately do *not* get it: 91.7% of
attacks at a target on one already agreed, and extending the rule there breaks them.

This is the argument for measuring a number rather than a boolean. 0.35 is nowhere near enough to
push a shot across the 1% legality gate, so the legal-set harness saw almost none of it.

The residual is the same scenery problem as everywhere else — the sim knows the terrain and not
what is standing on it:

| map | attacks | hit_chance exact |
|---|---|---|
| Dustbowl | 5,281 | 97.86% |
| Glacier | 2,736 | 96.89% |
| Twin Rivers | 714 | 96.78% |
| Crossroads | 3,854 | 96.08% |
| Cornfield | 411 | 93.92% |
| Trench Warfare | 1,244 | 89.55% |
| **Arboretum** | 2,519 | **81.06%** |

The largest remaining error bucket after Arboretum is ±0.01, which is a rounding boundary and not
a rule.

### Deliberate divergence

- **Dice are seeded once per match** (xorshift), not `randomize()`-per-roll. The real game cannot
  replay a match from a seed; this one can, which is what an RL run wants.
- **Procedural vegetation is approximated.** `scatter_vegetation` reproduces the *shape* of
  `initTileAt`'s rule (tree at `noise >= sparsity`, bush in the four below it, rocks at 20–24) with
  uniform noise instead of perlin, seeded per match. A board built this way is a plausible
  Arboretum rather than any Arboretum that was ever played. It is the only approximation in the
  port, and it is confined to maps that use tile material 0.

## Throughput

Uniform-random policy over the legal set, four seats, free-for-all, Dustbowl, matches run to a
real tier-4 win (30/30 finished, ~356 turns each).

| | |
|---|---|
| single process | **1,544 decisions/sec**, 625 turns/sec |
| 8 parallel processes | **~11,060 decisions/sec** aggregate (1,342–1,409 each) |
| the GameMaker client | ~1.4 decisions/sec/instance |

Scaling is close to linear at 8 workers on a 24-core box, so ~30k/sec is available. The cost of a
decision is the bounded Dijkstra; the integer action interface (`legal_action_ids` / `apply_id`)
was measured against the string one and made no difference, so the flood fill dominates.

Through the JSON driver, with a torch policy on the other end of the pipe, what a learner actually
gets is **AGENT** decisions — one seat in four — with a payload serialized and parsed for each:

| | |
|---|---|
| one driver, uniform-random agent | ~430 agent steps/sec |
| 8 drivers under PPO, scoring every offered action | **~1,170 agent steps/sec** |
| PPO on real GameMaker matches | 10.3 agent steps/sec |

## The RL interface

Use the integer API. Action ids are dense and stable for a given map size:

```
0 .. w*h-1     move to that tile index
w*h + seat     attack that seat
w*h + 8        rush
w*h + 9        reload / fortify
w*h + 10       tier up
w*h + 11       end turn
w*h + 12 ..    CARD SEAM -- play/discard per hand slot would go here
```

```hemlock
let map = rwmap.load_map("data/map/Dustbowl.rwm");
let st  = sim.new_match(map, 4, 4, 0, seed, null);   // players, teams, game length, seed, roster
while (!st.gameover) {
    let p = st.players[st.cur];
    sim.legal_action_ids(st, p, acts, sr, sl);        // acts is the legal mask, as ids
    sim.apply_id(st, p, chosen, sl);                  // returns true if the turn ended
}
```

`legal_actions` / `apply` are the string-id equivalents and exist for the conformance harness,
which has to compare against the game's own `move_4_2` / `atk_s1` ids.

## The bridge to the Python trainer

`driver.hml` runs the sim as a **subprocess speaking line-delimited JSON**, which is what
`raifuwars_rl/simenv.py` drives. Not HTTP: the Warrior protocol is HTTP because the *game* is the
client and may be on another machine, but here the sim is a child process, and at ~1,500
decisions/sec a connect-headers-teardown per decision costs more than the decision does.

```
sim   -> {"t":"act","payload":{…state…},"actions":[…]}     and blocks
agent -> {"i": <index into actions>}
sim   -> {"t":"end","won":true,"seat":0,"tier":4,…}        then the next match's "act"
agent -> {"t":"reset"}                                      abandons a match part-way
```

```
hemlock driver.hml maps=Dustbowl:4,Glacier:4 agent=0,1,2,3 seed=1 policy=greedy
    maps=Name:seats,…   (map, seat-count) pairs, rotated per match
    agent=0,1,2,3       which seat the learner plays, rotated per match; impossible pairs dropped
    policy=random|greedy   what the OTHER seats play
    roster=random|seat  characters per match; `seat` restores `sim.new_match`'s `seat % 13`
    seed=…  length=…  teams=ffa|N  maxturns=…
```

The payload shape is `src/payload.hml`, and it is the same file that **parses** a payload for
`conform.hml`. That is what makes `roundtrip.hml` mean anything: it feeds a real corpus row through
the validated reader, writes it straight back out, and diffs every field.

| | |
|---|---|
| state fields exact on all 20,383 rows | **50 / 52** — the two are `hand` and `cards_in_hand` |
| action `type` agreement | **100.00%** (337,857 rows) |
| `hit_chance` exact to 2dp | **95.75%**, mean abs error 0.014 |
| feature vectors identical to the game's own payload | **31/33 state, 26/27 action** |

The last row is the one that matters: `tools/compare_features.py` encodes the *same real state*
from both payloads and subtracts. Every column is bit-identical except `hand_frac` and
`hit_chance` — cards, and the 0.21% of shots above. A field spelled wrong would show up there as a
whole column of differences rather than as nothing at all, which is what it would be in training.

**Opponent seats.** `random` is `sim.random_action`, uniform over the legal set, and it is the only
policy the sim shipped with. `greedy` was added for the bridge — tier if you can, shoot the best
odds you have, walk toward whatever is worth standing on — because a uniform-random table never
walks to its own base and so never tiers, and a policy trained against one has never seen a race. A
uniform-random agent wins **24.3%** against `random` (chance is 25% on a four-seat board) and
**1.1%** against `greedy`. Neither is the game's `classic` AI, which is GML and is not ported.

### What PPO does with it

806 updates, **1.65M agent decisions in 35.7 minutes** — 8 drivers, six boards, all four seats,
warm-started from `runs/bc-long.pt`, opponents on `greedy`. The same run on real GameMaker matches
is 44 hours. Nothing was tuned; every hyperparameter is what `tools/ppo.py` already had.

Mean return over the trailing 100 episodes: **8.4 → ~15.5 by update 120**, then flat between 15 and
17.7 for the remaining 700. Zero non-finite features in 806 updates.

Measured afterwards on a **held-out seed** (424242, six boards, ~250 finished matches each):

| policy | mean return | win rate | mean final tier |
|---|---|---|---|
| uniform random | 4.17 | 0.0% | 1.61 |
| behaviour cloning (`bc-long.pt`) | 10.48 | 22.1% | 3.08 |
| **PPO over the sim (`ppo-sim/best.pt`)** | **16.41** | **74.6%** | **3.73** |

Chance on a four-seat free-for-all is 25%. The real-game teacher averages 9.59 return, so the
BC checkpoint reproducing 10.48 here is the reward function agreeing with itself across two
environments — which is the point of importing `default_reward` rather than rewriting it.

**Do not read 74.6% as strength at Raifu Wars.** It is 74.6% against `greedy`, which is forty
lines of scoring in `driver.hml`, on a board with no cards in it. The plateau after update 120 is
most likely the policy having exhausted what that opponent can teach it.

**A turn cannot run forever, and the driver enforces that.** Most actions consume something —
the move, the shot, the reload — but **Rush does not**. `playerRush` sets `isMove` and rolls;
`rushedThisTurn` is only set when the second move *lands*. A raifu that has moved and is now boxed
in gets an empty move list, so Rush stays legal, changes nothing, and stays legal. The greedy
opponent found this on its first long run: one driver span at 100% CPU forever while the other
seven envs blocked behind it. `greedy` now refuses Rush while `is_move` is already set, and the
driver ends any turn that reaches 512 decisions — 512 being roughly a hundred times a real turn,
so it never fires on anything legitimate. A learner playing argmax could walk into exactly the same
trap, and there it would present as a training run that quietly stopped producing steps.

## Reading this against the game

Where this port and the GML disagree, **the GML is right**. `docs/RULES.md` in the game repo is the
prose version and names the file behind every rule; `scripts/warriorActions` is the authority on
what is legal, and `scripts/warriorApplyAction` on what each action does. The corpus is
`data/cpu-traces-v3-ffa.jsonl`, and `conform.hml` is how the two are kept honest — rerun it after
any rule change.
