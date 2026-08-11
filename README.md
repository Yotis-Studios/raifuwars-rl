# raifuwars-rl

Reinforcement learning against **Raifu Wars**, driven through the
[Warrior protocol](https://github.com/yotisstudios/Warrior).

Separate from Warrior on purpose. Warrior is a game-agnostic protocol -- it knows about turns and
legal sets and nothing about raifus -- and feature encoders, reward shaping and a self-play league
are irreducibly game-specific. Putting them in Warrior would destroy the property the spec exists
to have.

| repo | owns |
|---|---|
| RaifuWars | the rules, exactly once |
| Warrior | protocol, reference sidecar, trace format, tournament/eval harness |
| **raifuwars-rl** | env, features, policy, training, reward |

## Why RL at all

Supervised fine-tuning on the built-in AI caps at imitating it, and it wins ~32% of its own seats.
A 4B trained on ~10k of its decisions reached 86% agreement and won 2 matches in 40. To go past the
teacher, something has to learn from outcomes rather than from the teacher.

RL's most valuable role here may not be playing the game but **producing a better teacher to
distil into an LLM seat** -- lifting the imitation ceiling rather than replacing the seat.

## Status

- `raifuwars_rl/env.py` -- gym-shaped env over a real match. Working.
- `tools/smoke_env.py` -- random legal policy plays a full match. Working.
- `tools/outcomes.py` -- recovers per-seat final tier/stars/kills/won from traces alone.

Not built yet: feature encoder, action-scoring policy, training loop.

## The two things that shape the design

**Control is inverted.** The game is the HTTP client; the sidecar is the server. The env runs a
sidecar in a thread and blocks the handler until the learner answers -- safe because Raifu Wars has
no per-action clock, which is the same property that lets an LLM take 30 seconds.

**The action space is the offered list**, 2 to ~670 entries, varying per decision. `step` takes an
INDEX into it, so an illegal action is unrepresentable rather than penalised. That forces the policy
to *score* actions rather than classify over a fixed head -- which also makes it generalise across
board sizes.

## Measured throughput

| | steps/sec |
|---|---|
| 1 instance, drawing | 6.8 (all four seats' decisions) |
| 1 instance, headless | 8.1 |
| 6 instances, headless | 34.6 |
| **1 env, agent steps only** | **~1.6** |

The last row is the one that matters and it is the smallest: an env controls ONE seat, so a
~40-step episode takes ~25s. Six instances gives roughly 10 agent steps/sec, so 1M agent steps is
about 30 hours. RL here is a multi-day run per experiment -- reward and architecture want to be
right before starting rather than tuned across it.

## Evaluation is free

A trained policy re-exports as a Warrior sidecar, so `warrior-tournament.sh`, `offer-rate.py` and
`hint-divergence.py` all work unchanged, against the same classic-AI anchor the LLM seats were
measured on. Success is beating **32% seat win rate**, not agreement with anything.
