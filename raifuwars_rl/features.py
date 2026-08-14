"""Hand-crafted features for a Raifu Wars decision.

Two encoders, because the policy SCORES ACTIONS rather than classifying over them:

    encode_state(state)            -> (D_STATE,)      the board, once
    encode_actions(state, actions) -> (N, D_ACTION)   one row per offered action

A score is then f(state, action_i) softmaxed over the N offered. That handles a legal set that
varies from 2 to ~670 between decisions, gives exact masking for free, and generalises across board
sizes -- none of which a fixed output head can do.

WHY HAND-CRAFTED. The structured state is already clean and small, an RL run here costs a day, and
features you can read are features you can debug when a run goes wrong. Learning an encoder from
the ASCII grid would spend most of that day rediscovering "a point is worth standing on".

EVERYTHING IS COMPUTED FROM STRUCTURED FIELDS, NEVER FROM THE NOTE TEXT. The game writes a helpful
English note on each action ("CAPTURES the unclaimed point here...") and parsing it would be a
second, worse copy of rules that already exist as data. Point ownership, distances and costs are
all in `state.board.points` and the action's own fields.

SCALED TO ROUGHLY [-1, 1]. Stars reach thousands and tiles reach 32; feeding either raw makes the
first layer spend its capacity on units. Distances are divided by board size so a 17x21 board and a
27x27 board produce comparable numbers -- which is the whole point of training across nine maps.
"""

import math
import os

import numpy as np

ACTION_TYPES = ["move", "attack", "rush", "reload", "play_card", "discard_card",
                "tier_up", "tier_choice", "select_tile", "cancel", "end_turn", "chat"]
_TYPE_INDEX = {t: i for i, t in enumerate(ACTION_TYPES)}

# TERRAIN, OPT-IN VIA RW_FEAT_COVER=1.
#
# WHY IT EXISTS. Nothing in the base feature set describes terrain. The policy can read hit_chance
# for a shot IT takes, but has no way to represent "this destination leaves me exposed" or "that
# one puts me behind a tree" -- only nearer/further. That is survivable on a board with no cover
# and fatal on one made of it:
#
#     Crossroads   0 cover tiles (0.0%)     policy wins 68%,  4.9 kills/match
#     Arboretum  172 cover tiles (37.2%)    policy wins 13%,  1.4 kills/match
#     Islands    321 cover tiles (33.4%)    untested at time of writing
#
# Crossroads is the only board with NO cover at all, which is also why every policy measured --
# two LLMs and this net -- wins there by parking and sniping, and collapses where cover exists.
# The map is degenerate for evaluation, and the feature set was fitted to it by accident.
#
# OPT-IN because D_STATE and D_ACTION are baked into every existing checkpoint: flipping these on
# unconditionally would silently break the served advisor and every saved .pt. Off by default, so
# old runs and tools/serve.py behave exactly as before; new experiments set the flag and train
# from scratch.
_COVER = os.environ.get("RW_FEAT_COVER", "").strip().lower() not in ("", "0", "false", "no")

STATE_FIELDS = [
    "hp_frac", "ammo_frac", "tier_frac", "tier_progress", "stars_log", "kills", "deaths",
    "range_frac", "fortified", "move_roll_frac", "moved", "attacked", "rushed", "played_card",
    "tiered", "in_territory", "hand_frac", "pos_x", "pos_y", "turn_log",
    "n_enemies_alive", "enemy_tier_max", "enemy_tier_lead", "nearest_enemy_dist",
    "nearest_enemy_hp", "enemy_stars_lead",
    "points_mine", "points_enemy", "points_free", "dist_nearest_free_point",
    "dist_nearest_enemy_point", "dist_own_base", "can_tier_now",
] + (["cover_density", "cover_here"] if _COVER else [])
D_STATE = len(STATE_FIELDS)

ACTION_FIELDS = (["type_" + t for t in ACTION_TYPES] + [
    "dest_dist", "dest_captures", "dest_is_base", "dest_point_value",
    "dest_dist_to_free_point", "dest_dist_to_nearest_enemy", "dest_toward_base",
    "hit_chance", "target_hp_frac", "target_tier_frac", "target_dist", "target_would_kill",
    "cost_frac", "tier_required", "affordable",
] + (["dest_cover"] if _COVER else []))
D_ACTION = len(ACTION_FIELDS)


def _f(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(x) or math.isinf(x)) else x


def _point_radius(pt):
    """How far a point's territory reaches, in chebyshev tiles.

    The game sends this as `radius`. Older corpora predate the field, but `radius` and
    `stars_per_turn` are the same number on the game's side -- `point.value` served as both, which
    is precisely why the extent was invisible for so long -- so the fallback is exact rather than
    a guess.
    """
    r = pt.get("radius")
    if r is None:
        r = pt.get("stars_per_turn")
    r = _f(r, 1.0)
    return r if r > 0 else 1.0


def _cheb(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def _grid(state):
    """The ascii board as grid[y][x] glyphs, or [] when unavailable.

    The payload's ascii is the ONLY place terrain appears -- board.points covers capture points and
    the player list covers raifus, but cover objects and water exist nowhere else in structured
    form. The first line is the column ruler and each row is prefixed by a two-character row label,
    so both are stripped; the remaining glyphs are space-separated, which is what split() wants.
    """
    asc = ((state.get("board") or {}).get("ascii")) or ""
    rows = []
    for ln in asc.split("\n")[1:]:
        if not ln.strip():
            continue
        rows.append(ln[2:].split())
    return rows


def _cover_at(grid, x, y):
    """Fraction of the 8 neighbours of (x,y) that are cover objects.

    Neighbours rather than the tile itself: standing ON cover is not what the game rewards -- cover
    reduces incoming hit chance when it sits BETWEEN shooter and target, so being surrounded by it
    is the readable proxy. Off-board neighbours count as open, which makes a board edge look
    exposed; that is the honest reading, since an edge tile gives no protection either.
    """
    if not grid:
        return 0.0
    n = 0
    xi, yi = int(x), int(y)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            gy, gx = yi + dy, xi + dx
            if 0 <= gy < len(grid) and 0 <= gx < len(grid[gy]) and grid[gy][gx] == "o":
                n += 1
    return n / 8.0


# NO EXPOSURE FEATURE, DELIBERATELY. The obvious companion to cover is "how many enemies can shoot
# this tile", and it was written, measured and removed: our range is 18 on a 21-wide board, so every
# enemy reaches every tile and the feature was CONSTANT at 0.60 across all 31 offered moves. A
# feature with no variance is a bias term that costs a weight. Distance to the nearest enemy at the
# destination is already carried by dest_dist_to_nearest_enemy, and a real exposure number would
# need the game's cover-and-distance falloff -- which is the rule this module exists not to
# reimplement.


def _board(state):
    b = state.get("board") or {}
    w = max(1.0, _f(b.get("width"), 21))
    h = max(1.0, _f(b.get("height"), 21))
    return b, w, h, max(w, h)


def encode_state(state):
    me = state.get("self") or {}
    b, w, h, span = _board(state)
    mx, my = (me.get("tile") or [0, 0])[:2]
    mx, my = _f(mx), _f(my)
    seat = int(_f(me.get("seat"), -1))
    team = _f(me.get("team"), -1)

    tp = me.get("tier_progress") or {}
    need = _f(tp.get("need"), 0.0)
    have = _f(tp.get("have"), 0.0)
    this = me.get("this_turn") or {}

    enemies = [p for p in (state.get("players") or [])
               if not p.get("is_self") and _f(p.get("team"), -2) != team]
    alive = [p for p in enemies if p.get("status") != "knocked out"]
    e_dists = []
    for p in alive:
        t = p.get("tile") or [0, 0]
        e_dists.append(_cheb(mx, my, _f(t[0]), _f(t[1])))
    nearest = min(e_dists) if e_dists else span
    nearest_hp = 0.0
    if e_dists:
        nearest_hp = _f(alive[int(np.argmin(e_dists))].get("health")) / 6.0

    my_tier = _f(me.get("tier"))
    enemy_tier_max = max((_f(p.get("tier")) for p in enemies), default=0.0)
    my_stars = _f(me.get("stars"))
    enemy_stars_max = max((_f(p.get("stars")) for p in enemies), default=0.0)

    pts = b.get("points") or []
    mine = enemy = free = 0
    d_free, d_enemy_pt, d_base = span, span, span
    for pt in pts:
        t = pt.get("tile") or [0, 0]
        d = _cheb(mx, my, _f(t[0]), _f(t[1]))
        rel = pt.get("relationship")
        if pt.get("is_base"):
            if int(_f(pt.get("held_by_seat"), -1)) == seat:
                # Distance to the base's EDGE, not its centre. The point covers a chebyshev
                # square of half-width `radius`, so a raifu one tile off the centre is already
                # standing where it may tier -- and this feature exists to say "how far from
                # being able to win", which at that moment is zero, not one.
                d_base = min(d_base, max(0.0, d - _point_radius(pt)))
            continue
        if rel == "friendly":
            mine += 1
        elif rel == "enemy":
            enemy += 1
            d_enemy_pt = min(d_enemy_pt, d)
        else:
            free += 1
            d_free = min(d_free, d)

    n_pts = max(1, len(pts))
    v = [
        _f(me.get("health")) / max(1.0, _f(me.get("max_health"), 6)),
        _f(me.get("ammo")) / max(1.0, _f(me.get("max_ammo"), 5)),
        my_tier / 4.0,
        (have / need) if need > 0 else 1.0,
        math.log1p(max(0.0, my_stars)) / 10.0,
        _f(me.get("kills")) / 5.0,
        _f(me.get("deaths")) / 5.0,
        _f(me.get("range")) / span,
        1.0 if me.get("is_fortified") else 0.0,
        _f(me.get("move_roll")) / 6.0,
        1.0 if this.get("moved") else 0.0,
        1.0 if this.get("attacked") else 0.0,
        1.0 if this.get("rushed") else 0.0,
        1.0 if this.get("played_card") else 0.0,
        1.0 if this.get("tiered") else 0.0,
        1.0 if me.get("in_friendly_territory") else 0.0,
        len(me.get("hand") or []) / 3.0,
        mx / w, my / h,
        math.log1p(max(0.0, _f(state.get("turn")))) / 6.0,
        len(alive) / 5.0,
        enemy_tier_max / 4.0,
        (my_tier - enemy_tier_max) / 4.0,
        nearest / span,
        nearest_hp,
        math.tanh((my_stars - enemy_stars_max) / 1000.0),
        mine / n_pts, enemy / n_pts, free / n_pts,
        d_free / span, d_enemy_pt / span, d_base / span,
        # Not "would tiering be legal" -- the action list answers that exactly. This is the
        # threshold being met, which is what makes heading for a base worth anything.
        1.0 if (need > 0 and have >= need) else 0.0,
    ]
    if _COVER:
        grid = _grid(state)
        cells = sum(len(r) for r in grid)
        v += [
            (sum(r.count("o") for r in grid) / cells) if cells else 0.0,
            _cover_at(grid, mx, my),
        ]
    a = np.asarray(v, dtype=np.float32)
    assert a.shape[0] == D_STATE, "state vector is %d, STATE_FIELDS says %d" % (a.shape[0], D_STATE)
    return a


def _parse_xy(action_id, prefix):
    try:
        rest = action_id[len(prefix):]
        xs, ys = rest.split("_", 1)
        return float(xs), float(ys)
    except Exception:                                           # noqa: BLE001
        return None


def encode_actions(state, actions):
    me = state.get("self") or {}
    b, w, h, span = _board(state)
    mx, my = (me.get("tile") or [0, 0])[:2]
    mx, my = _f(mx), _f(my)
    seat = int(_f(me.get("seat"), -1))
    team = _f(me.get("team"), -1)
    stars = _f(me.get("stars"))
    my_tier = _f(me.get("tier"))

    pts = b.get("points") or []
    free_pts = [p for p in pts if not p.get("is_base") and p.get("relationship") == "unclaimed"]
    my_base = [p for p in pts if p.get("is_base") and int(_f(p.get("held_by_seat"), -1)) == seat]
    enemies = [p for p in (state.get("players") or [])
               if not p.get("is_self") and _f(p.get("team"), -2) != team
               and p.get("status") != "knocked out"]
    by_seat = {int(_f(p.get("seat"), -1)): p for p in (state.get("players") or [])}

    def nearest_to(x, y, items):
        best = span
        for it in items:
            t = it.get("tile") or [0, 0]
            best = min(best, _cheb(x, y, _f(t[0]), _f(t[1])))
        return best

    base_now = nearest_to(mx, my, my_base)
    # Parsed ONCE for the whole action set rather than per row: the legal set reaches ~670 actions
    # and the grid is identical for all of them.
    grid = _grid(state) if _COVER else []
    cover_now = _cover_at(grid, mx, my) if _COVER else 0.0
    out = np.zeros((len(actions), D_ACTION), dtype=np.float32)
    for i, a in enumerate(actions):
        aid = str(a.get("action_id", ""))
        atype = str(a.get("type", ""))
        row = out[i]
        ti = _TYPE_INDEX.get(atype)
        if ti is not None:
            row[ti] = 1.0
        base = len(ACTION_TYPES)

        if aid.startswith("move_") or aid.startswith("pick_"):
            xy = _parse_xy(aid, "move_" if aid.startswith("move_") else "pick_")
            if xy:
                tx, ty = xy
                row[base + 0] = _cheb(mx, my, tx, ty) / span
                # A POINT IS A SQUARE, NOT A TILE. `playerIsInPoint` tests a chebyshev square of
                # half-width `radius`, so a base with radius 1 can be tiered from any of NINE
                # tiles. This asked for exact equality, which marked eight of those nine as "not
                # a base" -- on the only feature that identifies the winning tile.
                #
                # That is the original `is_base` bug one layer up: not a field that was always
                # false, but a field that was true one ninth as often as it should be, which is
                # far harder to see. check_features would report it as alive and plausible.
                for pt in pts:
                    t = pt.get("tile") or [0, 0]
                    if _cheb(_f(t[0]), _f(t[1]), tx, ty) <= _point_radius(pt):
                        if pt.get("is_base"):
                            row[base + 2] = 1.0
                        elif pt.get("relationship") != "friendly":
                            row[base + 1] = 1.0
                        row[base + 3] = _f(pt.get("stars_per_turn")) / 3.0
                        break
                row[base + 4] = nearest_to(tx, ty, free_pts) / span
                row[base + 5] = nearest_to(tx, ty, enemies) / span
                # Positive when the move closes distance to your own base, which is the only
                # place tiering is possible.
                row[base + 6] = (base_now - nearest_to(tx, ty, my_base)) / span

        if atype == "attack":
            row[base + 7] = _f(a.get("hit_chance"))
            tgt = by_seat.get(int(_f(a.get("target_seat"), -1)))
            if tgt:
                hp = _f(tgt.get("health"))
                row[base + 8] = hp / 6.0
                row[base + 9] = _f(tgt.get("tier")) / 4.0
                t = tgt.get("tile") or [0, 0]
                row[base + 10] = _cheb(mx, my, _f(t[0]), _f(t[1])) / span
                # A shot that could end them is worth distinguishing: kills feed the KO tier
                # track and remove a competitor for points.
                row[base + 11] = 1.0 if hp <= 2.0 else 0.0

        if atype in ("play_card", "discard_card"):
            cost = _f(a.get("cost_stars"))
            row[base + 12] = math.tanh(cost / 100.0)
            row[base + 13] = _f(a.get("tier_required")) / 4.0
            row[base + 14] = 1.0 if (stars >= cost and my_tier >= _f(a.get("tier_required"))) else 0.0

        if _COVER:
            # Defaults are the CURRENT tile's values, not zero. A non-move action leaves you where
            # you are, so "the cover I would have" is the cover I have -- zero would mean "this
            # action puts me in the open", which is a different and false claim.
            row[base + 15] = cover_now
            if aid.startswith("move_") or aid.startswith("pick_"):
                xy = _parse_xy(aid, "move_" if aid.startswith("move_") else "pick_")
                if xy:
                    row[base + 15] = _cover_at(grid, xy[0], xy[1])

    return out
