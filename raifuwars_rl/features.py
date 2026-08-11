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

import numpy as np

ACTION_TYPES = ["move", "attack", "rush", "reload", "play_card", "discard_card",
                "tier_up", "tier_choice", "select_tile", "cancel", "end_turn", "chat"]
_TYPE_INDEX = {t: i for i, t in enumerate(ACTION_TYPES)}

STATE_FIELDS = [
    "hp_frac", "ammo_frac", "tier_frac", "tier_progress", "stars_log", "kills", "deaths",
    "range_frac", "fortified", "move_roll_frac", "moved", "attacked", "rushed", "played_card",
    "tiered", "in_territory", "hand_frac", "pos_x", "pos_y", "turn_log",
    "n_enemies_alive", "enemy_tier_max", "enemy_tier_lead", "nearest_enemy_dist",
    "nearest_enemy_hp", "enemy_stars_lead",
    "points_mine", "points_enemy", "points_free", "dist_nearest_free_point",
    "dist_nearest_enemy_point", "dist_own_base", "can_tier_now",
]
D_STATE = len(STATE_FIELDS)

ACTION_FIELDS = (["type_" + t for t in ACTION_TYPES] + [
    "dest_dist", "dest_captures", "dest_is_base", "dest_point_value",
    "dest_dist_to_free_point", "dest_dist_to_nearest_enemy", "dest_toward_base",
    "hit_chance", "target_hp_frac", "target_tier_frac", "target_dist", "target_would_kill",
    "cost_frac", "tier_required", "affordable",
])
D_ACTION = len(ACTION_FIELDS)


def _f(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(x) or math.isinf(x)) else x


def _cheb(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


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
                d_base = min(d_base, d)
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
                for pt in pts:
                    t = pt.get("tile") or [0, 0]
                    if _f(t[0]) == tx and _f(t[1]) == ty:
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

    return out
