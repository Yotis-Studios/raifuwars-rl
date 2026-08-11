"""Encode a corpus and report what every feature actually does.

    python tools/check_features.py <traces.jsonl> [--n 5000]

A feature that is always zero, always constant, or occasionally NaN costs nothing to compute and
silently removes capacity from the model -- and after a day-long run there is no way to tell which
of thirty numbers was dead. Cheaper to look now.

Three faults it reports, in descending nastiness:

    NaN / inf     poisons a gradient and the loss goes to nan hours later with no clue why
    constant      a column carrying no information; either the feature is broken or it is
                  measuring something this corpus never varies
    near-constant std under 1% of the range: alive but nearly useless, worth a second look

It also prints the range, because a feature that reaches 40 when its neighbours reach 1 will
dominate the first layer regardless of whether it deserves to.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raifuwars_rl.features import (ACTION_FIELDS, STATE_FIELDS,   # noqa: E402
                                   encode_actions, encode_state)


def report(name, mat, fields):
    print("\n=== %s: %d rows x %d features ===" % (name, mat.shape[0], mat.shape[1]))
    bad_nan, const, near = [], [], []
    print("  %-26s %9s %9s %9s %9s" % ("feature", "mean", "std", "min", "max"))
    for i, f in enumerate(fields):
        col = mat[:, i]
        n_bad = int(np.sum(~np.isfinite(col)))
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            bad_nan.append(f)
            print("  %-26s %s" % (f, "ALL NON-FINITE"))
            continue
        mu, sd, lo, hi = finite.mean(), finite.std(), finite.min(), finite.max()
        flag = ""
        if n_bad:
            bad_nan.append(f)
            flag = "  <-- %d non-finite" % n_bad
        elif sd == 0:
            const.append(f)
            flag = "  <-- CONSTANT"
        elif sd < 0.01 * max(1e-9, hi - lo):
            near.append(f)
            flag = "  <-- near-constant"
        print("  %-26s %9.3f %9.3f %9.3f %9.3f%s" % (f, mu, sd, lo, hi, flag))

    print("\n  non-finite : %s" % (", ".join(bad_nan) or "none"))
    print("  constant   : %s" % (", ".join(const) or "none"))
    print("  near-const : %s" % (", ".join(near) or "none"))
    return bad_nan, const


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    S, A = [], []
    seen = 0
    with open(args.traces, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = d.get("state") or {}
            acts = d.get("available_actions") or []
            if not acts:
                continue
            S.append(encode_state(st))
            A.append(encode_actions(st, acts))
            seen += 1
            if seen >= args.n:
                break

    if not S:
        print("no usable rows")
        return 1
    smat = np.stack(S)
    amat = np.concatenate(A, axis=0)

    nan_s, const_s = report("state", smat, STATE_FIELDS)
    nan_a, const_a = report("action", amat, ACTION_FIELDS)

    if nan_s or nan_a:
        print("\nFAIL: non-finite features will poison training.")
        return 1
    print("\nOK: no non-finite features. Constants above are worth reading before a long run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
