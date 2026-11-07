"""Score the merge decision: python3 bench/jev/score_merge.py bench/jev/out-merge-*.json

A wrong merge overwrites a true fact; a missed merge keeps a duplicate line. Both are
counted, and thresholds are compared at a cost of WRONG_COST per wrong merge and 1 per
missed one (a lost fact is worse than a duplicate, Hightouch's over-merge lesson).
"""
import json
import sys

WRONG_COST = 5


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def tally(rows, merge):
    wrong = sum(1 for r in rows if merge(r) and not r["replaces"])
    missed = sum(1 for r in rows if not merge(r) and r["replaces"])
    done = sum(1 for r in rows if merge(r) and r["replaces"])
    return wrong, missed, done


def show(name, rows, merge):
    w, m, d = tally(rows, merge)
    n1 = sum(r["replaces"] for r in rows)
    print(f"  {name:34s} wrong merges {w:2d}/{len(rows) - n1:<2d}  missed {m:2d}/{n1:<2d}  "
          f"cost {WRONG_COST * w + m:3d}")


for path in sys.argv[1:]:
    rows = load(path)
    print(f"== {path}: {len(rows)} pairs, median {sorted(r['ms'] for r in rows)[len(rows) // 2]} ms per judgement")
    show("rules (today, no judge)", rows, lambda r: r["rules"])
    ths = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    for t in ths:
        show(f"judge alone, p >= {t}", rows, lambda r, t=t: r["p"] >= t)
    for t in ths:
        show(f"rules and judge p >= {t}", rows, lambda r, t=t: r["rules"] and r["p"] >= t)
    best = min(((WRONG_COST * tally(rows, lambda r, t=t: r["rules"] and r["p"] >= t)[0] +
                 tally(rows, lambda r, t=t: r["rules"] and r["p"] >= t)[1], t) for t in ths))
    print(f"  lowest cost with the rules: merge_min {best[1]} (cost {best[0]})")
    print("  by category: label, rules merged, p (judge)")
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    for c, rs in cats.items():
        print(f"    {c:24s} label {rs[0]['replaces']}  rules {sum(r['rules'] for r in rs)}/{len(rs)}  "
              f"p {' '.join(f'{r['p']:.2f}' for r in rs)}")
