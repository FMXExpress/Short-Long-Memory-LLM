"""Score guard outputs from build/jev_guard: python3 bench/jev/score.py bench/jev/out-*.json"""
import json, sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def stats(rows, decide):
    right = [r for r in rows if r["right"]]
    wrong = [r for r in rows if not r["right"]]
    keep = sum(decide(r) for r in right)
    stop = sum(not decide(r) for r in wrong)
    return keep, len(right), stop, len(wrong), (keep + stop) / max(1, len(rows))


def auc(rows):
    pos = [r["p_yes"] for r in rows if r["right"]]
    neg = [r["p_yes"] for r in rows if not r["right"]]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def line(name, s):
    keep, nr, stop, nw, acc = s
    return f"  {name:28s} right answers kept {keep:2d}/{nr:<2d}  wrong answers blocked {stop:2d}/{nw:<2d}  accuracy {acc:.2f}"


for path in sys.argv[1:]:
    rows = load(path)
    print(f"== {path}: {len(rows)} cases, median {sorted(r['ms'] for r in rows)[len(rows) // 2]} ms per readout")
    print(line("current guard", stats(rows, lambda r: r["guard_allow"])))
    print(line("readout p_yes >= 0.5", stats(rows, lambda r: r["p_yes"] >= 0.5)))
    best = max((stats(rows, lambda r, t=t: r["p_yes"] >= t)[4], t) for t in [i / 20 for i in range(1, 20)])
    print(line(f"readout, best threshold {best[1]:.2f}", stats(rows, lambda r: r["p_yes"] >= best[1])))
    print(f"  readout AUC {auc(rows):.3f}  (0.5 = chance)")
    for t in ("bridge", "inverse", "verify"):
        sub = [r for r in rows if r["type"] == t]
        g, j = stats(sub, lambda r: r["guard_allow"]), stats(sub, lambda r: r["p_yes"] >= 0.5)
        print(f"    {t:8s} guard acc {g[4]:.2f} (kept {g[0]}/{g[1]}, blocked {g[2]}/{g[3]})   "
              f"readout acc {j[4]:.2f} (kept {j[0]}/{j[1]}, blocked {j[2]}/{j[3]})   AUC {auc(sub):.2f}")
