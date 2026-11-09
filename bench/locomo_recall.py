#!/usr/bin/env python3
"""How much of each LoCoMo answer's evidence retrieval puts in reach, without the model.

LoCoMo marks the turns that hold each answer (qa[].evidence, "D1:3"). memagent stores
turn n of the conversation as message n (bench_locomo.py loads them in order with /log),
so the evidence is a set of message ids. build/msg_probe returns what retrieval keeps
for each question; this reports, per category, the share of evidence turns in the top k
and the share of questions with all their evidence in the top k.

  python3 bench/locomo_recall.py --data locomo10.json --db <data dir>/memory.db \
      [--sample 0] [--pool 30] [--ks 3,5,8,12] [--hits out.jsonl]

The db must hold the conversation as its first messages (a bench_locomo.py --keep dir).
Build the probe first: fpc -O2 -Sh -Fumemagent -FUbuild/units -obuild/msg_probe memagent/tests/msg_probe.lpr
"""
import argparse, json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_locomo import CATS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dia_ids(sample):
    conv = sample["conversation"]
    out, i = [], 1
    while f"session_{i}" in conv:
        out += [t["dia_id"] for t in conv[f"session_{i}"]]
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--ks", default="3,5,8,12")
    ap.add_argument("--hits", help="reuse (if present) or save the probe's output here")
    a = ap.parse_args()
    sample = json.load(open(a.data))[a.sample]
    ids = {d: n + 1 for n, d in enumerate(dia_ids(sample))}
    qs = [q for q in sample["qa"] if q.get("category") in CATS]
    if a.hits and os.path.exists(a.hits):
        hits = [json.loads(l) for l in open(a.hits)]
    else:
        inp = "".join(q["question"].replace("\n", " ") + "\n" for q in qs)
        res = subprocess.run([os.path.join(ROOT, "build", "msg_probe"), os.path.join(ROOT, "cache"), a.db,
                              str(len(ids) + 1), str(a.pool)], input=inp, capture_output=True, text=True, check=True)
        hits = [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")]
        if a.hits:
            open(a.hits, "w").write("".join(json.dumps(h) + "\n" for h in hits))
    assert len(hits) == len(qs), (len(hits), len(qs))
    ks = [int(k) for k in a.ks.split(",")]
    stats = {}
    for q, h in zip(qs, hits):
        ev = {ids[e] for e in q.get("evidence", []) if e in ids}
        if not ev:
            continue
        got = [x["id"] for x in h["hits"]]
        s = stats.setdefault(CATS[q["category"]], {"n": 0, **{k: [0.0, 0] for k in ks}})
        s["n"] += 1
        for k in ks:
            found = len(ev & set(got[:k]))
            s[k][0] += found / len(ev)
            s[k][1] += found == len(ev)
    print(f"{'category':12} {'n':>4} " + " ".join(f"  share@{k:<3} all@{k:<3}" for k in ks))
    for c, s in stats.items():
        print(f"{c:12} {s['n']:>4} " + " ".join(f"  {s[k][0]/s['n']:.2f}      {s[k][1]/s['n']:.2f}   " for k in ks))


if __name__ == "__main__":
    main()
