#!/usr/bin/env python3
"""Does a q8_0 KV cache change what the model says?

Every question in bench/facts.json (the two tests per fact, plus the unknown ones) is asked
with all 30 facts in the system prompt, as retrieval would put them there, once with
[llama] kv_cache = f16 and once with q8_0, decoding greedily (temperature 0, 40 tokens).
Reports how many answers are identical word for word, how many still name the expected
value under each cache, and the answers that differ.

  python3 bench/kv_check.py --profile qwen3-0.6b [--out bench/results/kv-qwen3-0.6b.json]
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BENCH_LLAMA = os.path.join(REPO, "build", "bench_llama")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def run(profile, kv, asks, work):
    os.makedirs(work, exist_ok=True)
    src, dst = os.path.join(work, "ask.json"), os.path.join(work, f"answers-{kv}.json")
    json.dump(asks, open(src, "w", encoding="utf-8"))
    data = os.path.join(work, f"data-{kv}")
    os.makedirs(data, exist_ok=True)
    subprocess.run([BENCH_LLAMA, "--config", os.path.join(REPO, "memagent.ini"), "--data", data,
                    "--profile", profile, "--set", f"kv_cache={kv}", "ask", "-", src, dst], check=True,
                   env=dict(os.environ, MEMAGENT_TEMPERATURE="0"))
    return json.load(open(dst, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="qwen3-0.6b")
    ap.add_argument("--out")
    ap.add_argument("--a", default="f16", help="first kv_cache setting")
    ap.add_argument("--b", default="q8_0", help="second (f16 against f16: is decoding repeatable?)")
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "facts.json"), encoding="utf-8"))
    priors = open(os.path.join(REPO, "priors.md"), encoding="utf-8").read().strip()
    memory = "What you know about the user:\n" + "\n".join("- " + f["text"] for f in data["facts"])
    system = priors + "\n\n" + memory
    asks, want = [], []
    for f in data["facts"]:
        for q in f["tests"]:
            asks.append({"system": system, "user": q, "lora": False})
            want.append(f["value"])
    for q in data.get("unknown", []):
        asks.append({"system": system, "user": q if isinstance(q, str) else q.get("q", ""), "lora": False})
        want.append(None)
    work = tempfile.mkdtemp(prefix="kvcheck-")
    f16 = run(a.profile, a.a, asks, os.path.join(work, "a"))
    q8 = run(a.profile, a.b, asks, os.path.join(work, "b"))
    same = sum(1 for x, y in zip(f16, q8) if x == y)
    hit = lambda ans: sum(1 for x, w in zip(ans, want) if w and norm(w) in norm(x))
    n_known = sum(1 for w in want if w)
    res = {"profile": a.profile, "a": a.a, "b": a.b, "questions": len(asks), "identical": same,
           "value_named_a": hit(f16), "value_named_b": hit(q8), "with_value": n_known,
           "differ": [{"user": s["user"], a.a: x, a.b + ("" if a.b != a.a else " again"): y} for s, x, y in zip(asks, f16, q8) if x != y]}
    print(json.dumps({k: v for k, v in res.items() if k != "differ"}))
    for d in res["differ"]:
        v = list(d.values())
        print(f"  {v[0]}\n    a: {v[1]!r}\n    b: {v[2]!r}")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
