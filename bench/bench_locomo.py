"""LoCoMo (Maharana et al. 2024, arXiv 2402.17753) on the real memagent binary.

A standard long-term conversational memory benchmark, so memagent's numbers can be set
beside published ones. Each LoCoMo sample is one long conversation between two people
over many dated sessions, then questions about it. memagent's fact extraction is built
for one user's "my sister ..." statements, so this measures its conversation memory:
each turn goes in with /log (stored and indexed, no reply), as "[<session date>]
<speaker>: <text>", and each question is asked as a normal turn, answered from the
retrieved messages.

Scored with LoCoMo's token F1 (lower-case, no punctuation or articles), per category as
Mem0 and A-MEM report them: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop;
category 5 (adversarial, unanswerable) is left out as most papers do.

  curl -sSL -o locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
  python3 bench/bench_locomo.py --data locomo10.json --sample 0 --out bench/results/locomo-small-s0.json
"""
import argparse
import collections
import json
import os
import re
import shutil
import string
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_integration import Agent, answer_text, blocks  # noqa: E402

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if not p or not g or n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def turns(sample):
    conv = sample["conversation"]
    i = 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        for t in conv[f"session_{i}"]:
            text = t.get("text", "")
            if t.get("blip_caption"):
                text += f" (shared a photo: {t['blip_caption']})"
            yield f"[{date}] {t['speaker']}: {text}".replace("\n", " ")
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--profile", default="small")
    ap.add_argument("--rag-k", type=int, default=3)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="leave out the first N questions (run a sample in parts)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep")
    ap.add_argument("--out")
    a = ap.parse_args()

    sample = json.load(open(a.data, encoding="utf-8"))[a.sample]
    qs = [q for q in sample["qa"] if q.get("category") in CATS and "answer" in q]
    qs = qs[a.skip:]
    if a.max_questions:
        qs = qs[:a.max_questions]
    work = a.keep or tempfile.mkdtemp(prefix="locomo-")
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, "data")
    shutil.rmtree(data, ignore_errors=True)
    agent = Agent(work, a.profile, a.rag_k, a.threads, os.path.join(os.path.dirname(HERE), "models.json"), a.hops)
    lines = ["/log " + t for t in turns(sample)] + [q["question"].replace("\n", " ") for q in qs]
    txt, secs = agent.run(data, lines, "locomo")
    ans = {inp: answer_text(out) for inp, out in blocks(txt) if not inp.startswith("/")}
    rows, by = [], collections.defaultdict(list)
    for q in qs:
        got = ans.get(q["question"].replace("\n", " "), "")
        s = f1(got, q["answer"])
        rows.append({"category": CATS[q["category"]], "question": q["question"], "answer": q["answer"], "got": got,
                     "f1": round(s, 4)})
        by[CATS[q["category"]]].append(s)
    res = {"sample": sample.get("sample_id", a.sample), "profile": a.profile, "rag_k": a.rag_k, "hops": a.hops,
           "turns": sum(1 for _ in turns(sample)), "questions": len(qs), "seconds": round(secs),
           "f1": {c: round(sum(v) / len(v), 4) for c, v in by.items()},
           "f1_all": round(sum(r["f1"] for r in rows) / max(1, len(rows)), 4),
           "counts": {c: len(v) for c, v in by.items()}}
    print(json.dumps(res, indent=1))
    if a.out:
        res["answers"] = rows
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
