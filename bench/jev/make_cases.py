"""Guard test cases from the GPU integration runs: (question, answer, right or wrong).

Each case is an answer a model really gave (bench/results/gpu/*.json, retrieval and
retrieval + sleep arms, no guard in that harness), labelled by the benchmark's scorer.
Abstentions are left out: a guard has nothing to decide there. The facts a judge may
use are what two-hop retrieval returns for the question (integration-small-hops2.json);
the current guard also checks against every stored fact (facts.json).

  python3 bench/jev/make_cases.py   -> bench/jev/cases.json, bench/jev/facts.json
"""
import json, os, random, sqlite3, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "trainer"))
import memlib  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "bench"))
from bench_integration_hf import recalled_facts  # noqa: E402

hops2 = {r["q"]: r["recalled"] for r in json.load(open(os.path.join(REPO, "bench/results/integration-small-hops2.json")))["answers"]}
runs = {"1.5B": "integration-hf-small.json", "4B": "integration-hf-medium.json", "24B": "integration-hf-large-gentle.json"}
import re, unicodedata
NAMES = set()
for f in json.load(open(os.path.join(REPO, "bench/facts_integration.json")))["facts"]:
    m = re.match(r"^(?:My \w+'s name is (\w+)|(\w+)'s )", f["say"])
    if m:
        NAMES.add(m.group(1) or m.group(2))


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def strict_right(row, ans):
    """the scorer, without accents tripping it, and an inverse answer must name only
    the right relative: "Omar, Priya and Bruno" claims two strangers are relatives"""
    has = lambda v: norm(v) in norm(ans)
    ok = any(has(v) for v in row["accept"])
    if row["type"] == "verify":
        ok = ok and not has("no" if row["accept"] == ["yes"] else "yes")
    if row["type"] == "inverse" and ok:
        ok = not [n for n in NAMES if has(n) and n not in row["accept"]]
    return ok


cases = []
for model, f in runs.items():
    r = json.load(open(os.path.join(REPO, "bench/results/gpu", f)))
    for row in r["answers"]:
        for arm in ("retrieval", "sleep"):
            if arm not in row or memlib.is_abstain(row[arm]) or not row[arm].strip():
                continue
            cases.append({"model": model, "arm": arm, "type": row["type"], "q": row["q"], "answer": row[arm][:300],
                          "right": strict_right(row, row[arm]), "facts": recalled_facts(hops2.get(row["q"], ""), 6)})
# balance right and wrong answers, per type, so a judge that allows everything scores 50%
rng = random.Random(11)
out = []
for t in ("direct", "bridge", "inverse", "verify"):
    right = [c for c in cases if c["type"] == t and c["right"]]
    wrong = [c for c in cases if c["type"] == t and not c["right"]]
    rng.shuffle(right); rng.shuffle(wrong)
    k = min(20, len(right), len(wrong))
    out += right[:k] + wrong[:k]
for i, c in enumerate(out):
    c["id"] = i
db = sqlite3.connect(f"file:{sys.argv[1] if len(sys.argv) > 1 else '/home/user/benchint-hop2/told/memory.db'}?mode=ro", uri=True)
facts = [{"text": t, "subject": s, "value": v} for t, s, v in
         db.execute("select text, subject, value from facts where superseded=0 and forgotten=0 order by id")]
json.dump(out, open(os.path.join(HERE, "cases.json"), "w"), indent=1, ensure_ascii=False)
json.dump(facts, open(os.path.join(HERE, "facts.json"), "w"), indent=1, ensure_ascii=False)
print(len(cases), "answers;", len(out), "cases:", {t: sum(1 for c in out if c["type"] == t) for t in ("direct", "bridge", "inverse", "verify")},
      "right", sum(c["right"] for c in out), "| facts", len(facts))
