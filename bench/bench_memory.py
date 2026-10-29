"""Memory benchmark: how well does the agent know facts it was told?

Trains a memory LoRA on bench/facts.json with the real trainer
(trainer/consolidate.py), then asks every fact's held-out natural questions
in four conditions, as the agent would ask them (priors as system prompt):

  base        base model, no memory context        (floor: should be ~0)
  rag         base model + the fact in the prompt  (RAG with perfect retrieval)
  lora        LoRA, no memory context              (what consolidation bought)
  lora+rag    LoRA + the fact in the prompt

and measures confabulation (a stored value in answers to questions about
things never stated) and interference (another fact's value in the answer).

  python3 bench/bench_memory.py --profile tiny --out bench/results/x.json
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "trainer"))
import memlib  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def has(ans, value):
    return f" {norm(value)} " in f" {norm(ans)} "


def build_db(path, facts):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE facts(id INTEGER PRIMARY KEY, text TEXT, subject TEXT, value TEXT, questions TEXT,"
                " kind TEXT, confidence REAL, source_msg INTEGER, created_at INTEGER,"
                " superseded INTEGER DEFAULT 0, forgotten INTEGER DEFAULT 0, baked_in TEXT DEFAULT '')")
    for i, f in enumerate(facts, 1):
        con.execute("INSERT INTO facts(id,text,subject,value,questions,kind,confidence) VALUES(?,?,?,?,?,?,?)",
                    (i, f["text"], f["subject"], f["value"], json.dumps(memlib.subject_questions(f["subject"])),
                     "user", 0.9))
    con.commit()
    con.close()


def ask(tok, model, priors, q, memory=None):
    sys_msg = priors
    if memory:
        sys_msg += "\n\nRelevant memory:\n- " + memory
    ans, _, _ = memlib.generate(tok, model, [{"role": "system", "content": sys_msg},
                                             {"role": "user", "content": q}], 40)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--models")
    ap.add_argument("--facts", default=os.path.join(HERE, "facts.json"))
    ap.add_argument("--n", type=int, default=0, help="use only the first N facts")
    ap.add_argument("--adapter", help="evaluate an existing adapter instead of training one")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    data = json.load(open(a.facts, encoding="utf-8"))
    facts = data["facts"][:a.n] if a.n else data["facts"]
    priors = open(os.path.join(REPO, "priors.md"), encoding="utf-8").read().strip()
    prof = memlib.load_profile(a.profile, a.models)
    work = tempfile.mkdtemp(prefix="benchmem-")

    # 1. extraction: does the rule extractor store the value we expect?
    ext_ok = sum(1 for f in facts if any(norm(x["value"]) == norm(f["value"])
                                         for x in memlib.extract_facts_rules(f["say"])))

    # 2. consolidation, with the production trainer
    train = {}
    adapter = a.adapter
    if not adapter:
        db = os.path.join(work, "memory.db")
        build_db(db, facts)
        cmd = [sys.executable, os.path.join(REPO, "trainer", "consolidate.py"), "--db", db,
               "--out", os.path.join(work, "adapters"), "--profile", a.profile,
               "--priors", os.path.join(REPO, "priors.md"), "--no-gate"]
        if a.models:
            cmd += ["--models", a.models]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        train = json.loads(p.stdout.strip().splitlines()[-1])
        train["wall_seconds"] = round(time.time() - t, 1)
        if train.get("exit") not in (0,):
            print(p.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"training failed: {train}")
        adapter = train["path"]

    # 3. evaluation
    from peft import PeftModel
    tok, model, _ = memlib.load_base(prof)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    values = [f["value"] for f in facts]
    conds = {"base": (False, False), "rag": (False, True), "lora": (True, False), "lora+rag": (True, True)}
    score = {c: 0 for c in conds}
    wrong = {c: 0 for c in conds}
    rows = []
    t = time.time()
    for i, f in enumerate(facts):
        for q in f["tests"]:
            row = {"q": q, "value": f["value"]}
            for c, (use_lora, use_rag) in conds.items():
                if use_lora:
                    ans = ask(tok, model, priors, q, f["text"] if use_rag else None)
                else:
                    with model.disable_adapter():
                        ans = ask(tok, model, priors, q, f["text"] if use_rag else None)
                ok = has(ans, f["value"])
                score[c] += ok
                if not ok and any(has(ans, v) for j, v in enumerate(values) if j != i):
                    wrong[c] += 1
                row[c] = ans
            rows.append(row)
    # confabulation: questions about things never stated. "near" conditions put
    # the look-alike fact a retriever would return into the prompt, which is
    # what the agent's RAG does for most of these questions.
    by_subject = {f["subject"]: f for f in facts}
    conf = {"base": 0, "lora": 0, "rag-near": 0, "lora+rag-near": 0}
    n_near = 0
    unknown_rows = []
    for u in data["unknown"]:
        u = u if isinstance(u, dict) else {"q": u, "near": None}
        near = by_subject.get(u.get("near") or "")
        row = {"q": u["q"], "near": near["text"] if near else None}
        with model.disable_adapter():
            row["base"] = ask(tok, model, priors, u["q"])
            if near:
                row["rag-near"] = ask(tok, model, priors, u["q"], near["text"])
        row["lora"] = ask(tok, model, priors, u["q"])
        if near:
            n_near += 1
            row["lora+rag-near"] = ask(tok, model, priors, u["q"], near["text"])
        for c in conf:
            if c in row:
                conf[c] += any(has(row[c], v) for v in values)
        unknown_rows.append(row)
    n_q = len(rows)
    res = {
        "label": a.label, "profile": a.profile, "n_facts": len(facts), "n_questions": n_q,
        "extraction_value_acc": round(ext_ok / len(facts), 3),
        "accuracy": {c: round(score[c] / n_q, 3) for c in conds},
        "interference": {c: round(wrong[c] / n_q, 3) for c in conds},
        "confabulation": {c: round(conf[c] / (n_near if "near" in c else len(data["unknown"])), 3)
                          for c in conf},
        "train": {k: train.get(k) for k in ("recall", "ppl_ratio", "seconds", "wall_seconds", "gate_ok")},
        "eval_seconds": round(time.time() - t, 1),
        "answers": rows,
        "unknown_answers": unknown_rows,
    }
    summary = {k: v for k, v in res.items() if k not in ("answers", "unknown_answers")}
    print(json.dumps(summary, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
