"""Does consolidation into the weights buy anything retrieval does not?

The real memagent binary, bench/facts_integration.json (gen_integration.py), two arms
that hold the same facts:

  retrieval   facts stored (/remember), never trained: retrieval + guard, base model
  consolidated  the same store after a deep sleep (/sleep): retrieval + deep LoRA + guard
  weights     the consolidated model with no retrieval and no guard (/weights)

Each question type is scored per arm, and split by whether retrieval put every fact the
question needs into the prompt (/recall, top rag_k). If the LoRA only helps where
retrieval misses, the weights are a backup for retrieval; if it helps on bridge and
inverse questions where retrieval did find both facts, it is integrating them, which is
what complementary-learning-systems theory says slow learning is for.

Each arm runs in its own process on a copy of the data, so answers from one arm are
never retrievable as chat history in another.

  python3 bench/bench_integration.py --profile small --out bench/results/integration-small.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BIN = os.path.join(REPO, "build", "memagent")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def has(ans, value):
    return f" {norm(value)} " in f" {norm(ans)} "


def blocks(transcript):
    """[(input line, raw output)] from a piped memagent transcript."""
    out, cur, buf = [], None, []
    for line in transcript.splitlines():
        if line.startswith("you> "):
            if cur is not None:
                out.append((cur, "\n".join(buf).strip()))
            cur, buf = line[5:], []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out.append((cur, "\n".join(buf).strip()))
    return out


def answer_text(raw):
    """the reply without the agent's [bracketed] status lines"""
    keep = []
    for line in raw.splitlines():
        for p in ("pal> ", "weights> "):
            if line.startswith(p):
                line = line[len(p):]
        if line.startswith("[") or not line.strip():
            continue
        keep.append(line)
    return " ".join(keep).strip()


def stored_text(say):
    """what the rule extractor stores for a fact said as "My X's name is N." """
    return re.sub(r"^My ", "Your ", say)


class Agent:
    def __init__(self, work, profile, rag_k, threads, models):
        self.work, self.profile, self.rag_k, self.threads, self.models = work, profile, rag_k, threads, models

    def ini(self, data):
        path = os.path.join(self.work, os.path.basename(data) + ".ini")
        cfg = {
            "agent": {"profile": self.profile, "engine": "llama", "models": self.models,
                      "priors": f"{REPO}/priors.md", "data_dir": data, "repo_dir": REPO,
                      "recent_turns": "0", "rag_k": str(self.rag_k)},
            "llama": {"threads": str(self.threads)},
            "gpu": {"infer": "-1", "train": "-1"},
            "consolidate": {"auto": "0", "trainer": "llama", "tiers": "on", "threads": str(self.threads)},
            "retrieval": {"cache_dir": f"{REPO}/cache"},
        }
        with open(path, "w") as f:
            for sec, kv in cfg.items():
                f.write(f"[{sec}]\n" + "".join(f"{k}={v}\n" for k, v in kv.items()))
        return path

    def run(self, data, lines, name):
        tpath = os.path.join(self.work, name + ".txt")
        t = time.time()
        with open(tpath, "w") as out:
            subprocess.run([BIN, "--config", self.ini(data)], input="\n".join(lines + ["/quit"]) + "\n",
                           stdout=out, stderr=subprocess.STDOUT, text=True,
                           env=dict(os.environ, MEMAGENT_TEMPERATURE="0"))
        secs = time.time() - t
        print(f"[bench_integration] {name}: {secs:.0f} s", flush=True)
        return open(tpath, encoding="utf-8").read(), secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="small")
    ap.add_argument("--data", default=os.path.join(HERE, "facts_integration.json"))
    ap.add_argument("--per-type", type=int, default=0, help="use only the first N questions of each type")
    ap.add_argument("--rag-k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--arms", default="retrieval,consolidated,weights",
                    help="which arms to run now; 'sleep' only trains. Scores include arms run earlier in --keep")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a profile setting in a copy of models.json: train.KEY by default (paraphrases=4), "
                         "or SECTION.KEY (gate.max_ppl_ratio=3)")
    ap.add_argument("--keep", help="reuse (or keep) the work directory, e.g. to skip a finished deep sleep")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    d = json.load(open(a.data, encoding="utf-8"))
    qs = [q for t in d["questions"] for q in (d["questions"][t][:a.per_type] if a.per_type else d["questions"][t])]
    work = a.keep or tempfile.mkdtemp(prefix="benchint-")
    os.makedirs(work, exist_ok=True)
    models = f"{REPO}/models.json"
    if a.set:
        cfg = json.load(open(models, encoding="utf-8"))
        for kv in a.set:
            key, val = kv.split("=", 1)
            sec, key = key.split(".", 1) if "." in key else ("train", key)   # gate.max_ppl_ratio=3
            cfg["profiles"][a.profile].setdefault(sec, {})[key] = json.loads(val) if re.fullmatch(r"-?[\d.]+", val) else val
        models = os.path.join(work, "models.json")
        json.dump(cfg, open(models, "w", encoding="utf-8"), indent=1)
    agent = Agent(work, a.profile, a.rag_k, a.threads, models)
    arms = a.arms.split(",")
    res = {"label": a.label, "profile": a.profile, "settings": a.set, "rag_k": a.rag_k, "n_facts": len(d["facts"]),
           "n_questions": len(qs), "seconds": {}, "work": work}

    # 1. the store: every fact through the rule extractor, as the agent stores what it is told
    told = os.path.join(work, "told")
    if not os.path.exists(os.path.join(told, "memory.db")):
        txt, res["seconds"]["remember"] = agent.run(told, [f"/remember {f['say']}" for f in d["facts"]], "remember")
        res["stored"] = len(re.findall(r"\[memory \+", txt))

    # 2. what retrieval puts in the prompt for each question
    recall_dir = os.path.join(work, "recall")
    shutil.rmtree(recall_dir, ignore_errors=True)
    shutil.copytree(told, recall_dir)
    txt, res["seconds"]["recall"] = agent.run(recall_dir, [f"/recall {q['q']}" for q in qs], "recall")
    recalled = {}
    for inp, out in blocks(txt):
        if inp.startswith("/recall "):
            facts = [ln for ln in out.splitlines() if ln.strip().startswith("fact #")][:a.rag_k]
            recalled[inp[8:]] = "\n".join(facts)

    def covered(q):
        got = recalled.get(q["q"], "")
        return all(stored_text(n)[:-1] in got for n in q["needs"])

    # 3. the arms
    answers = {}
    if "retrieval" in arms:
        dr = os.path.join(work, "arm-retrieval")
        shutil.rmtree(dr, ignore_errors=True)
        shutil.copytree(told, dr)
        txt, res["seconds"]["retrieval"] = agent.run(dr, [q["q"] for q in qs], "arm-retrieval")
        answers["retrieval"] = {inp: answer_text(out) for inp, out in blocks(txt)}
        json.dump(answers["retrieval"], open(os.path.join(work, "answers-retrieval.json"), "w"))
    if {"sleep", "consolidated", "weights"} & set(arms):
        slept = os.path.join(work, "slept")
        if not os.path.exists(os.path.join(slept, "adapters", "current")) and not os.path.exists(
                os.path.join(slept, ".slept")):
            shutil.rmtree(slept, ignore_errors=True)
            shutil.copytree(told, slept)
            txt, res["seconds"]["deep_sleep"] = agent.run(slept, ["/sleep", "/wait", "/status"], "sleep")
            open(os.path.join(slept, ".slept"), "w").write(txt[-4000:])
        for m in glob_manifests(slept):
            res["train"] = {k: m.get(k) for k in ("id", "tier", "n_facts", "n_passed", "recall", "ppl_ratio",
                                                  "seconds")}
        for arm, prefix in (("consolidated", ""), ("weights", "/weights ")):
            if arm not in arms:
                continue
            da = os.path.join(work, "arm-" + arm)
            shutil.rmtree(da, ignore_errors=True)
            shutil.copytree(slept, da)
            txt, res["seconds"][arm] = agent.run(da, [prefix + q["q"] for q in qs], "arm-" + arm)
            answers[arm] = {inp[len(prefix):]: answer_text(out) for inp, out in blocks(txt)}
            json.dump(answers[arm], open(os.path.join(work, f"answers-{arm}.json"), "w"))

    # answers from earlier stages (--keep): each arm can run in its own call
    for arm in ("retrieval", "consolidated", "weights"):
        path = os.path.join(work, f"answers-{arm}.json")
        if arm not in answers and os.path.exists(path):
            answers[arm] = json.load(open(path))
    secs_path = os.path.join(work, "seconds.json")
    if os.path.exists(secs_path):
        res["seconds"] = {**json.load(open(secs_path)), **res["seconds"]}
    json.dump(res["seconds"], open(secs_path, "w"))

    # 4. scores per arm, question type, and retrieval coverage
    rows, table = [], {}
    for q in qs:
        row = {"type": q["type"], "q": q["q"], "accept": q["accept"], "covered": covered(q),
               "recalled": recalled.get(q["q"], "")}
        for arm, ans in answers.items():
            got = ans.get(q["q"], "")
            ok = any(has(got, v) for v in q["accept"])
            if q["type"] == "verify":       # a yes/no question is wrong when it says both
                ok = ok and not has(got, "no" if q["accept"] == ["yes"] else "yes")
            row[arm] = got
            row[arm + "_ok"] = ok
            for key in (q["type"], q["type"] + (" / found" if row["covered"] else " / missed"), "all"):
                cell = table.setdefault(key, {}).setdefault(arm, [0, 0])
                cell[0] += ok
                cell[1] += 1
        rows.append(row)
    res["coverage"] = {t: round(sum(r["covered"] for r in rows if r["type"] == t) /
                                max(1, sum(1 for r in rows if r["type"] == t)), 3) for t in d["questions"]}
    res["accuracy"] = {k: {arm: f"{c[0]}/{c[1]} = {c[0] / c[1]:.3f}" for arm, c in v.items()}
                       for k, v in sorted(table.items())}
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    if a.out:
        res["answers"] = rows
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def glob_manifests(data):
    import glob
    out = []
    for man in sorted(glob.glob(os.path.join(data, "adapters", "*", "manifest.json"))):
        out.append(json.load(open(man)))
    return out


if __name__ == "__main__":
    main()
