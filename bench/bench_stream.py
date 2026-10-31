"""Continual-memory benchmark: the real memagent binary over several sessions.

A single batch of facts trained once (bench_memory, bench_agent) does not show
what an always-on agent faces: facts arrive over time, the fast LoRA is
retrained on top of a deep one, deep sleeps rebuild everything, and values get
corrected. This benchmark plays that out, scripted so it is reproducible:

  for each session of N facts:
      tell the facts (and, in later sessions, corrections of earlier ones)
      /nap, /wait              the fast LoRA learns the session's facts
      checkpoint               one held-out question per fact so far + unknown questions,
                               asked twice: through the full agent (retrieval + LoRAs + guard)
                               and from the weights alone (/weights)
      /sleep, /wait            a deep sleep rebuilds the deep LoRA from every fact
  final checkpoint             after the last deep sleep

Each checkpoint splits accuracy into facts from the latest session ("new") and
from earlier sessions ("old": retention), reports confabulation on the unknown
questions, and the run reports every training run's time and gate results and
the average turn latency.

  python3 bench/bench_stream.py --out bench/results/stream-tiny.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def has(ans, value):
    return f" {norm(value)} " in f" {norm(ans)} "


def turns(transcript):
    """[(input line, output text)] from a piped memagent transcript."""
    out, cur, buf = [], None, []
    for line in transcript.splitlines():
        if line.startswith("you> "):
            if cur is not None:
                out.append((cur, " ".join(buf).strip()))
            cur, buf = line[5:], []
        elif cur is not None:
            if line.startswith("pal> "):
                buf.append(line[5:])
            elif line.startswith("weights> "):
                buf.append(line[9:])
            elif buf and not line.startswith("["):
                buf.append(line)
    if cur is not None:
        out.append((cur, " ".join(buf).strip()))
    return out


# corrections told in a later session: (fact index, new "say", new value)
CORRECTIONS = [(0, "My dog's name is Max.", "Max"), (9, "My favorite color is green.", "green")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--facts", default=os.path.join(HERE, "facts.json"))
    ap.add_argument("--session", type=int, default=10, help="facts per session")
    ap.add_argument("--n", type=int, default=0, help="use only the first N facts")
    ap.add_argument("--unknown", type=int, default=6, help="unknown questions per checkpoint")
    ap.add_argument("--corrections", type=int, default=1, help="1: correct two facts in the last session")
    ap.add_argument("--ini", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="extra memagent.ini settings, e.g. consolidate.deep_after=8")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    data = json.load(open(a.facts, encoding="utf-8"))
    facts = data["facts"][:a.n] if a.n else data["facts"]
    unknown = [u["q"] if isinstance(u, dict) else u for u in data["unknown"]]
    sessions = [facts[i:i + a.session] for i in range(0, len(facts), a.session)]
    work = tempfile.mkdtemp(prefix="benchstream-")
    extra = {}
    for kv in a.ini:
        key, val = kv.split("=", 1)
        sec, k = key.split(".", 1)
        extra.setdefault(sec, {})[k] = val
    ini = {
        "agent": {"profile": a.profile, "engine": "llama", "models": f"{REPO}/models.json",
                  "priors": f"{REPO}/priors.md", "data_dir": f"{work}/data", "repo_dir": REPO,
                  "recent_turns": "2"},
        "gpu": {"infer": "-1", "train": "-1"},
        "consolidate": {"auto": "0", "trainer": "llama", "tiers": "on"},
        "retrieval": {"cache_dir": f"{REPO}/cache"},
    }
    for sec, kv in extra.items():
        ini.setdefault(sec, {}).update(kv)
    with open(os.path.join(work, "memagent.ini"), "w") as f:
        for sec, kv in ini.items():
            f.write(f"[{sec}]\n" + "".join(f"{k}={v}\n" for k, v in kv.items()))

    expected = {i: f["value"] for i, f in enumerate(facts)}
    script, plan = [], []          # plan: (checkpoint, kind, fact index or None, question)
    n_cp = 0
    for s, chunk in enumerate(sessions):
        base = s * a.session
        script += [f["say"] for f in chunk]
        if a.corrections and s == len(sessions) - 1 and s > 0:
            for idx, say, value in CORRECTIONS:
                if idx < base:
                    script.append(say)
                    expected[idx] = value
        script += ["/nap", "/wait"]

        def checkpoint(upto):
            nonlocal n_cp
            for i in range(upto):
                q = facts[i]["tests"][n_cp % len(facts[i]["tests"])]
                for mode in ("system", "weights"):
                    plan.append((n_cp, mode, i, q))
                    script.append(q if mode == "system" else "/weights " + q)
            for j in range(a.unknown):
                q = unknown[(n_cp * a.unknown + j) % len(unknown)]
                for mode in ("system", "weights"):
                    plan.append((n_cp, mode, None, q))
                    script.append(q if mode == "system" else "/weights " + q)
            n_cp += 1

        checkpoint(base + len(chunk))
        script += ["/sleep", "/wait"]
    checkpoint(len(facts))
    script += ["/status", "/quit"]

    env = dict(os.environ, MEMAGENT_TEMPERATURE="0", MEMAGENT_TIMING="1")
    tpath = os.path.join(work, "transcript.txt")
    print(f"[bench_stream] {len(facts)} facts in {len(sessions)} sessions, {n_cp} checkpoints; "
          f"transcript: {tpath}", flush=True)
    t0 = time.time()
    with open(tpath, "w") as out:
        subprocess.run([os.path.join(REPO, "build", "memagent"), "--config", os.path.join(work, "memagent.ini")],
                       input="\n".join(script) + "\n", stdout=out, stderr=subprocess.STDOUT, text=True, env=env)
    wall = time.time() - t0
    text = open(tpath, encoding="utf-8").read()

    # answers in script order: every question line produces one turn
    answers = {}
    qi = 0
    for inp, outp in turns(text):
        if qi < len(plan) and (inp == plan[qi][3] or inp == "/weights " + plan[qi][3]):
            answers[qi] = outp
            qi += 1
    values = list({v for v in expected.values()} | {f["value"] for f in facts})

    checkpoints = []
    for cp in range(n_cp):
        newest = min(cp, len(sessions) - 1)
        lo = newest * a.session if cp < len(sessions) else len(facts)   # final checkpoint: all "old"
        row = {"checkpoint": cp, "after": "deep sleep (final)" if cp >= len(sessions) else f"session {cp + 1}, fast run",
               "facts": 0}
        for mode in ("system", "weights"):
            ok = {"new": [0, 0], "old": [0, 0]}
            conf = [0, 0]
            for k, (c, m, i, q) in enumerate(plan):
                if c != cp or m != mode or k not in answers:
                    continue
                ans = answers[k]
                if i is None:
                    conf[0] += any(has(ans, v) for v in values)
                    conf[1] += 1
                else:
                    grp = "new" if i >= lo else "old"
                    # a correction counts from the checkpoint after the session that told it
                    want = expected[i] if cp >= len(sessions) - 1 else facts[i]["value"]
                    ok[grp][0] += has(ans, want)
                    ok[grp][1] += 1
                    row["facts"] = max(row["facts"], i + 1)
            row[mode] = {
                "new": round(ok["new"][0] / ok["new"][1], 3) if ok["new"][1] else None,
                "old": round(ok["old"][0] / ok["old"][1], 3) if ok["old"][1] else None,
                "all": round((ok["new"][0] + ok["old"][0]) / max(1, ok["new"][1] + ok["old"][1]), 3),
                "confabulation": round(conf[0] / conf[1], 3) if conf[1] else None,
            }
        checkpoints.append(row)

    runs = []
    for man in sorted(glob.glob(os.path.join(work, "data", "adapters", "*", "manifest.json"))):
        m = json.load(open(man))
        runs.append({k: m.get(k) for k in ("id", "tier", "n_facts", "n_passed", "replay", "replay_passed",
                                           "recall", "ppl_ratio", "seconds")})
    gate_rejects = len(re.findall(r"rejected by the eval gate", text))
    timing = [float(x) for x in re.findall(r"\[timing\].* total (\d+) ms", text)]
    res = {
        "label": a.label, "profile": a.profile, "n_facts": len(facts), "sessions": len(sessions),
        "settings": a.ini, "checkpoints": checkpoints, "runs": runs, "gate_rejects": gate_rejects,
        "train_seconds": round(sum(r["seconds"] or 0 for r in runs), 1),
        "turn_ms_avg": round(sum(timing) / len(timing)) if timing else None,
        "answered": f"{len(answers)}/{len(plan)}", "wall_seconds": round(wall, 1), "transcript": tpath,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    for r in runs:
        print(" ", r)
    if a.out:
        res["answers"] = [{"checkpoint": c, "mode": m, "fact": i, "q": q, "answer": answers.get(k)}
                          for k, (c, m, i, q) in enumerate(plan)]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
