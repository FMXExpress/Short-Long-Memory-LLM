"""System benchmark: the real memagent binary, end to end.

Tells the agent every fact the way a user would ("say"), then asks the
held-out questions and the unknown questions twice: before consolidation
(short-term memory only: extraction + RAG) and after one /sleep (LoRA + RAG).
Everything is the production path: rule/LLM extraction, hybrid retrieval with
the reranker, the agent's prompt, the trainer and the hot swap.

  python3 bench/bench_agent.py --profile tiny --out bench/results/agent-tiny.json
  python3 bench/bench_agent.py --profile tiny --engine llama   (in-process llama.cpp)
"""
import argparse
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


def replies(transcript):
    """{question: reply} from a memagent transcript (piped mode echoes 'you> ...')."""
    out, cur, buf = [], None, []
    for line in transcript.splitlines():
        if line.startswith("you> "):
            if cur is not None:
                out.append((cur, " ".join(buf).strip()))
            cur, buf = line[5:], []
        elif line.startswith("pal> "):
            buf.append(line[5:])
        elif cur is not None and buf and not line.startswith("["):
            buf.append(line)
    if cur is not None:
        out.append((cur, " ".join(buf).strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--facts", default=os.path.join(HERE, "facts.json"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--engine", default="python", choices=["python", "llama"])
    a = ap.parse_args()

    data = json.load(open(a.facts, encoding="utf-8"))
    facts = data["facts"][:a.n] if a.n else data["facts"]
    unknown = [u["q"] if isinstance(u, dict) else u for u in data["unknown"]]
    values = [f["value"] for f in facts]
    work = tempfile.mkdtemp(prefix="benchagent-")
    ini = os.path.join(work, "memagent.ini")
    with open(ini, "w") as f:
        f.write(f"""[agent]
profile={a.profile}
engine={a.engine}
models={REPO}/models.json
priors={REPO}/priors.md
data_dir={work}/data
repo_dir={REPO}
recent_turns=2
[gpu]
infer=-1
train=-1
[consolidate]
auto=0
[retrieval]
cache_dir={REPO}/cache
""")
    questions = [(q, f["value"]) for f in facts for q in f["tests"]]
    script = [f["say"] for f in facts] + ["/facts"]
    script += [q for q, _ in questions] + unknown
    script += ["/sleep", "/wait"]
    script += [q for q, _ in questions] + unknown + ["/status", "/quit"]
    env = dict(os.environ, MEMAGENT_TEMPERATURE="0", MEMAGENT_TIMING="1")
    t = time.time()
    tpath = os.path.join(work, "transcript.txt")
    print(f"[bench_agent] live transcript: {tpath}", flush=True)
    # stream to a file, so a slow or stuck run can be inspected while it runs
    with open(tpath, "w") as out:
        subprocess.run([os.path.join(REPO, "build", "memagent"), "--config", ini],
                       input="\n".join(script) + "\n", stdout=out, stderr=subprocess.STDOUT, text=True, env=env)
    wall = time.time() - t

    class _P:
        stdout = open(tpath).read()
    p = _P()

    rs = replies(p.stdout)
    by_turn = [r for r in rs if not r[0].startswith("/")]
    n_say = len(facts)
    qa = by_turn[n_say:]
    phase_len = len(questions) + len(unknown)
    phases = {"before_sleep": qa[:phase_len], "after_sleep": qa[phase_len:2 * phase_len]}
    stored = len(re.findall(r"^\[memory [+~]#", p.stdout, re.M))
    res = {"label": a.label, "profile": a.profile, "engine": a.engine, "n_facts": len(facts), "facts_stored": stored,
           "wall_seconds": round(wall, 1)}
    for name, rows in phases.items():
        known_rows, unk_rows = rows[:len(questions)], rows[len(questions):]
        acc = sum(has(ans, v) for (_, ans), (_, v) in zip(known_rows, questions)) / max(1, len(questions))
        conf = sum(any(has(ans, v) for v in values) for _, ans in unk_rows) / max(1, len(unknown))
        res[name] = {"accuracy": round(acc, 3), "confabulation": round(conf, 3), "n": len(rows)}
    m = re.search(r"hot-swapped LoRA .*?(\d+)/(\d+) facts now in the weights \(recall ([\d.]+), held-out ppl x([\d.]+)\)",
                  p.stdout)
    res["consolidation"] = {"baked": m.group(1), "facts": m.group(2), "recall": m.group(3),
                            "drift": m.group(4)} if m else "no hot swap in transcript"
    m = re.search(r"^timing:.*$", p.stdout, re.M)
    res["timing"] = m.group(0) if m else None
    res["transcript"] = os.path.join(work, "transcript.txt")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
