"""The integration benchmark on any profile, in Python (transformers), for a GPU host.

bench_integration.py drives the real memagent binary (llama.cpp), which only exists
where it was built. This runs the same questions on a GPU box with the same facts in
the prompt: each question gets the facts the agent's own retrieval chose for it
(the "recalled" lists that bench_integration.py records), in the agent's prompt
layout (priors, then "Relevant memory:" with one "- fact" line each). What differs
from the agent: no tools block, no guard, and transformers instead of llama.cpp. So
compare arms and model sizes within this harness, not against bench_integration.py.

  retrieval   base model + retrieved facts
  sleep       base model + deep-sleep LoRA (trainer/consolidate.py) + retrieved facts
  weights     base model + that LoRA, no retrieved facts

  python3 bench/bench_integration_hf.py --profile large --recalled bench/results/integration-small-edited.json \\
      --adapter /workspace/ds-large/mem-... --out bench/results/integration-hf-large.json
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "trainer"))
sys.path.insert(0, HERE)
import memlib  # noqa: E402
from bench_integration import has, stored_text  # noqa: E402

FACT_LINE = re.compile(r"fact #\d+\s+cos [-\d.]+\s+rerank [-\d.e]+\s+(.*)$")


def recalled_facts(text, k):
    out = []
    for line in text.splitlines():
        m = FACT_LINE.search(line.strip())
        if m:
            out.append(m.group(1).strip())
    return out[:k]


def strip_thinking(ans):
    ans = re.sub(r"(?s)<think>.*?</think>|\[THINK\].*?\[/THINK\]", "", ans)
    ans = re.sub(r"(?s)^.*?(</think>|\[/THINK\])", "", ans)   # a cut-off opening tag
    return ans.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--models")
    ap.add_argument("--data", default=os.path.join(HERE, "facts_integration.json"))
    ap.add_argument("--recalled", required=True, help="a bench_integration.py results file (its answers[].recalled)")
    ap.add_argument("--rag-k", type=int, default=3)
    ap.add_argument("--adapter", help="PEFT adapter dir for the sleep and weights arms")
    ap.add_argument("--arms", default="retrieval,sleep,weights")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--priors", default=os.path.join(REPO, "priors.md"))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prof = memlib.load_profile(a.profile, a.models)
    d = json.load(open(a.data, encoding="utf-8"))
    qs = [q for t in d["questions"] for q in d["questions"][t]]
    rec = {r["q"]: r["recalled"] for r in json.load(open(a.recalled, encoding="utf-8"))["answers"]}
    priors = open(a.priors, encoding="utf-8").read().strip()
    arms = [x for x in a.arms.split(",") if x]
    t0 = time.time()
    tok, model, device = memlib.load_base(prof)
    res = {"label": a.label, "profile": prof["name"], "base": prof["base"], "harness": "transformers",
           "adapter": a.adapter, "rag_k": a.rag_k, "seconds": {"load": round(time.time() - t0, 1)}}

    def ask(q, facts):
        sys_ = priors + ("\n\nRelevant memory:\n" + "\n".join("- " + f for f in facts) if facts else "")
        ans, _, _ = memlib.generate(tok, model, [{"role": "system", "content": sys_}, {"role": "user", "content": q}],
                                    a.max_new)
        return strip_thinking(ans)

    answers = {}
    for arm in arms:
        if arm in ("sleep", "weights") and not hasattr(model, "peft_config"):
            if not a.adapter:
                raise SystemExit(f"arm {arm} needs --adapter")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
            model.eval()
        t = time.time()
        answers[arm] = {}
        for i, q in enumerate(qs):
            facts = [] if arm == "weights" else recalled_facts(rec.get(q["q"], ""), a.rag_k)
            answers[arm][q["q"]] = ask(q["q"], facts)
            print(f"[{arm} {i + 1}/{len(qs)}] {q['q']} -> {answers[arm][q['q']][:100]!r}", flush=True)
        res["seconds"][arm] = round(time.time() - t, 1)

    rows, table = [], {}
    for q in qs:
        got = rec.get(q["q"], "")
        cov = all(stored_text(n)[:-1] in got for n in q["needs"])
        row = {"type": q["type"], "q": q["q"], "accept": q["accept"], "covered": cov,
               "recalled": recalled_facts(got, a.rag_k)}
        for arm, ans in answers.items():
            g = ans[q["q"]]
            ok = any(has(g, v) for v in q["accept"])
            if q["type"] == "verify":
                ok = ok and not has(g, "no" if q["accept"] == ["yes"] else "yes")
            row[arm], row[arm + "_ok"] = g, ok
            for key in (q["type"], q["type"] + (" / found" if cov else " / missed"), "all"):
                c = table.setdefault(key, {}).setdefault(arm, [0, 0])
                c[0] += ok
                c[1] += 1
        rows.append(row)
    res["accuracy"] = {k: {arm: f"{c[0]}/{c[1]} = {c[0] / c[1]:.3f}" for arm, c in v.items()}
                       for k, v in sorted(table.items())}
    print(json.dumps({k: v for k, v in res.items() if k != "answers"}, indent=1))
    res["answers"] = rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
