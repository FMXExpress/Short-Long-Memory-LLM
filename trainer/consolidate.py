"""Consolidation job: bake the active facts in memory.db into a fresh LoRA.

Run by memagent as a one-shot process on the training GPU:

    CUDA_VISIBLE_DEVICES=1 python trainer/consolidate.py --db memory.db --out adapters

Exit codes: 0 promoted, 2 rejected by the eval gate, 3 nothing to train, 1 error.
The last stdout line is always a JSON summary.
"""
import argparse
import datetime as dt
import json
import os
import random
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib  # noqa: E402
from memlib import log  # noqa: E402


def read_facts(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, text, subject, value, questions FROM facts "
        "WHERE superseded = 0 AND forgotten = 0 ORDER BY id").fetchall()
    con.close()
    facts = []
    for r in rows:
        qs = json.loads(r["questions"] or "[]") or memlib.subject_questions(r["subject"])
        facts.append({"id": r["id"], "text": r["text"], "subject": r["subject"],
                      "value": r["value"], "questions": qs})
    return facts


def summary(code, **kw):
    print(json.dumps({"exit": code, **kw}), flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--models")
    ap.add_argument("--priors", help="system prompt file, so training matches inference")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    prof = memlib.load_profile(a.profile, a.models)
    tr, gate = prof["train"], prof["gate"]
    priors = open(a.priors, encoding="utf-8").read().strip() if a.priors else ""

    facts = read_facts(a.db)
    if not facts:
        return summary(3, reason="no active facts")
    log(f"[consolidate] profile={prof['name']} base={prof['base']} facts={len(facts)}")

    tok, model, device = memlib.load_base(prof, for_training=True)
    log(f"[consolidate] device={device}")

    # Anchors: the base model's own answers to generic prompts. Training on
    # them alongside the facts keeps the adapter from drifting the priors.
    # A disjoint held-out set measures drift, since the trained-on anchors
    # would always look better after training.
    def self_answers(prompts):
        out = []
        for q in prompts:
            msgs = ([{"role": "system", "content": priors}] if priors else []) + [{"role": "user", "content": q}]
            ans, _, _ = memlib.generate(tok, model, msgs, 48)
            out.append(msgs + [{"role": "assistant", "content": ans}])
        return out
    n_anc = tr["anchors"]
    anchors = self_answers(memlib.ANCHOR_PROMPTS[:n_anc])
    heldout = self_answers(memlib.ANCHOR_PROMPTS[n_anc:n_anc + max(4, n_anc // 2)])
    base_heldout_loss = memlib.mean_loss(model, tok, heldout, tr["max_len"])

    train_chats, probes = memlib.fact_examples(facts, tr["paraphrases"], priors)
    data = [memlib.tokenize_example(tok, m, tr["max_len"]) for m in train_chats + anchors]
    log(f"[consolidate] examples={len(data)} (facts {len(train_chats)}, anchors {len(anchors)})")

    lc = prof["lora"]
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["targets"], bias="none", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=0.0)
    steps = tr["epochs"] * ((len(data) + tr["batch"] - 1) // tr["batch"]) // tr["grad_accum"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / max(1, steps)))

    step = 0
    for ep in range(tr["epochs"]):
        model.train()
        random.shuffle(data)
        tot, n = 0.0, 0
        for i, (ids, att, lab) in enumerate(memlib.batches(data, tr["batch"], tok.pad_token_id)):
            out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                        labels=lab.to(model.device))
            (out.loss / tr["grad_accum"]).backward()
            if (i + 1) % tr["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            tot += out.loss.item()
            n += 1
        log(f"[consolidate] epoch {ep + 1}/{tr['epochs']} loss={tot / max(n, 1):.4f}")

    # Eval gate: every active fact is probed with a held-out question and no
    # memory context, so a pass means the knowledge is in the weights.
    model.eval()
    model.config.use_cache = True
    passed, results = [], []
    for p in probes:
        msgs = ([{"role": "system", "content": priors}] if priors else []) + \
               [{"role": "user", "content": p["question"]}]
        ans, _, _ = memlib.generate(tok, model, msgs, 48)
        ok = memlib.answer_matches(ans, p["value"])
        results.append({"id": p["id"], "question": p["question"], "answer": ans, "ok": ok})
        if ok:
            passed.append(p["id"])
        log(f"[probe] {'PASS' if ok else 'FAIL'} #{p['id']} {p['question']!r} -> {ans!r}")
    recall = len(passed) / len(probes)
    heldout_loss = memlib.mean_loss(model, tok, heldout, tr["max_len"])
    ppl_ratio = memlib.ppl(heldout_loss) / memlib.ppl(base_heldout_loss)
    log(f"[consolidate] recall={recall:.2f} heldout_ppl_ratio={ppl_ratio:.3f}")

    metrics = {"recall": round(recall, 4), "ppl_ratio": round(ppl_ratio, 4),
               "n_facts": len(facts), "n_passed": len(passed),
               "seconds": round(time.time() - t0, 1), "device": device}
    if recall < gate["min_recall"] or ppl_ratio > gate["max_ppl_ratio"]:
        return summary(2, reason="eval gate", **metrics)

    adapter_id = "mem-" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = os.path.join(a.out, adapter_id)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    manifest = {"id": adapter_id, "path": os.path.abspath(path), "profile": prof["name"],
                "base": prof["base"], "created": dt.datetime.now().isoformat(timespec="seconds"),
                "fact_ids": [f["id"] for f in facts], "passed_fact_ids": passed,
                "probes": results, **metrics}
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp = os.path.join(a.out, "ready.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"id": adapter_id, "path": os.path.abspath(path)}, f)
    os.replace(tmp, os.path.join(a.out, "ready.json"))
    return summary(0, id=adapter_id, path=os.path.abspath(path), passed_fact_ids=passed, **metrics)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # always leave a parseable last line for memagent
        import traceback
        traceback.print_exc()
        sys.exit(summary(1, reason=f"{type(e).__name__}: {e}"))
