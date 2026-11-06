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


def run_dpo(model, tok, facts, priors, tr, prof, seed, params):
    """Second stage: DPO for abstention (see memlib "DPO abstention")."""
    import torch
    import torch.nn.functional as F
    cfg = tr["dpo"]
    rng = random.Random(seed + 1)
    neg_train, neg_eval = memlib.split_negative_subjects(facts, rng)
    max_len = tr["max_len"]

    def gen(msgs):
        model.eval()
        model.config.use_cache = True
        ans, _, _ = memlib.generate(tok, model, msgs, 40)
        return ans

    n_unknown = max(8, round(len(facts) * cfg.get("unknown", 1.0)))
    n_known = max(8, round(len(facts) * cfg.get("known", 1.0)))
    pairs = memlib.dpo_pairs(gen, facts, priors, rng, n_unknown, n_known, cfg.get("near_frac", 0.5), neg_train)
    own = sum(1 for _, c, r in pairs if c == memlib.ABSTAIN and r not in [f["text"] for f in facts])
    log(f"[dpo] pairs={len(pairs)} (unknown {n_unknown}, known {n_known}; {own} rejected answers are the model's own)")
    chosen = [memlib.tokenize_example(tok, m + [{"role": "assistant", "content": c}], max_len) for m, c, _ in pairs]
    rejected = [memlib.tokenize_example(tok, m + [{"role": "assistant", "content": r}], max_len) for m, _, r in pairs]

    # reference = the model as SFT left it, scored once (no second copy in memory)
    model.eval()
    bs = cfg.get("batch", 4)
    ref_c, ref_r = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            ref_c.append(memlib.seq_logprob(model, tok, chosen[i:i + bs], tok.pad_token_id)[0])
            ref_r.append(memlib.seq_logprob(model, tok, rejected[i:i + bs], tok.pad_token_id)[0])
    ref_c, ref_r = torch.cat(ref_c), torch.cat(ref_r)

    beta, w = cfg.get("beta", 0.1), cfg.get("sft_weight", 0.2)
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 5e-5), weight_decay=0.0)
    model.config.use_cache = False
    order = list(range(len(pairs)))
    for ep in range(cfg.get("epochs", 2)):
        model.train()
        rng.shuffle(order)
        tot = acc = n = 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            pc, nc = memlib.seq_logprob(model, tok, [chosen[j] for j in idx], tok.pad_token_id)
            pr, _ = memlib.seq_logprob(model, tok, [rejected[j] for j in idx], tok.pad_token_id)
            margin = beta * ((pc - ref_c[idx]) - (pr - ref_r[idx]))
            loss = -F.logsigmoid(margin).mean() + w * (-(pc / nc.clamp(min=1))).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            tot += loss.item() * len(idx)
            acc += (margin > 0).sum().item()
            n += len(idx)
        log(f"[dpo] epoch {ep + 1} loss={tot / n:.4f} pair-accuracy={acc / n:.2f}")

    # abstention on subjects DPO never saw, and refusals of known facts
    abst = sum(memlib.is_abstain(gen(([{"role": "system", "content": priors}] if priors else []) +
                                     [{"role": "user", "content": rng.choice(memlib.subject_questions(s))}]))
               for s in neg_eval)
    stats = {"pairs": len(pairs), "own_rejected": own, "final_loss": round(tot / n, 4),
             "pair_accuracy": round(acc / n, 3), "abstain_unseen": round(abst / max(1, len(neg_eval)), 3)}
    log(f"[dpo] abstains on {abst}/{len(neg_eval)} unseen unknown subjects")
    return stats


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
    ap.add_argument("--gguf", metavar="LLAMA_SRC",
                    help="also write adapter.gguf for the llama.cpp engine, with the converter in this checkout")
    ap.add_argument("--no-gate", action="store_true", help="save the adapter even if the eval gate fails (benchmarks)")
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
    # the last 8 prompts measure drift whatever the number of training anchors (Mem.Train)
    n_held = 8
    anchors = self_answers(memlib.ANCHOR_PROMPTS[:min(n_anc, len(memlib.ANCHOR_PROMPTS) - n_held)])
    heldout = self_answers(memlib.ANCHOR_PROMPTS[-n_held:])
    base_heldout_loss = memlib.mean_loss(model, tok, heldout, tr["max_len"])

    extra = memlib.name_questions if tr.get("name_questions") else None
    n_para = tr["paraphrases"] + (3 if extra else 0)
    rng = random.Random(a.seed)
    if tr.get("examples") == "varied":
        train_chats, probes = memlib.varied_examples(facts, n_para, priors, rng)
    else:
        train_chats, probes = memlib.fact_examples(facts, n_para, priors, extra)
    k = max(8, round(len(facts) * tr.get("negatives", 0))) if tr.get("negatives") else 0
    negatives = memlib.negative_examples(facts, k, priors, rng) if k else []
    k = max(8, round(len(facts) * tr.get("raft", 0))) if tr.get("raft") else 0
    raft = memlib.raft_examples(k, priors, rng) if k else []
    # anchors are a fixed set; with many facts they shrink to a few percent of
    # the data and drift grows (tiny: 1.2 at 3 facts, 1.8 at 30). anchor_ratio
    # replays them to keep a constant share.
    replay = list(anchors)
    ratio = tr.get("anchor_ratio")
    if ratio and anchors:
        n_other = len(train_chats) + len(negatives) + len(raft)
        want = round(ratio * n_other / (1 - ratio))
        replay = [anchors[i % len(anchors)] for i in range(max(len(anchors), want))]
    data = [memlib.tokenize_example(tok, m, tr["max_len"]) for m in train_chats + negatives + raft + replay]
    log(f"[consolidate] examples={len(data)} (facts {len(train_chats)}, negatives {len(negatives)}, "
        f"raft {len(raft)}, anchors {len(replay)})")

    lc = prof["lora"]
    model.config.use_cache = False
    if tr.get("grad_checkpointing"):
        # recompute activations in the backward pass: ~30% slower, a fraction of
        # the activation memory (a 1.5B model at batch 8 otherwise needs >14 GB on CPU)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
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

    dpo_stats = run_dpo(model, tok, facts, priors, tr, prof, a.seed, params) if tr.get("dpo") else None

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

    refusals = sum(1 for r in results if memlib.is_abstain(r["answer"]))
    metrics = {"recall": round(recall, 4), "ppl_ratio": round(ppl_ratio, 4),
               "refused_known": round(refusals / max(1, len(results)), 4),
               "n_facts": len(facts), "n_passed": len(passed), "dpo": dpo_stats,
               "seconds": round(time.time() - t0, 1), "device": device}
    gate_ok = recall >= gate["min_recall"] and ppl_ratio <= gate["max_ppl_ratio"]
    metrics["gate_ok"] = gate_ok
    if not gate_ok and not a.no_gate:
        return summary(2, reason="eval gate", **metrics)

    adapter_id = "mem-" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = os.path.join(a.out, adapter_id)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if a.gguf:
        # before ready.json: the in-process engine loads adapter.gguf on the swap
        import gguf_export
        _, metrics["gguf_seconds"] = gguf_export.export_adapter(path, prof, a.gguf)
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
