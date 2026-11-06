"""Write facts into the weights by editing, not training (MEMIT), and save the edit as a LoRA.

Deep sleep trains a LoRA on every fact for hours, and on the 1.5B profile it could
not learn 112 facts within the drift gate (bench/README.md, integration experiment).
This writes the same facts in one closed-form step instead:

  1. For each subject (a person, or "my sister"), find the change z - h to the hidden
     state after layer L, at the subject's last token, that makes the model say its
     facts (a few dozen Adam steps on one vector; the model stays frozen).
  2. Spread that change over the MLP output matrices of layers l1..L. Each layer solves

        dW = R K^T (lam C + K K^T)^-1

     where K are the subjects' keys (the down_proj inputs at the subject token), R the
     part of the change this layer takes, and C = E[k k^T] over general text: the C
     term keeps the outputs for every other key where they were, which is what keeps
     drift down.
  3. dW has rank <= number of subjects, so it factors exactly into LoRA B @ A on
     down_proj: the agent loads it like any adapter (convert_lora_to_gguf.py).

Meng et al. 2022 (ROME), 2023 (MEMIT). All facts about one subject share one key,
so a subject's facts are written together, in one target.

  python3 trainer/edit_facts.py --data DATA_DIR --profile small --activate
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib  # noqa: E402
from consolidate import read_facts  # noqa: E402

DEV = "cpu"   # set in main: the model's device

PREFIXES = ["", "Okay. ", "By the way, ", "Note: ", "I see. "]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- facts -> subjects

def split_subject(subject):
    """"Emil's city" -> ("Emil", "city"); "my son's name" -> ("my son", "name")."""
    m = re.match(r"^(.+?)'s (.+)$", subject)
    return (m.group(1), m.group(2)) if m else (subject, "")


def statement_prompts(f):
    """The fact as a sentence to complete: ("Emil's city is", " Riga")."""
    text = f["text"].rstrip(" .")
    i = text.lower().rfind(f["value"].lower())
    if i <= 0:
        return []
    stem = text[:i].rstrip()
    out = [stem]
    if stem.startswith("Your "):
        out.append("My " + stem[5:])
    return [(s, " " + text[i:]) for s in out]


def chat_text(tok, q, a=None):
    msgs = [{"role": "user", "content": q}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s, (a if a is not None else None)


class Item:
    """One prompt: ids, which positions are target tokens, and the subject's last token."""

    def __init__(self, ids, labels, pos):
        self.ids, self.labels, self.pos = ids, labels, pos


def subject_pos(tok, text, region_start, core):
    """Index of the token holding the last character of the first `core` word after region_start."""
    m = re.compile(r"\b" + re.escape(core) + r"\b", re.I).search(text, region_start)
    if not m:
        return None
    end = m.end() - 1
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= end < b:
            return enc["input_ids"], i
    return None


def make_item(tok, prompt, target, core, region_start=0):
    r = subject_pos(tok, prompt, region_start, core)
    if r is None:
        return None
    ids, pos = r
    tgt = tok(target, add_special_tokens=False)["input_ids"] if target else []
    return Item(ids + tgt, [-100] * len(ids) + tgt, pos)


def subjects_from_facts(tok, facts, n_questions):
    """{subject: {"core": word, "facts": [...], "items": [Item], "essence": Item}}"""
    subs = {}
    for f in facts:
        who, _ = split_subject(f["subject"])
        key = who.lower()
        core = who.split()[-1]
        s = subs.setdefault(key, {"name": who, "core": core, "facts": [], "items": []})
        s["facts"].append(f)
        for p in PREFIXES:
            for stem, tgt in statement_prompts(f):
                it = make_item(tok, p + stem, tgt, core, len(p))
                if it:
                    s["items"].append(it)
        # chat questions the agent would ask itself; never the last one, which is the probe
        qs = list(dict.fromkeys(f["questions"]))[:-1][:n_questions]
        for q in qs:
            text, _ = chat_text(tok, q)
            it = make_item(tok, text, f["text"], core, text.rfind(q))
            if it:
                s["items"].append(it)
    for s in subs.values():
        lead = s["name"][0].upper() + s["name"][1:]
        s["essence"] = make_item(tok, f"{lead} is", None, s["core"])
    return subs


# ---------------------------------------------------------------- model plumbing

def pad(items, pad_id):
    import torch
    n = max(len(it.ids) for it in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    lab = torch.full((len(items), n), -100, dtype=torch.long)
    att = torch.zeros((len(items), n), dtype=torch.long)
    for j, it in enumerate(items):
        ids[j, :len(it.ids)] = torch.tensor(it.ids)
        lab[j, :len(it.labels)] = torch.tensor(it.labels)
        att[j, :len(it.ids)] = 1
    pos = torch.tensor([it.pos for it in items])
    return ids.to(DEV), lab.to(DEV), att.to(DEV), pos.to(DEV)


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _with_hidden(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class AddAt:
    """Adds delta to the output of a decoder layer at given (row, position) pairs."""

    def __init__(self, layer):
        self.layer, self.delta, self.rows, self.pos, self.h = layer, None, None, None, None
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        h = _hidden(out)
        self.h = h[self.rows, self.pos].detach() if self.rows is not None else None
        if self.delta is None:
            return out
        h = h.clone()
        h[self.rows, self.pos] = h[self.rows, self.pos] + self.delta
        return _with_hidden(out, h)

    def remove(self):
        self.handle.remove()


def compute_target(model, tok, layer, s, steps, lr, kl_w, wd, clamp):
    """z for one subject: the hidden state after `layer` at the subject token that makes
    every item say its target. Returns (z, h at start, final nll, steps used)."""
    import torch
    import torch.nn.functional as F
    items = s["items"]
    ess = s["essence"]
    ids, lab, att, pos = pad(items + [ess], tok.pad_token_id)
    rows = torch.arange(len(items) + 1, device=DEV)
    hook = AddAt(model.model.layers[layer])
    hook.rows, hook.pos = rows, pos
    # logits only where they are scored: every position over a 152k vocabulary is GBs
    tgt = lab[:, 1:]
    rows_t, cols_t = (tgt != -100).nonzero(as_tuple=True)
    tgt_ids = tgt[rows_t, cols_t]
    e_row, e_last = len(items), len(ess.ids) - 1

    def run():
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        lp = torch.log_softmax(model.lm_head(h[rows_t, cols_t]).float(), -1)
        ess_lp = F.log_softmax(model.lm_head(h[e_row, e_last]).float(), -1)
        return lp, ess_lp
    with torch.no_grad():
        _, ess_ref = run()
        h0 = hook.h.mean(0)
    delta = torch.zeros(h0.shape, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    max_norm = clamp * h0.norm()
    nll = float("nan")
    step = 0
    for step in range(steps):
        hook.delta = delta
        lp, ess_lp = run()
        tok_nll = -lp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        per = torch.zeros(len(items), device=DEV).index_add(0, rows_t, tok_nll) / \
            torch.zeros(len(items), device=DEV).index_add(0, rows_t, torch.ones_like(tok_nll)).clamp(min=1)
        nll_t = per.mean()
        kl = F.kl_div(ess_lp, ess_ref, log_target=True, reduction="sum")
        loss = nll_t + kl_w * kl + wd * delta.norm() ** 2 / h0.norm() ** 2
        nll = float(nll_t.detach())
        if nll < 0.03:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())
    hook.remove()
    return (h0 + delta.detach()), h0, nll, step + 1


def subject_keys(model, tok, layers, out_layer, subs):
    """Per subject: the down_proj input (key) at each edited layer and the hidden state
    after out_layer, both at the subject token and averaged over its items."""
    import torch
    keys = {l: [] for l in layers}
    cur = []
    grab = {}

    def pre(l):
        def f(mod, args):
            grab[l] = args[0]
        return f
    hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre(l)) for l in layers]
    out_hook = AddAt(model.model.layers[out_layer])
    with torch.no_grad():
        for s in subs:
            ids, _, att, pos = pad(s["items"], tok.pad_token_id)
            rows = torch.arange(len(s["items"]), device=DEV)
            out_hook.rows, out_hook.pos = rows, pos
            model(input_ids=ids, attention_mask=att)
            for l in layers:
                keys[l].append(grab[l][rows, pos].mean(0))
            cur.append(out_hook.h.mean(0))
    for h in hs:
        h.remove()
    out_hook.remove()
    return {l: torch.stack(v) for l, v in keys.items()}, torch.stack(cur)


def cov_path(cache_dir, profile, l):
    return os.path.join(cache_dir, f"{profile}-L{l}.pt")


def covariance(model, tok, layers, n_tokens, cache_dir, profile, load=True):
    """C_l = E[k k^T] of the down_proj inputs over general text (wikitext-103), cached.
    load=False only makes sure the cache exists (each C is 320 MB at 1.5B)."""
    import torch
    paths = {l: cov_path(cache_dir, profile, l) for l in layers}
    if all(os.path.exists(p) for p in paths.values()):
        return {l: torch.load(p, map_location=DEV) for l, p in paths.items()} if load else None
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    texts = [x["text"] for x in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                             split="train[:40000]") if len(x["text"]) > 400]
    acc = {l: torch.zeros(model.config.intermediate_size, model.config.intermediate_size, device=DEV) for l in layers}
    count = 0

    class Stop(Exception):
        pass

    def pre(l):
        def f(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            acc[l].addmm_(x.T, x)
            if l == max(layers):
                raise Stop
        return f
    hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre(l)) for l in layers]
    t = time.time()
    with torch.no_grad():
        i = 0
        while count < n_tokens and i < len(texts):
            batch = texts[i:i + 8]
            i += 8
            for text in batch:   # one at a time: no padding tokens in the statistics
                ids = tok(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(DEV)
                try:
                    model(input_ids=ids)
                except Stop:
                    pass
                count += ids.shape[1]
            log(f"[edit] covariance {count}/{n_tokens} tokens ({time.time() - t:.0f} s)")
    for h in hs:
        h.remove()
    out = {}
    for l in layers:
        out[l] = acc[l] / count
        torch.save(out[l].cpu(), paths[l])
        del acc[l]
    return out if load else None


# ---------------------------------------------------------------- export

def save_lora(path, deltas, base, rank=None):
    """dW per layer -> PEFT adapter (B @ A on down_proj, alpha = r so the scale is 1)."""
    import torch
    from safetensors.torch import save_file
    r = rank or max(int((torch.linalg.svdvals(d) > 1e-6 * torch.linalg.svdvals(d)[0]).sum()) for d in deltas.values())
    tensors = {}
    for l, d in deltas.items():
        d = d.float().cpu()
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        sq = S[:r].sqrt()
        pre = f"base_model.model.model.layers.{l}.mlp.down_proj"
        tensors[pre + ".lora_A.weight"] = (sq[:, None] * Vh[:r]).contiguous()
        tensors[pre + ".lora_B.weight"] = (U[:, :r] * sq[None, :]).contiguous()
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base, "r": r,
               "lora_alpha": r, "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none",
               "fan_in_fan_out": False, "layers_to_transform": sorted(deltas)},
              open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    return r


def to_gguf(path, base):
    from gguf_export import _converter, _snapshot
    out = os.path.join(path, "adapter.gguf")
    p = subprocess.run([sys.executable, _converter(None, "convert_lora_to_gguf.py"), path,
                        "--base", _snapshot(base, ["*.json"]), "--outtype", "f32", "--outfile", out + ".tmp"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:])
    os.replace(out + ".tmp", out)
    return out


def activate(data, adapter_id, path, manifest):
    """What the agent does when it promotes a deep LoRA (TAgent.PromoteManifest)."""
    con = sqlite3.connect(os.path.join(data, "memory.db"))
    now = int(time.time())
    con.execute("UPDATE adapters SET status='retired' WHERE status='active'")
    con.execute("INSERT OR REPLACE INTO adapters (id, path, created_at, n_facts, n_passed, recall, ppl_ratio, status) "
                "VALUES (?,?,?,?,?,?,?, 'active')", (adapter_id, path, now, manifest["n_facts"],
                                                     manifest["n_passed"], manifest["recall"], manifest["ppl_ratio"]))
    con.execute("UPDATE facts SET baked_in=''")
    con.executemany("UPDATE facts SET baked_in=? WHERE id=?", [(adapter_id, i) for i in manifest["passed_fact_ids"]])
    for k, v in (("active_adapter", adapter_id), ("active_adapter_path", path), ("fast_adapter", ""),
                 ("fast_adapter_path", ""), ("last_consolidation", str(now)), ("forget_pending", "0")):
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (k, v))
    con.commit()
    con.close()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="agent data dir: reads memory.db, writes adapters/edit-*")
    ap.add_argument("--profile")
    ap.add_argument("--models")
    ap.add_argument("--priors", default=os.path.join(memlib.REPO_DIR, "priors.md"))
    ap.add_argument("--layers", default="4,5,6,7,8")
    ap.add_argument("--lam", type=float, default=15000.0, help="weight of C: higher = less drift, weaker edits")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--kl", type=float, default=0.0625)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=0.75, help="max |z - h| / |h|")
    ap.add_argument("--questions", type=int, default=3, help="chat questions per fact in the target")
    ap.add_argument("--cov-tokens", type=int, default=100000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects (a quick check)")
    ap.add_argument("--activate", action="store_true", help="make the edit the data dir's active deep LoRA")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-gguf", action="store_true", help="PEFT adapter only (no llama.cpp converter here)")
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    t0 = time.time()
    phases = {}
    prof = memlib.load_profile(a.profile, a.models)
    layers = [int(x) for x in a.layers.split(",")]
    out_layer = max(layers)
    priors = open(a.priors, encoding="utf-8").read().strip() if a.priors else ""
    facts = read_facts(os.path.join(a.data, "memory.db"))
    global DEV
    # float32 everywhere: the solve divides by small eigenvalues of C
    tok, model, _ = memlib.load_base({**prof, "dtype": "float32"})
    DEV = model.device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    subs = list(subjects_from_facts(tok, facts, a.questions).values())
    if a.limit:
        subs = subs[:a.limit]
        keep = {f["id"] for s in subs for f in s["facts"]}
        facts = [f for f in facts if f["id"] in keep]
    log(f"[edit] profile={prof['name']} facts={len(facts)} subjects={len(subs)} layers={layers} "
        f"items={sum(len(s['items']) for s in subs)}")
    phases["load"] = round(time.time() - t0, 1)

    def self_answers(prompts):
        out = []
        for q in prompts:
            msgs = ([{"role": "system", "content": priors}] if priors else []) + [{"role": "user", "content": q}]
            ans, _, _ = memlib.generate(tok, model, msgs, 48)
            out.append(msgs + [{"role": "assistant", "content": ans}])
        return out
    t = time.time()
    heldout = self_answers(memlib.ANCHOR_PROMPTS[-8:])
    base_loss = memlib.mean_loss(model, tok, heldout, 256)
    phases["anchors"] = round(time.time() - t, 1)

    t = time.time()
    covariance(model, tok, layers, a.cov_tokens, os.path.join(memlib.REPO_DIR, "cache", "edit-cov"), prof["name"],
               load=False)
    phases["covariance"] = round(time.time() - t, 1)

    # 1. targets
    t = time.time()
    with torch.enable_grad():
        Z = []
        for i, s in enumerate(subs):
            z, h0, nll, n = compute_target(model, tok, out_layer, s, a.steps, a.lr, a.kl, a.wd, a.clamp)
            Z.append(z)
            log(f"[edit] target {i + 1}/{len(subs)} {s['name']!r}: {len(s['facts'])} facts, nll {nll:.3f} "
                f"after {n} steps, |dz|/|h| {float((z - h0).norm() / h0.norm()):.2f}")
    Z = torch.stack(Z)
    phases["targets"] = round(time.time() - t, 1)

    # 2. spread over the layers
    t = time.time()
    orig = {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in layers}
    for i, l in enumerate(layers):
        K, cur = subject_keys(model, tok, layers, out_layer, subs)
        R = (Z - cur) / (len(layers) - i)                      # n x d
        Kl = K[l].T                                            # m x n
        C = torch.load(cov_path(os.path.join(memlib.REPO_DIR, "cache", "edit-cov"), prof["name"], l), map_location=DEV)
        A = a.lam * C + Kl @ Kl.T
        del C
        X = torch.linalg.solve(A, Kl)                          # m x n
        dW = R.T @ X.T                                         # d x m
        model.model.layers[l].mlp.down_proj.weight.add_(dW)
        log(f"[edit] layer {l}: |dW| {float(dW.norm()):.3f} vs |W| {float(orig[l].norm()):.1f}; "
            f"residual left {float((Z - cur).norm(dim=1).mean()):.2f}")
    phases["solve"] = round(time.time() - t, 1)

    # 3. the gate's numbers: a held-out question per fact, no memory context; drift
    t = time.time()
    passed, results = [], []
    values = {f["value"].lower() for f in facts}
    for f in facts:
        q = list(dict.fromkeys(f["questions"]))[-1]
        msgs = ([{"role": "system", "content": priors}] if priors else []) + [{"role": "user", "content": q}]
        ans, _, _ = memlib.generate(tok, model, msgs, 48)
        ok = memlib.answer_matches(ans, f["value"])
        wrong = [v for v in values if v != f["value"].lower() and memlib.answer_matches(ans, v)]
        results.append({"id": f["id"], "question": q, "answer": ans, "ok": ok, "other_value": wrong})
        if ok:
            passed.append(f["id"])
        log(f"[probe] {'PASS' if ok else 'FAIL'} #{f['id']} {q!r} -> {ans!r}")
    recall = len(passed) / max(1, len(facts))
    ppl_ratio = memlib.ppl(memlib.mean_loss(model, tok, heldout, 256)) / memlib.ppl(base_loss)
    phases["eval"] = round(time.time() - t, 1)
    binding = sum(1 for r in results if not r["ok"] and r["other_value"])
    log(f"[edit] recall={recall:.3f} heldout_ppl_ratio={ppl_ratio:.3f} binding_errors={binding}")

    # 4. the LoRA
    t = time.time()
    adapter_id = "edit-" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = os.path.abspath(os.path.join(a.data, "adapters", adapter_id))
    deltas = {l: (model.model.layers[l].mlp.down_proj.weight.detach() - orig[l]) for l in layers}
    rank = save_lora(path, deltas, prof["base"], rank=len(subs))   # each layer's dW has rank <= subjects
    if not a.no_gguf:
        to_gguf(path, prof["base"])
    phases["export"] = round(time.time() - t, 1)
    manifest = {"id": adapter_id, "path": path, "profile": prof["name"], "base": prof["base"], "tier": "deep",
                "trainer": "edit", "created": dt.datetime.now().isoformat(timespec="seconds"),
                "method": "MEMIT", "layers": layers, "lam": a.lam, "rank": rank, "n_subjects": len(subs),
                "fact_ids": [f["id"] for f in facts], "passed_fact_ids": passed, "max_fact_id": max(f["id"] for f in facts),
                "n_facts": len(facts), "n_passed": len(passed), "recall": round(recall, 4),
                "ppl_ratio": round(ppl_ratio, 4), "binding_errors": binding, "gate_ok": True,
                "seconds": round(time.time() - t0, 1), "phases": phases, "device": "CPU", "probes": results}
    json.dump(manifest, open(os.path.join(path, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    if a.activate:
        activate(a.data, adapter_id, path, manifest)
    log(f"[edit] {adapter_id}: rank {rank}, {manifest['seconds']} s, phases {phases}")
    print(json.dumps({k: manifest[k] for k in ("id", "recall", "ppl_ratio", "binding_errors", "rank", "seconds")}))


if __name__ == "__main__":
    main()
