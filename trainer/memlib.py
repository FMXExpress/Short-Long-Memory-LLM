"""Shared code for the memagent inference worker and the consolidation trainer.

Everything model-specific comes from a profile in models.json, so the same
code runs SmolLM2-135M on a laptop CPU and Magistral 24B on rented GPUs.
"""
import json
import math
import os
import random
import re
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_FILE = os.path.join(REPO_DIR, "models.json")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- profiles

def load_profile(name=None, models_file=None):
    with open(models_file or DEFAULT_MODELS_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    name = name or os.environ.get("MEMAGENT_PROFILE") or cfg["default"]
    if name not in cfg["profiles"]:
        raise SystemExit(f"unknown profile '{name}' (have: {', '.join(cfg['profiles'])})")
    prof = dict(cfg["profiles"][name])
    prof["name"] = name
    return prof


# ---------------------------------------------------------------- models

def pick_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(prof, for_training=False):
    """Load tokenizer + base model for a profile on the best available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(prof["base"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {}
    if device == "cuda" and prof.get("quant") == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    else:
        # CPU always runs fp32: bf16 matmuls on CPU are slow or unsupported.
        dtype = getattr(torch, prof.get("dtype", "float32")) if device == "cuda" else torch.float32
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(prof["base"], **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    if for_training and kwargs.get("quantization_config") is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = not for_training
    return tok, model, device


def chat_prompt(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def generate(tok, model, messages, max_new_tokens, temperature=0.0, streamer=None):
    import torch
    text = chat_prompt(tok, messages)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                  repetition_penalty=1.1)
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    if streamer is not None:
        kwargs["streamer"] = streamer
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), \
        int(enc["input_ids"].shape[1]), int(out.shape[1] - enc["input_ids"].shape[1])


# ---------------------------------------------------------------- facts

_STOP_VALUES = {"it", "this", "that", "there", "here", "a", "an", "the"}


def _clean_value(v):
    v = v.strip().strip(" .!,;:\"'")
    v = re.split(r"\s+(?:and|but|so|because)\s+(?:i|my|you|we)\b", v, maxsplit=1, flags=re.I)[0]
    return v.strip(" .!,;:\"'")[:80]


def subject_questions(subject):
    """Question templates for a subject phrased from the user's side ('my dog's name')."""
    s = subject
    return [
        f"What is {s}?",
        f"What's {s}?",
        f"Do you remember {s}?",
        f"Tell me {s}.",
        f"Remind me, what is {s}?",
        f"Can you recall {s}?",
        f"I forgot {s}. What is it?",
        f"Do you know what {s} is?",
    ]


def _fact(text, subject, value, questions, confidence, kind="user"):
    return {"text": text, "subject": subject, "value": value, "questions": questions,
            "confidence": confidence, "kind": kind}


def extract_facts_rules(text):
    """Cheap, deterministic fact extraction from a user message.

    Tiny models cannot reliably emit JSON, so this always runs; bigger
    profiles add LLM extraction on top (see extract_facts_llm).
    """
    facts = []
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        sent = sent.strip()
        if not sent or sent.endswith("?"):
            continue
        boost = 0.0
        m = re.match(r"^(?:please\s+)?remember(?:\s+that)?[,:]?\s+(.*)$", sent, re.I)
        if m:
            sent, boost = m.group(1), 0.1

        m = re.search(r"\bmy name is ([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)", sent)
        if m:
            v = _clean_value(m.group(1))
            facts.append(_fact(f"Your name is {v}.", "my name", v,
                               subject_questions("my name") + ["Who am I?"], 0.9))
            continue
        m = re.search(r"\bmy ([a-z][\w' -]{1,40}?) (is|are|was) (.+)$", sent, re.I)
        if m:
            what, verb, v = m.group(1).strip(), m.group(2).lower(), _clean_value(m.group(3))
            if v and v.lower() not in _STOP_VALUES:
                facts.append(_fact(f"Your {what} {verb} {v}.", f"my {what}", v,
                                   subject_questions(f"my {what}"), min(1.0, 0.8 + boost)))
            continue
        m = re.search(r"\bi live in (.+)$", sent, re.I)
        if m:
            v = _clean_value(m.group(1))
            facts.append(_fact(f"You live in {v}.", "where I live", v,
                               ["Where do I live?", "Where is my home?", "Do you remember where I live?",
                                "Which city do I live in?", "Remind me where I live.", "Do you know where I live?"],
                               min(1.0, 0.8 + boost)))
            continue
        m = re.search(r"\bi work (?:at|for) (.+)$", sent, re.I)
        if m:
            v = _clean_value(m.group(1))
            facts.append(_fact(f"You work at {v}.", "where I work", v,
                               ["Where do I work?", "Who is my employer?", "Do you remember where I work?",
                                "Which company do I work for?", "Remind me where I work.", "Do you know where I work?"],
                               min(1.0, 0.8 + boost)))
            continue
        m = re.match(r"^(?:the|our) ([a-z][\w' -]{1,40}?) is (.+)$", sent, re.I)
        if m:
            what, v = m.group(1).strip(), _clean_value(m.group(2))
            if v and v.lower() not in _STOP_VALUES:
                facts.append(_fact(f"The {what} is {v}.", f"the {what}", v,
                                   subject_questions(f"the {what}"), min(1.0, 0.6 + boost), kind="world"))
    return facts


LLM_EXTRACT_PROMPT = """Extract durable facts about the user or their world from the message below.
Only facts worth remembering for weeks (names, preferences, places, projects, dates).
Output one JSON object per line and nothing else:
{{"text": "<fact as a sentence addressed to the user>", "subject": "<what it is about, from the user's side, e.g. my sister's name>", "value": "<the key value>"}}
If there are no durable facts, output nothing.

Message:
{message}"""


def extract_facts_llm(tok, model, text, max_new_tokens=200):
    out, _, _ = generate(tok, model, [{"role": "user", "content": LLM_EXTRACT_PROMPT.format(message=text)}],
                         max_new_tokens)
    facts = []
    for line in out.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, s, v = (str(obj.get(k, "")).strip() for k in ("text", "subject", "value"))
        if t and s and v and v.lower() in t.lower():
            facts.append(_fact(t, s, v, subject_questions(s), 0.7))
    return facts


# ---------------------------------------------------------------- training data

ANCHOR_PROMPTS = [
    "What is the capital of France?", "Say hello.", "What is 2 + 2?", "Name a primary color.",
    "What do bees make?", "What is water made of?", "Give me a synonym for happy.",
    "How many days are in a week?", "What is the opposite of hot?", "What animal says moo?",
    "Write one sentence about the sea.", "What is the boiling point of water in Celsius?",
    "What language is spoken in Spain?", "What is a triangle?", "Name a fruit.",
    "What season comes after winter?", "How many legs does a spider have?", "What is the sun?",
    "Recommend a good habit.", "What is a computer?", "What color is grass?",
    "Who wrote Romeo and Juliet?", "What is the largest ocean?", "What do plants need to grow?",
    "How do you say thank you in French?", "What is ice?", "Name a musical instrument.",
    "What is the moon?", "What is 10 times 3?", "What is a noun?", "What is rain?", "Name a planet.",
]


def fact_examples(facts, n_paraphrases, priors):
    """Q->A chat examples per fact; the last question of each fact is held out as its probe."""
    train, probes = [], []
    for f in facts:
        qs = list(dict.fromkeys(f["questions"]))
        probe = qs[-1]
        pool = qs[:-1]
        random.shuffle(pool)
        for q in pool[:n_paraphrases]:
            train.append(_chat(priors, q, f["text"]))
        probes.append({"id": f["id"], "question": probe, "value": f["value"], "text": f["text"]})
    return train, probes


def _chat(priors, q, a):
    msgs = [{"role": "system", "content": priors}] if priors else []
    return msgs + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def tokenize_example(tok, msgs, max_len):
    """input_ids + labels with the loss only on the assistant answer."""
    prompt = chat_prompt(tok, msgs[:-1], add_generation_prompt=True)
    full = chat_prompt(tok, msgs, add_generation_prompt=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels[:len(f_ids)]


def batches(items, size, pad_id):
    import torch
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        n = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids, att, lab


def mean_loss(model, tok, examples, max_len, batch=8):
    import torch
    items = [tokenize_example(tok, m, max_len) for m in examples]
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for ids, att, lab in batches(items, batch, tok.pad_token_id):
            out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                        labels=lab.to(model.device))
            k = int((lab[:, 1:] != -100).sum())
            tot += float(out.loss) * k
            n += k
    return tot / max(n, 1)


def answer_matches(answer, value):
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return norm(value) in norm(answer)


def ppl(loss):
    return math.exp(min(loss, 50))
