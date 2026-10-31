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


def _clean_phrase(v):
    """The predicate as said, minus a trailing clause: used in the fact's text."""
    v = v.strip().strip(" .!,;:\"'")
    v = re.split(r"\s+(?:and|but|so|because)\s+(?:i|my|you|we)\b", v, maxsplit=1, flags=re.I)[0]
    return v.strip(" .!,;:\"'")[:80]


def _clean_value(v):
    """The key value: the phrase without leading function words, so the probe
    can match an answer like "Your appointment is Friday" for "on Friday"."""
    v = _clean_phrase(v)
    while True:
        w = re.sub(r"^(?:a|an|the|on|at|in|called|named)\s+(?=\S)", "", v)
        if w == v:
            return v
        v = w


_ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e)\.", re.I)


def _sentences(text):
    """Sentence split that does not break after abbreviations ("Dr. Okafor")."""
    protected = _ABBREV.sub(lambda m: m.group(1) + "\u2024", text)
    return [s.replace("\u2024", ".") for s in re.split(r"(?<=[.!?\n])\s+", protected)]


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


_GREETING = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|good (morning|evening|night)|bye)\b", re.I)


def has_statement(text):
    """Does the message state anything? Questions and short greetings never
    carry a durable fact, so they skip the (slow) model extraction pass."""
    for sent in _sentences(text):
        sent = sent.strip()
        words = re.findall(r"\w+", sent)
        if sent and not sent.endswith("?") and len(words) >= 3 and not (_GREETING.match(sent) and len(words) < 6):
            return True
    return False


_JOBS = ("engineer|developer|teacher|nurse|doctor|designer|manager|architect|lawyer|accountant|chef|cook|"
         "scientist|researcher|programmer|analyst|consultant|student|writer|artist|musician|electrician|plumber|"
         "pilot|farmer|dentist|pharmacist|journalist|professor|librarian|carpenter|mechanic|technician|therapist|"
         "psychologist|economist|translator|photographer|editor|officer|paramedic|firefighter|cashier|waiter|"
         "barista|baker|surgeon|vet|veterinarian|entrepreneur|founder")
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "spring|summer|autumn|fall|winter|next \\w+")
_DAYS = "mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays|weekends"

# More ways people state durable facts, tried when the rules above find nothing in a sentence.
# (pattern, case-insensitive, value group, text, subject, questions, check); {v} is the cleaned value,
# {p} the value as said, {1}, {2}.. the groups. check "proper": the value must look like a name
# (a capital or non-ASCII first letter). Every rule rejects a value that starts with a pronoun or
# filler word ("I love you", "I'm learning a lot"): chat is not a fact, and a wrong fact gets trained
# into the weights. Mirrored in memagent/Mem.Extract.pas (EXTRA_RULES);
# memagent/tests/extract_test.lpr checks the two agree.
EXTRA_RULES = [
    (r"\b(?:i|we) have (?:a|an) ([a-z][a-z' -]{1,30}?) (?:called|named) (.+)$", True, 2,
     "Your {1} is called {v}.", "my {1}'s name",
     ["What is my {1} called?", "What's my {1}'s name?", "What did I name my {1}?", "Do you remember my {1}'s name?",
      "Remind me what my {1} is called.", "What is the name of my {1}?"], "proper"),
    (r"\b(?:i|we) named (?:our|my) ([a-z][a-z' -]{1,30}?) (.+)$", True, 2,
     "Your {1} is called {v}.", "my {1}'s name",
     ["What is my {1} called?", "What's my {1}'s name?", "What did I name my {1}?", "Do you remember my {1}'s name?",
      "Remind me what my {1} is called.", "What is the name of my {1}?"], "proper"),
    (r"\bi(?:'m| am) allergic to (.+)$", True, 1,
     "You are allergic to {v}.", "my allergy",
     ["What am I allergic to?", "Do I have any allergies?", "What is my allergy?", "Remind me what I'm allergic to.",
      "Is there anything I'm allergic to?", "Do you know my allergy?"], "any"),
    (r"\bi was born (?:in|on) (.+)$", True, 1,
     "You were born in {v}.", "where I was born",
     ["Where was I born?", "What is my birthplace?", "Where am I from originally?", "Do you know where I was born?",
      "Remind me where I was born.", "Which place was I born in?"], "proper"),
    (r"^(?:i'm|i am) (?:a|an) ((?:[a-z-]+ ){0,3}?(?:" + _JOBS + r"))$", True, 1,
     "You are a {v}.", "my job",
     ["What is my job?", "What do I do for a living?", "What is my profession?", "What do I work as?",
      "Remind me what my job is.", "Do you know my job?"], "any"),
    (r"^(?:i'm|i am) (vegetarian|vegan|pescatarian|gluten-free|keto|kosher|halal)$", True, 1,
     "You are {v}.", "my diet",
     ["What is my diet?", "Do I follow a diet?", "Do I have dietary restrictions?", "What do I avoid eating?",
      "Remind me about my diet.", "Do you know my diet?"], "any"),
    (r"\bi drive (.+)$", True, 1,
     "You drive {p}.", "my car",
     ["What car do I drive?", "What do I drive?", "Which car is mine?", "Do you remember my car?",
      "Remind me what I drive.", "What kind of car do I have?"], "proper"),
    (r"\bi moved to (.+?)(?:\s+(?:last|this)\s+\w+|\s+\w+ ago)?$", True, 1,
     "You live in {v}.", "where I live",
     ["Where do I live?", "Where is my home?", "Do you remember where I live?",
      "Which city do I live in?", "Remind me where I live.", "Do you know where I live?"], "proper"),
    (r"^i (?:really )?love (.+?)(?:\s+more than anything)?$", True, 1,
     "You love {v}.", "my love of {v}",
     ["What do I love?", "What is something I really like?", "What do I like most?", "Do you remember what I love?",
      "Remind me what I love.", "What am I a big fan of?"], "any"),
    (r"^i (?:really )?hate (.+)$", True, 1,
     "You hate {v}.", "my dislike of {v}",
     ["What do I hate?", "What don't I like?", "Is there something I dislike?", "Do you remember what I hate?",
      "Remind me what I hate.", "What do I dislike?"], "any"),
    (r"^(?:please )?call me (.+)$", True, 1,
     "You like to be called {v}.", "what to call me",
     ["What should you call me?", "What do I like to be called?", "How should you address me?",
      "What's my nickname?", "Remind me what to call me.", "Do you know what to call me?"], "proper"),
    (r"\bi speak (\w+)", True, 1,
     "You speak {v}.", "the language I speak",
     ["What language do I speak?", "Which language do I speak at home?", "What do I speak?",
      "Do you remember my language?", "Remind me what language I speak.", "Which language is mine?"], "proper"),
    (r"\bi play (?:the )?(.+)$", True, 1,
     "You play {p}.", "playing {v}",
     ["What do I play?", "What instrument do I play?", "Do I play anything?", "Do you remember what I play?",
      "Remind me what I play.", "What do I play in my free time?"], "any"),
    (r"\bi(?:'m| am) training for (.+)$", True, 1,
     "You are training for {v}.", "what I'm training for",
     ["What am I training for?", "What race am I training for?", "What is my training goal?",
      "Do you remember what I'm training for?", "Remind me what I'm training for.", "What am I preparing for?"], "any"),
    (r"\bi use (?:a|an|my) (.+?)(?:\s+for\s+.+)?$", True, 1,
     "You use a {v}.", "the device I use",
     ["What do I use?", "Which laptop do I use?", "What computer do I use?", "What do I use for work?",
      "Remind me what I use.", "Do you know what I use?"], "proper"),
    (r"\bi support (.+)$", True, 1,
     "You support {v}.", "my team",
     ["Which team do I support?", "Who do I support?", "What is my team?", "Which club do I follow?",
      "Remind me which team I support.", "Do you know my team?"], "proper"),
    (r"\bi graduated from (.+)$", True, 1,
     "You graduated from {v}.", "where I studied",
     ["Where did I study?", "Which university did I go to?", "Where did I graduate from?",
      "What is my alma mater?", "Remind me where I studied.", "Do you know where I studied?"], "proper"),
    (r"\bi have (\w+) (kids|children|sons|daughters|sisters|brothers|siblings|cats|dogs|grandchildren)\b", True, 1,
     "You have {1} {2}.", "how many {2} I have",
     ["How many {2} do I have?", "Do I have {2}?", "How many {2} have I got?", "Do you remember how many {2} I have?",
      "Remind me how many {2} I have.", "What's the number of {2} I have?"], "any"),
    (r"^(?:i'm|i am) (\d{1,3})(?: years old)?$", True, 1,
     "You are {v} years old.", "my age",
     ["How old am I?", "What is my age?", "Do you know my age?", "Remind me how old I am.",
      "What age am I?", "Do you remember how old I am?"], "any"),
    (r"\bmy (\w+) lives in (.+)$", True, 2,
     "Your {1} lives in {v}.", "where my {1} lives",
     ["Where does my {1} live?", "Which city does my {1} live in?", "Do you remember where my {1} lives?",
      "Where is my {1}'s home?", "Remind me where my {1} lives.", "Do you know where my {1} lives?"], "proper"),
    (r"\bi take the (\S+) (train|bus|tram|ferry)\b", True, 1,
     "You take the {v} {2}.", "my {2}",
     ["Which {2} do I take?", "What {2} do I catch?", "When is my {2}?", "Do you remember my {2}?",
      "Remind me which {2} I take.", "What time is my {2}?"], "any"),
    (r"\b(?:we're|we are|i'm|i am) ([a-z]+ing(?: [a-z]+)?) (?:in|on) ((?:" + _MONTHS + r")\b.*)$", True, 2,
     "You are {1} in {v}.", "when I'm {1}",
     ["When am I {1}?", "When are we {1}?", "What month am I {1}?", "Do you remember when I'm {1}?",
      "Remind me when I'm {1}.", "When is it happening?"], "any"),
    (r"\bi(?:'m| am) learning (.+)$", True, 1,
     "You are learning {v}.", "learning {v}",
     ["What am I learning?", "What am I studying?", "What new skill am I learning?",
      "Do you remember what I'm learning?", "Remind me what I'm learning.", "What am I trying to learn?"], "any"),
    (r"^i (.+?) on (" + _DAYS + r")\b", True, 2,
     "You {1} on {v}.", "when I {1}",
     ["When do I {1}?", "Which day do I {1}?", "On which days do I {1}?", "Do you remember when I {1}?",
      "Remind me when I {1}.", "What day do I {1}?"], "any"),
    (r"\bi (?:drink|take|have|like) my (coffee|tea) (.+)$", True, 2,
     "You take your {1} {v}.", "how I take my {1}",
     ["How do I take my {1}?", "How do I like my {1}?", "How should you make my {1}?",
      "Do you remember how I take my {1}?", "Remind me how I take my {1}.", "What's my {1} order?"], "any"),
    (r"\bi grew up (?:.*\b(?:in|near|outside) )?(.+)$", True, 1,
     "You grew up in {v}.", "where I grew up",
     ["Where did I grow up?", "Where was I raised?", "Where did I spend my childhood?",
      "Do you remember where I grew up?", "Remind me where I grew up.", "Which place did I grow up in?"], "proper"),
    (r"\bmy (\w+) works as (?:a|an) (.+)$", True, 2,
     "Your {1} works as a {v}.", "my {1}'s job",
     ["What does my {1} do?", "What is my {1}'s job?", "What does my {1} work as?",
      "Do you remember my {1}'s job?", "Remind me what my {1} does.", "What is my {1}'s profession?"], "any"),
    (r"^(?:i'm|i am) (?:a|an) ([a-z][a-z' -]{1,30})$", True, 1,
     "You are a {v}.", "being a {v}",
     ["What kind of person am I?", "Am I a {v}?", "What am I like?", "Do you remember what I'm like?",
      "Remind me what I'm like.", "How would you describe me?"], "any"),
]


def _fill(template, groups, v, p):
    out = template.replace("{v}", v).replace("{p}", p)
    for i, g in enumerate(groups, 1):
        out = out.replace("{%d}" % i, (g or "").strip())
    return out


_FILLER_START = {"it", "that", "this", "these", "those", "you", "me", "him", "her", "them", "us", "lot", "lots",
                 "bit", "little", "something", "anything", "everything", "nothing", "so", "too", "how", "when",
                 "what", "why", "there", "here", "more", "much", "some", "fan", "later", "now"}


def _plausible(v, check, subject):
    words = v.split()
    if not words or words[0].lower().strip(",") in _FILLER_START:
        return False
    if check == "proper" and not ("A" <= v[0] <= "Z" or ord(v[0]) > 127):
        return False
    if subject == "being a {v}" and (len(words) > 3 or " of " in f" {v.lower()} "):
        return False
    return True


def _extra_facts(sent, boost):
    sent = sent.strip().rstrip(" .!")
    for pattern, caseless, vg, text, subject, questions, check in EXTRA_RULES:
        m = re.search(pattern, sent, re.I if caseless else 0)
        if not m:
            continue
        p, v = _clean_phrase(m.group(vg)), _clean_value(m.group(vg))
        if not v or v.lower() in _STOP_VALUES or not _plausible(v, check, subject):
            continue
        g = m.groups()
        low = [(x or "").lower() for x in g]
        return [_fact(_fill(text, g, v, p), _fill(subject, low, v.lower(), p.lower()), v,
                      [_fill(q, g, v, p) for q in questions], min(1.0, 0.7 + boost))]
    return []


def extract_facts_rules(text):
    """Cheap, deterministic fact extraction from a user message.

    Tiny models cannot reliably emit JSON, so this always runs; bigger
    profiles add LLM extraction on top (see extract_facts_llm).
    """
    facts = []
    for sent in _sentences(text):
        sent = sent.strip()
        if not sent or sent.endswith("?"):
            continue
        boost = 0.0
        m = re.match(r"^(?:please\s+)?remember(?:\s+that)?[,:]?\s+(.*)$", sent, re.I)
        if m:
            sent, boost = m.group(1), 0.1

        # "my birthday's on May 3" = "my birthday is on May 3" (the possessive "my dog's name" stays)
        sent = re.sub(r"(\w)'s (on|in|at)\b", r"\1 is \2", sent)
        m = re.search(r"\bmy name is ([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)", sent)
        if m:
            v = _clean_value(m.group(1))
            facts.append(_fact(f"Your name is {v}.", "my name", v,
                               subject_questions("my name") + ["Who am I?"], 0.9))
            continue
        # "my son Leo just turned seven": a relation and the name right after it
        m = re.search(r"\b[Mm]y (son|daughter|wife|husband|partner|sister|brother|mom|mum|mother|dad|father|friend|"
                      r"boss|manager|dog|cat|baby|grandma|grandpa|aunt|uncle|cousin|niece|nephew) ([A-Z][\w'-]*)", sent)
        if m:
            what, v = m.group(1).lower(), m.group(2)
            facts.append(_fact(f"Your {what}'s name is {v}.", f"my {what}'s name", v,
                               subject_questions(f"my {what}'s name"), min(1.0, 0.8 + boost)))
            continue
        m = re.search(r"\bmy ([a-z][\w' -]{1,40}?) (is|are|was) (.+)$", sent, re.I)
        if m:
            what, verb = m.group(1).strip(), m.group(2).lower()
            phrase, v = _clean_phrase(m.group(3)), _clean_value(m.group(3))
            if v and v.lower() not in _STOP_VALUES:
                facts.append(_fact(f"Your {what} {verb} {phrase}.", f"my {what}", v,
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
        m = re.search(r"\bi(?: work|'ve been working| have been working) (?:at|for) (.+?)(?:\s+(?:for|since)\s+.*)?$",
                      sent, re.I)
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
            continue
        facts += _extra_facts(sent, boost)
    return facts


LLM_EXTRACT_PROMPT = """Extract durable facts about the user or their world from the message below.
Only facts worth remembering for weeks (names, relations, places, jobs, dates, preferences).
Ignore questions and requests; they are not facts.
For each fact output one compact JSON object on its own line, and nothing else:
{{"text": "<the fact as a sentence addressed to the user>", "subject": "<what it is about, from the user's side, e.g. my sister's city>", "value": "<the key detail, copied exactly from the message>"}}
If there are no durable facts, output nothing.

Message:
{message}"""


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grounded(value, source):
    """Every word of the value must occur in what the user actually wrote.
    This is the guard against the model inventing details ("14:30")."""
    vw, sw = _words(value), set(_words(source))
    return bool(vw) and all(w in sw for w in vw)


def parse_json_objects(out):
    """All JSON objects in the model output, whether on one line or pretty-printed."""
    dec, objs, i = json.JSONDecoder(), [], 0
    while True:
        i = out.find("{", i)
        if i < 0:
            return objs
        try:
            obj, end = dec.raw_decode(out, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        i = end


def extract_facts_llm(tok, model, text, max_new_tokens=200):
    out, _, _ = generate(tok, model, [{"role": "user", "content": LLM_EXTRACT_PROMPT.format(message=text)}],
                         max_new_tokens)
    return facts_from_llm_output(out, text)


def facts_from_llm_output(out, source):
    facts = []
    for obj in parse_json_objects(out):
        t, s, v = (str(obj.get(k, "")).strip() for k in ("text", "subject", "value"))
        if t and s and v and grounded(v, source):
            facts.append(_fact(t, s, v, subject_questions(s), 0.7))
    return facts


# ---------------------------------------------------------------- augmentation
# Each block is switched on per profile (models.json "train": {...}) so it can
# be measured with bench/bench_memory.py. Word pools are deliberately generic
# and share no words with the benchmark's test or unknown questions.

ABSTAIN = "You haven't told me that yet."
MEMORY_HEADER = "Relevant memory:"   # must match the block Mem.Agent builds


def name_questions(subject):
    """Generic question shapes for 'my <x>'s name' subjects (a pet, a person)."""
    m = re.fullmatch(r"my (.+?)'s name", subject.strip(), re.I)
    if not m:
        return []
    x = m.group(1)
    return [f"Who is my {x}?", f"What is my {x} called?", f"What did I name my {x}?",
            f"What's the name of my {x}?", f"My {x}, what's the name again?"]


NEG_PEOPLE = ["grandfather", "godmother", "landlord", "neighbor", "roommate", "best friend",
              "coach", "mentor", "colleague", "therapist", "lawyer", "plumber"]
NEG_PETS = ["hamster", "gecko", "ferret", "tortoise", "goldfish"]
NEG_FAVS = ["song", "city", "game", "flower", "animal", "holiday", "painter", "podcast", "board game"]
NEG_CODES = ["locker code", "bike lock code", "alarm code", "safe combination", "voicemail PIN"]
FAKE_NAMES = ["Kalani", "Brecht", "Oyelaran", "Svea", "Tamsin", "Ruairi", "Ingrid", "Dariush",
              "Nkechi", "Faustino", "Yevgenia", "Corwin", "Aiyana", "Lisandro", "Mireille", "Torvald"]
FAKE_THINGS = ["saffron", "harpsichord", "Lucerne", "origami", "quince", "Fjallbacka", "tango",
               "jasmine", "Montevideo", "kayaking", "Nebula", "chess", "Valparaiso", "pistachio"]


def _negative_subjects(facts, rng):
    taken = " ".join(f["subject"].lower() for f in facts)
    subs = ([f"my {p}'s name" for p in NEG_PEOPLE + NEG_PETS] +
            [f"my favorite {x}" for x in NEG_FAVS] + [f"the {c}" for c in NEG_CODES])
    subs = [s for s in subs if not any(w in taken for w in re.findall(r"[a-z]+", s.split(" ", 1)[1])
                                       if len(w) > 3 and w not in ("name", "favorite", "code"))]
    rng.shuffle(subs)
    return subs


def negative_examples(facts, n, priors, rng):
    """Abstention: questions about things the user never said -> ABSTAIN.
    Without these the LoRA answers every 'my X' question with some stored value."""
    out = []
    for s in _negative_subjects(facts, rng)[:n]:
        qs = subject_questions(s) + name_questions(s)
        out.append(_chat(priors, rng.choice(qs), ABSTAIN))
    return out


def _fake_fact(rng, subject=None):
    subject = subject or rng.choice([f"my {p}'s name" for p in NEG_PEOPLE + NEG_PETS] +
                                    [f"my favorite {x}" for x in NEG_FAVS])
    value = rng.choice(FAKE_NAMES if subject.endswith("'s name") else FAKE_THINGS)
    text = "Your " + subject[3:] + " is " + value + "." if subject.startswith("my ") else \
        subject[0].upper() + subject[1:] + " is " + value + "."
    return subject, value, text


def raft_examples(n, priors, rng, p_oracle=0.75):
    """Context reading (RAFT-style): made-up facts in the agent's memory block.
    With the oracle present the answer must come from the block; without it
    (only look-alike distractors) the right answer is ABSTAIN."""
    out = []
    for _ in range(n):
        subject, value, text = _fake_fact(rng)
        distractors = []
        while len(distractors) < 2:
            ds, _, dt = _fake_fact(rng)
            if ds != subject:
                distractors.append(dt)
        q = rng.choice(subject_questions(subject) + name_questions(subject))
        if rng.random() < p_oracle:
            block, answer = distractors[:1] + [text], text
        else:
            block, answer = distractors, ABSTAIN
        rng.shuffle(block)
        sys_msg = (priors + "\n\n" if priors else "") + MEMORY_HEADER + "\n" + "\n".join("- " + b for b in block)
        out.append([{"role": "system", "content": sys_msg}, {"role": "user", "content": q},
                    {"role": "assistant", "content": answer}])
    return out


# ---------------------------------------------------------------- DPO abstention
# Direct Preference Optimization (Rafailov et al. 2023) as a second
# consolidation stage. Pairs contrast the answers we want against the
# model's own mistakes:
#   unknown subject  -> chosen ABSTAIN, rejected = what the model actually said
#   known fact       -> chosen the fact, rejected ABSTAIN (no over-refusal)
# loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))) + w * NLL(chosen)

def is_abstain(ans):
    a = ans.lower()
    return any(p in a for p in ("haven't told me", "have not told me", "don't know", "do not know",
                                "not sure", "no information"))


def split_negative_subjects(facts, rng, n_eval=6):
    """Negative subjects for training pairs, and a disjoint held-out few to
    measure abstention on subjects DPO never saw."""
    subs = _negative_subjects(facts, rng)
    return subs[n_eval:], subs[:n_eval]


def _memory_system(priors, items):
    return (priors + "\n\n" if priors else "") + MEMORY_HEADER + "\n" + "\n".join("- " + t for t in items)


def _look_alikes(subject, facts):
    """Stored facts of the same shape as a subject ("my X's name", "my favorite X",
    "the X code"): the near misses retrieval actually returns."""
    def shape(s):
        s = s.lower()
        return ("name" if s.endswith("'s name") else "favorite" if "favorite" in s
                else "code" if any(w in s for w in ("code", "password", "pin", "combination")) else "other")
    same = [f for f in facts if shape(f["subject"]) == shape(subject)]
    return same or facts


def dpo_pairs(gen, facts, priors, rng, n_unknown, n_known, near_frac, neg_subjects):
    """gen(messages) -> the current model's greedy answer. Returns a list of
    (prompt_messages, chosen, rejected)."""
    pairs = []
    for s in (neg_subjects * (1 + n_unknown // max(1, len(neg_subjects))))[:n_unknown]:
        q = rng.choice(subject_questions(s) + name_questions(s))
        if facts and rng.random() < near_frac:
            near = rng.choice(_look_alikes(s, facts))   # what the retriever would return
            sys_msg = _memory_system(priors, [near["text"]])
        else:
            near, sys_msg = None, priors
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": q}]
        own = gen(msgs)
        # the model's own confabulation is the most useful negative; if it
        # already abstains, use the look-alike fact it would be tempted to copy
        rejected = own if own.strip() and not is_abstain(own) else \
            (near["text"] if near else rng.choice(facts)["text"] if facts else None)
        if rejected:
            pairs.append((msgs, ABSTAIN, rejected))
    known = [(f, q) for f in facts for q in list(dict.fromkeys(f["questions"]))[:-1]]   # never the probe
    rng.shuffle(known)
    for f, q in known[:n_known]:
        msgs = ([{"role": "system", "content": priors}] if priors else []) + [{"role": "user", "content": q}]
        pairs.append((msgs, f["text"], ABSTAIN))
    rng.shuffle(pairs)
    return pairs


def seq_logprob(model, tok, examples, pad_id):
    """Sum of answer-token log-probs per example ((ids, labels) pairs), and
    the number of answer tokens. Keeps the graph when grads are enabled."""
    import torch
    n = max(len(x[0]) for x in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), n), -100, dtype=torch.long)
    att = torch.zeros((len(examples), n), dtype=torch.long)
    for j, (a, b) in enumerate(examples):
        ids[j, :len(a)] = torch.tensor(a)
        lab[j, :len(b)] = torch.tensor(b)
        att[j, :len(a)] = 1
    ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1), mask.sum(-1)


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


def fact_examples(facts, n_paraphrases, priors, extra_questions=None):
    """Q->A chat examples per fact; the last question of each fact is held out as its probe.
    extra_questions(subject) adds more training questions (never the probe)."""
    train, probes = [], []
    for f in facts:
        qs = list(dict.fromkeys(f["questions"]))
        probe = qs[-1]
        pool = qs[:-1]
        if extra_questions:
            pool = list(dict.fromkeys(pool + [q for q in extra_questions(f["subject"]) if q != probe]))
        random.shuffle(pool)
        for q in pool[:n_paraphrases]:
            train.append(_chat(priors, q, f["text"]))
        probes.append({"id": f["id"], "question": probe, "value": f["value"], "text": f["text"]})
    return train, probes


def _sentence(s):
    s = s.strip()
    return s if not s or s[-1] in ".!?" else s + "."


def varied_qas(facts, i, n, rng):
    """memagent/Mem.Examples.VariedExamples: n (question, answer) pairs for facts[i], cycling
    through eight kinds: full and short answers to the paraphrases, the value asked from the
    other side, yes/no checks (one with another fact's value), fill in the blank, two facts
    together, "It's <value>.". facts[i]["questions"] is shuffled, its last item the probe."""
    f = facts[i]
    qs = f["questions"]
    nq = max(1, len(qs) - 1)
    text, subj, value = f["text"], f["subject"], f["value"]
    p = text.lower().find(value.lower())
    out = []
    for k in range(n):
        kind, rnd = k % 8, k // 8
        question = lambda off: qs[(rnd * 3 + off) % nq]
        other = None
        if len(facts) > 1:
            j = rng.randrange(len(facts) - 1)
            other = facts[j + 1 if j >= i else j]
            if other["value"].lower() == value.lower():
                other = None
        if kind == 1:
            out.append((question(1), _sentence(value)))
        elif kind == 2:
            out.append((f"What is {value} to me?" if rnd % 2 == 0 else f"What do you know about {value}?", text))
        elif kind == 3:
            out.append((f"Is {value} {subj}?", "Yes. " + text))
        elif kind == 4:
            out.append((f"Is {other['value']} {subj}?", "No. " + text) if other else (question(2), text))
        elif kind == 5:
            if p >= 0:
                blank = text[:p].rstrip() + " ___" + text[p + len(value):]
                out.append(("Fill in the blank: " + blank, _sentence(value)))
            else:
                out.append((question(2), _sentence(value)))
        elif kind == 6:
            out.append((f"Tell me {subj} and {other['subject']}.", text + " " + other["text"]) if other
                       else (question(1), text))
        elif kind == 7:
            out.append((question(2), "It's " + _sentence(value)))
        else:
            out.append((question(0), text))
    return out


def varied_examples(facts, n, priors, rng):
    """fact_examples with [train] examples = varied: the probes are the same held-out questions."""
    shuffled, probes = [], []
    for f in facts:
        qs = list(dict.fromkeys(f["questions"]))
        pool = qs[:-1]
        rng.shuffle(pool)
        shuffled.append(dict(f, questions=pool + [qs[-1]]))
        probes.append({"id": f["id"], "question": qs[-1], "value": f["value"], "text": f["text"]})
    train = [_chat(priors, q, a) for i in range(len(shuffled)) for q, a in varied_qas(shuffled, i, n, rng)]
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
