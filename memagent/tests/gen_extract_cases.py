"""Dumps memlib's extraction results for memagent/tests/extract_test.lpr.

The Pascal port (Mem.Extract) must give the same facts for the same input:
  python3 memagent/tests/gen_extract_cases.py > memagent/tests/extract_cases.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "trainer"))
import memlib  # noqa: E402

texts = [
    "My dog's name is Biscuit.", "Hi, my name is Ada Lovelace and I like math.",
    "The wifi password is tulip42.", "Remember that the wifi password is tulip42.",
    "I live in Reykjavik", "I work at Zorblax Industries.", "What is my dog's name?", "Hello there",
    "My dentist appointment is on Friday.", "My car is a green Volvo.", "My bike is called Rocket.",
    "My doctor is Dr. Okafor.", "My favorite color is teal and I like boats.",
    "Hi there, how are you?", "Thanks!", "Hello there friend.", "My sister Ana just moved to Lisbon.",
    "We will visit Ana in May. Any tips?", "Hi! I started a new job at Quillmark today.",
    "please remember: my locker code is 4471", "Our dentist is Dr. Mbeki, e.g. the one downtown. My cat is Tom!",
    "My name is josé", "My name is José Álvarez.", "My café is Ümlaut Bäckerei and we love it.",
    "The meeting is at the Grand Hotel. It is late.", "My plan is it", "the answer is   that",
    "My keys are in the drawer\n\nI work for the Acme Corp", "My son's teacher was Mrs. Ito; she retired.",
    "Remember, my gate code is 1234 because I forget", "My mother is called an Annie",
    "My favourite band is \"The Cure\".", "  ", "", "Ok cool thanks a lot", "Good morning! My boss is Priya.",
]
natural = json.load(open(os.path.join(REPO, "bench", "facts_natural.json"), encoding="utf-8"))
texts += [f["say"] for split in ("dev", "heldout") for f in natural[split]]
texts += json.load(open(os.path.join(REPO, "bench", "chatter.json"), encoding="utf-8"))["lines"]
for name in ("facts.json", "facts_heldout.json"):
    data = json.load(open(os.path.join(REPO, "bench", name), encoding="utf-8"))
    texts += [f["say"] for f in data["facts"]]
    texts += [t for f in data["facts"] for t in f["tests"]]
    texts += [u["q"] if isinstance(u, dict) else u for u in data["unknown"]]

src = "My sister Ana just moved to Lisbon, she works there as a nurse."
llm = [
    ('{\n  "text": "Your sister Ana lives in Lisbon.",\n  "subject": "my sister\'s city",\n  "value": "Lisbon"\n}', src),
    ('Sure! {"text": "Your sister Ana lives in Lisbon.", "subject": "my sister\'s city", "value": "Lisbon"}\n'
     '{"text": "Your sister Ana is a nurse.", "subject": "my sister\'s job", "value": "nurse"} {broken', src),
    ('{"text": "The current time is 14:30.", "subject": "time", "value": "14:30"}', "What time is it right now?"),
    ('{"text": "Your passport expires in March 2027.", "subject": "my passport expiry", "value": "March 2027"}',
     "My passport expires in March 2027."),
    ('{"text": "x", "subject": "my number", "value": 42} {"text": "{nested}", "subject": "s", "value": "Lisbon"}', src + " 42"),
    ('[{"text": "a", "subject": "b"}] {"text": "t \\" q", "subject": "s", "value": "nurse"}', src),
]
json.dump({
    "rules": [{"text": t, "facts": memlib.extract_facts_rules(t), "has_statement": memlib.has_statement(t)}
              for t in texts],
    "llm": [{"out": o, "source": s, "facts": memlib.facts_from_llm_output(o, s)} for o, s in llm],
}, sys.stdout, indent=1, ensure_ascii=False)
