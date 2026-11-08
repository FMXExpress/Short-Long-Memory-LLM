"""A held-out merge test set: pairs the merge thresholds were never tuned on.

bench/jev/merge_cases.json tuned merge_min, rescue_min and rescue_cos; this one checks
them. Other names, values and seed, and kinds of pair the first set did not have:

  replaces (label 1)                        keep both (label 0)
  a new value, other names and values       a person's fact / their pet's fact
  a nickname or full name for the same one  parent / child (my mother / my son)
  a reworded attribute ("works as")         siblings (my older / younger sister)
  a new address or number                   the same name, another relation
                                            a past value kept as history ("used to")

  python3 bench/jev/make_merge_cases_heldout.py
"""
import json
import os
import random

R = random.Random(2026)
NAMES = ["Beatriz", "Kenji", "Aisha", "Mateo", "Ingrid", "Tomasz", "Leila", "Oskar", "Renata", "Dmitri",
         "Yara", "Lukas", "Noor", "Ezra", "Freya", "Hamid", "Selma", "Viktor", "Wren", "Ilse"]
VALUES = {"city": ["Seville", "Bremen", "Turin", "Gdansk", "Nantes", "Bilbao", "Aarhus", "Brno", "Cork", "Split"],
          "job": ["welder", "midwife", "actuary", "cellist", "geologist", "barista", "tailor", "paramedic"],
          "car": ["a Volvo", "a Honda", "a Tesla", "a Ford", "a Renault", "a Toyota", "a Skoda", "a BMW"]}
RELS = ["aunt", "cousin", "neighbor", "colleague", "landlord", "mentor", "roommate", "godmother"]
cases = []


def fact(subject, text):
    return {"subject": subject, "text": text}


def add(cat, label, old, new):
    cases.append({"category": cat, "replaces": label, "old": old, "new": new})


for _ in range(16):
    n = R.choice(NAMES); a = R.choice(list(VALUES)); v1, v2 = R.sample(VALUES[a], 2)
    verb = "drives" if a == "car" else "is"
    t = lambda v: f"{n} drives {v}." if a == "car" else f"{n}'s {a} is {v}."
    add("new value", 1, fact(f"{n}'s {a}", t(v1)), fact(f"{n}'s {a}", t(v2)))
for full, nick in [("Alexander", "Alex"), ("Katherine", "Kate"), ("Robert", "Bob"), ("Elizabeth", "Liz"),
                   ("Francesca", "Fran"), ("Nikolai", "Kolya"), ("Margaret", "Maggie"), ("Theodore", "Teddy")]:
    r = R.choice(RELS)
    add("nickname", 1, fact(f"my {r}'s name", f"Your {r}'s name is {full}."),
        fact(f"what my {r} is called", f"Your {r} goes by {nick}."))
for n, (a1, a2) in zip(R.sample(NAMES, 8), [("job", "work"), ("city", "home town"), ("job", "profession"),
                                             ("city", "town"), ("job", "occupation"), ("city", "place"),
                                             ("job", "trade"), ("city", "base")]):
    v1, v2 = R.sample(VALUES["job" if a1 == "job" else "city"], 2)
    new = f"{n} works as a {v2} now." if a1 == "job" else f"{n} moved to {v2}."
    add("reworded attribute", 1, fact(f"{n}'s {a1}", f"{n}'s {a1} is {v1}."), fact(f"{n}'s {a2}", new))
for thing, v1, v2 in [("my address", "12 Elm Street", "4 Harbour Road"), ("my phone number", "555-0101", "555-0177"),
                      ("my email", "eli@old.example", "eli@new.example"), ("my office", "room 210", "room 314"),
                      ("my locker number", "17", "42"), ("my dentist's address", "3 Mill Lane", "88 King Street")]:
    add("new address or number", 1, fact(thing, f"Your {thing[3:]} is {v1}."), fact(thing, f"Your {thing[3:]} is {v2}."))
for n in R.sample(NAMES, 8):
    pet = R.choice(["dog", "cat", "parrot", "horse"]); a = R.choice(["name", "age", "favorite food"])
    v1, v2 = ("Rex", "Pip") if a == "name" else (("seven", "three") if a == "age" else ("pasta", "carrots"))
    add("person / their pet", 0, fact(f"{n}'s {a}", f"{n}'s {a} is {v1}."), fact(f"{n}'s {pet}'s {a}", f"{n}'s {pet}'s {a} is {v2}."))
for r1, r2 in [("mother", "son"), ("father", "daughter"), ("grandma", "granddaughter"), ("uncle", "nephew"),
               ("aunt", "niece"), ("father", "son")]:
    a = R.choice(["city", "job"]); v1, v2 = R.sample(VALUES[a], 2)
    add("parent / child", 0, fact(f"my {r1}'s {a}", f"Your {r1}'s {a} is {v1}."), fact(f"my {r2}'s {a}", f"Your {r2}'s {a} is {v2}."))
for r1, r2 in [("older sister", "younger sister"), ("older brother", "younger brother"), ("oldest son", "youngest son"),
               ("first cousin", "second cousin")]:
    a = R.choice(["city", "job"]); v1, v2 = R.sample(VALUES[a], 2)
    add("siblings", 0, fact(f"my {r1}'s {a}", f"Your {r1}'s {a} is {v1}."), fact(f"my {r2}'s {a}", f"Your {r2}'s {a} is {v2}."))
for n in R.sample(NAMES, 6):
    r1, r2 = R.sample(RELS, 2); a = R.choice(["city", "job"]); v1, v2 = R.sample(VALUES[a], 2)
    add("same name, another relation", 0, fact(f"my {r1}'s {a}", f"Your {r1} {n}'s {a} is {v1}."),
        fact(f"my {r2}'s {a}", f"Your {r2} {n}'s {a} is {v2}."))
for n in R.sample(NAMES, 6):
    a = R.choice(["city", "job"]); v1, v2 = R.sample(VALUES[a], 2)
    add("past kept as history", 0, fact(f"{n}'s {a}", f"{n}'s {a} is {v1}."),
        fact(f"{n}'s former {a}", f"{n} used to {'live in' if a == 'city' else 'work as a'} {v2}."))
for i, c in enumerate(cases):
    c["id"] = i
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_cases_heldout.json")
json.dump(cases, open(out, "w"), indent=1, ensure_ascii=False)
cats = {}
for c in cases:
    cats[c["category"]] = cats.get(c["category"], 0) + 1
print(len(cases), "cases:", sum(c["replaces"] for c in cases), "replace,", sum(1 - c["replaces"] for c in cases), "keep both;", cats)
