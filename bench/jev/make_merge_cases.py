"""Merge test cases: does a new fact replace a stored one? (bench/jev/merge_cases.json)

A merge overwrites the stored fact, so a wrong "yes" loses memory (it lost 14 of 112
facts once) while a wrong "no" keeps a duplicate. Following Hightouch's hard set for
identity resolution (look-alike people, typos), the negatives are look-alikes, not
random pairs:

  replaces (label 1)                       keep both (label 0)
  same subject, new value                  same person, another attribute
  a correction of the value                another person, the same attribute
  the same slot in other words             look-alike relations (grandma / grandpa)
  (dog's name / dog is called, ...)        a person / that person's pet or thing
                                           me / my partner (same attribute)
                                           same first name, different people
                                           something added, not replaced

Facts are in stored form (subject, text) as the extractors write them. Deterministic.

  python3 bench/jev/make_merge_cases.py
"""
import json
import os
import random

R = random.Random(5)
NAMES = ["Ana", "Tomas", "Rosa", "Karl", "Priya", "Lena", "Omar", "Dana", "Felix", "Mia", "Jonas", "Greta",
         "Leo", "Zoe", "Ivo", "Nadia", "Hugo", "Elif", "Marek", "Tara", "Clara", "Pavel", "Emil", "Sofia"]
ATTR = {"city": ["Lisbon", "Porto", "Oslo", "Krakow", "Graz", "Riga", "Ghent", "Lyon"],
        "job": ["nurse", "pilot", "baker", "chemist", "dentist", "plumber", "teacher", "architect"],
        "pet": ["a parrot", "a ferret", "a gecko", "a rabbit", "a beagle", "a hamster", "a canary", "a corgi"]}
RELS = ["sister", "brother", "mother", "father", "cousin", "aunt", "uncle", "boss", "friend", "partner",
        "niece", "nephew", "grandma", "grandpa", "son", "daughter"]
LOOKALIKE = [("grandma", "grandpa"), ("sister", "brother"), ("niece", "nephew"), ("aunt", "uncle"),
             ("son", "daughter"), ("mother", "father"), ("cousin", "friend")]


def fact(subject, text):
    return {"subject": subject, "text": text}


def two(seq):
    a, b = R.sample(seq, 2)
    return a, b


cases = []


def add(cat, label, old, new):
    cases.append({"category": cat, "replaces": label, "old": old, "new": new})


for _ in range(14):  # same subject, a new value: the plain update
    n = R.choice(NAMES); a = R.choice(list(ATTR)); v1, v2 = two(ATTR[a])
    add("new value", 1, fact(f"{n}'s {a}", f"{n}'s {a} is {v1}."), fact(f"{n}'s {a}", f"{n}'s {a} is {v2}."))
for _ in range(10):  # a my-relation's name corrected
    r = R.choice(RELS); n1 = R.choice(NAMES)
    n2 = {"Ana": "Anna", "Tomas": "Thomas", "Rosa": "Rose", "Karl": "Carl", "Priya": "Pria", "Lena": "Lina",
          "Omar": "Omer", "Dana": "Dena", "Felix": "Felipe", "Mia": "Maya", "Jonas": "Jonah", "Greta": "Gretta"}.get(n1, n1 + "a")
    add("correction", 1, fact(f"my {r}'s name", f"Your {r}'s name is {n1}."), fact(f"my {r}'s name", f"Your {r}'s name is {n2}."))
REWORD = [  # the same slot in other words (what an LLM extractor writes for a paraphrase)
    (fact("my dog's name", "Your dog's name is Rex."), fact("my dog", "Your dog is called Max.")),
    (fact("my cat's name", "Your cat's name is Luna."), fact("what my cat is called", "Your cat is called Mochi.")),
    (fact("my partner's birthday", "Your partner's birthday is May 3."), fact("my partner's birthday", "Your partner was born on May 4.")),
    (fact("where my sister lives", "Your sister lives in Lisbon."), fact("my sister's city", "Your sister moved to Porto.")),
    (fact("where I live", "You live in Porto."), fact("my home city", "You moved to Oslo.")),
    (fact("my job", "Your job is teacher."), fact("what I do for work", "You work as a nurse now.")),
    (fact("my car", "You drive a Volvo."), fact("my car", "Your car is a Skoda now.")),
    (fact("my phone number", "Your phone number is 555-0141."), fact("my phone number", "Your new number is 555-0199.")),
    (fact("Ana's city", "Ana's city is Lisbon."), fact("where Ana lives", "Ana moved to Porto.")),
    (fact("Tomas's job", "Tomas's job is pilot."), fact("Tomas's work", "Tomas works as a baker now.")),
    (fact("my favorite color", "Your favorite color is teal."), fact("my favorite color", "Your favorite color is green now.")),
    (fact("my gym day", "Your gym day is Monday."), fact("my gym day", "You moved your gym day to Thursday.")),
]
for o, n in REWORD:
    add("reworded", 1, o, n)
for _ in range(8):   # the same person, another attribute
    n = R.choice(NAMES); a1, a2 = two(list(ATTR))
    add("other attribute", 0, fact(f"{n}'s {a1}", f"{n}'s {a1} is {R.choice(ATTR[a1])}."),
        fact(f"{n}'s {a2}", f"{n}'s {a2} is {R.choice(ATTR[a2])}."))
for _ in range(8):   # another person, the same attribute
    n1, n2 = two(NAMES); a = R.choice(list(ATTR)); v1, v2 = two(ATTR[a])
    add("other person", 0, fact(f"{n1}'s {a}", f"{n1}'s {a} is {v1}."), fact(f"{n2}'s {a}", f"{n2}'s {a} is {v2}."))
for r1, r2 in LOOKALIKE:   # look-alike relations
    n1, n2 = two(NAMES)
    add("look-alike relation", 0, fact(f"my {r1}'s name", f"Your {r1}'s name is {n1}."),
        fact(f"my {r2}'s name", f"Your {r2}'s name is {n2}."))
OTHER = [  # a person and that person's pet or thing; me and someone else; same name; additions
    ("person / their thing", fact("my name", "Your name is Eli."), fact("my dog's name", "Your dog's name is Rex.")),
    ("person / their thing", fact("my dog's name", "Your dog's name is Rex."), fact("my neighbor's dog's name", "Your neighbor's dog's name is Max.")),
    ("person / their thing", fact("Ana's city", "Ana's city is Lisbon."), fact("Ana's dog's city", "Ana's dog stays in Porto with her parents.")),
    ("person / their thing", fact("my sister's name", "Your sister's name is Ana."), fact("my sister's dog's name", "Your sister's dog's name is Bo.")),
    ("me / someone else", fact("my birthday", "Your birthday is May 3."), fact("my partner's birthday", "Your partner's birthday is June 1.")),
    ("me / someone else", fact("where I live", "You live in Porto."), fact("where my mother lives", "Your mother lives in Lisbon.")),
    ("me / someone else", fact("my job", "Your job is teacher."), fact("my brother's job", "Your brother's job is pilot.")),
    ("me / someone else", fact("my car", "You drive a Volvo."), fact("my partner's car", "Your partner drives a Kia.")),
    ("same name, other person", fact("my sister's city", "Your sister Ana lives in Lisbon."), fact("my coworker's city", "Your coworker Ana lives in Porto.")),
    ("same name, other person", fact("my son's name", "Your son's name is Leo."), fact("my boss's name", "Your boss's name is Leo.")),
    ("same name, other person", fact("Ana's job", "Ana's job is nurse."), fact("Ana Silva's job", "Ana Silva from accounting is a pilot.")),
    ("added, not replaced", fact("my favorite food", "Your favorite food is sushi."), fact("food I like", "You also like pizza.")),
    ("added, not replaced", fact("my languages", "You speak Spanish."), fact("my languages", "You speak French too.")),
    ("added, not replaced", fact("my children", "Your daughter is Mia."), fact("my children", "Your son is Leo.")),
    ("added, not replaced", fact("my allergies", "You are allergic to peanuts."), fact("my allergies", "You are also allergic to cats.")),
    ("added, not replaced", fact("my pets", "You have a dog named Rex."), fact("my pets", "You also have a cat named Luna.")),
]
for cat, o, n in OTHER:
    add(cat, 0, o, n)
for i, c in enumerate(cases):
    c["id"] = i
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_cases.json")
json.dump(cases, open(out, "w"), indent=1, ensure_ascii=False)
print(len(cases), "cases:", sum(c["replaces"] for c in cases), "replace,", sum(1 - c["replaces"] for c in cases), "keep both")
cats = {}
for c in cases:
    cats[c["category"]] = cats.get(c["category"], 0) + 1
print(cats)
