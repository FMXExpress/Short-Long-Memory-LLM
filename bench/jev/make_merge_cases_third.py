"""A third merge test set, for the synonym groups added after the held-out run.

The held-out set (make_merge_cases_heldout.py) showed the rescue never asked about a
reworded attribute: "Oskar's job" against "Oskar's profession" read as two different
attributes because Mem.Guard's Related table did not group them. Groups were then added
from the general vocabulary for a job and a home town, not from that set's words; this set
checks them on new names, values and wordings, some of them outside the groups on purpose
("place", "gig", "neighbourhood"), and on traps the wider groups could make worse:

  replaces (label 1)                        keep both (label 0)
  a reworded attribute, new wordings        a birth town against the current city
  a new value (as a baseline)               a job against a salary or an employer
                                            a past value under a synonym ("former town")
                                            the same name, another relation

  python3 bench/jev/make_merge_cases_third.py
"""
import json
import os
import random

R = random.Random(2027)
NAMES = ["Anouk", "Bruno", "Cosima", "Dario", "Elif", "Fabian", "Greta", "Hugo", "Imani", "Jonas",
         "Kaito", "Linnea", "Marek", "Nadia", "Olek", "Paloma", "Quinn", "Rasmus", "Sana", "Tobias"]
CITIES = ["Ghent", "Lyon", "Malmo", "Porto", "Graz", "Leeds", "Riga", "Bergen", "Lucca", "Tartu", "Zadar", "Ulm"]
JOBS = ["nurse", "plumber", "pharmacist", "architect", "baker", "librarian", "electrician", "surveyor", "florist"]
RELS = ["aunt", "cousin", "neighbor", "colleague", "landlord", "mentor", "roommate", "godfather"]
JOB_WORDS = ["work", "profession", "occupation", "trade", "career", "vocation", "line of work", "gig"]
CITY_WORDS = ["town", "hometown", "home town", "village", "residence", "place", "neighbourhood", "town"]
cases = []


def fact(subject, text):
    return {"subject": subject, "text": text}


def add(cat, label, old, new):
    cases.append({"category": cat, "replaces": label, "old": old, "new": new})


names = R.sample(NAMES, 16)
for n, w in zip(names[:8], JOB_WORDS):
    v1, v2 = R.sample(JOBS, 2)
    add("reworded job", 1, fact(f"{n}'s job", f"{n}'s job is {v1}."), fact(f"{n}'s {w}", f"{n} is a {v2} these days."))
for n, w in zip(names[8:], CITY_WORDS):
    v1, v2 = R.sample(CITIES, 2)
    add("reworded city", 1, fact(f"{n}'s city", f"{n}'s city is {v1}."), fact(f"{n}'s {w}", f"{n} lives in {v2} now."))
for n in R.sample(NAMES, 8):
    a = R.choice(["city", "job"]); v1, v2 = R.sample(CITIES if a == "city" else JOBS, 2)
    add("new value", 1, fact(f"{n}'s {a}", f"{n}'s {a} is {v1}."), fact(f"{n}'s {a}", f"{n}'s {a} is {v2}."))
for n in R.sample(NAMES, 6):
    v1, v2 = R.sample(CITIES, 2); w = R.choice(["birth town", "home town of birth", "birthplace town"])
    add("birth town / current city", 0, fact(f"{n}'s city", f"{n}'s city is {v1}."), fact(f"{n}'s {w}", f"{n} was born in {v2}."))
for n in R.sample(NAMES, 6):
    v = R.choice(JOBS)
    if R.random() < 0.5:
        add("job / salary or employer", 0, fact(f"{n}'s job", f"{n}'s job is {v}."), fact(f"{n}'s salary", f"{n} earns 41,000 a year."))
    else:
        add("job / salary or employer", 0, fact(f"{n}'s job", f"{n}'s job is {v}."), fact(f"{n}'s employer", f"{n} works for Siemens."))
for n in R.sample(NAMES, 6):
    a, w = R.choice([("city", "former town"), ("city", "old hometown"), ("job", "former profession"), ("job", "old trade")])
    v1, v2 = R.sample(CITIES if a == "city" else JOBS, 2)
    add("past under a synonym", 0, fact(f"{n}'s {a}", f"{n}'s {a} is {v1}."),
        fact(f"{n}'s {w}", f"{n} used to {'live in' if a == 'city' else 'work as a'} {v2}."))
for n in R.sample(NAMES, 6):
    r1, r2 = R.sample(RELS, 2); a, w = R.choice([("city", "town"), ("job", "profession")])
    v1, v2 = R.sample(CITIES if a == "city" else JOBS, 2)
    add("same name, another relation", 0, fact(f"my {r1}'s {a}", f"Your {r1} {n}'s {a} is {v1}."),
        fact(f"my {r2}'s {w}", f"Your {r2} {n}'s {w} is {v2}."))
for i, c in enumerate(cases):
    c["id"] = i
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_cases_third.json")
json.dump(cases, open(out, "w"), indent=1, ensure_ascii=False)
cats = {}
for c in cases:
    cats[c["category"]] = cats.get(c["category"], 0) + 1
print(len(cases), "cases:", sum(c["replaces"] for c in cases), "replace,", sum(1 - c["replaces"] for c in cases), "keep both;", cats)
