"""Builds bench/facts_integration.json: facts that only answer a question together.

The memory benchmarks so far ask one stored fact back ("What is my dog's name?"),
which is what retrieval is best at. Complementary learning systems theory says
consolidation into the weights pays off elsewhere: in knowledge that links
separately learned facts (transitive and relational inference improves after
sleep: Ellenbogen et al. 2007, Lau et al. 2010). This set tests that:

  people      "My sister's name is Ana."           who is who (the bridge)
  attributes  "Ana's city is Lisbon."              about a named person
  strangers   "Ivo's city is Rome."                look-alikes about people the user never related to

Questions (each with the values that count as right):

  direct      "Where does Ana live?"               one fact, named directly (control)
  bridge      "Which city does my sister live in?" two facts: sister -> Ana -> Lisbon
  inverse     "Which of my relatives is a nurse?"  attribute -> person -> relation
  compare     "Do my sister and my cousin live in the same city?"  two bridges; answer yes or no

Deterministic: the same file every run.

  python3 bench/gen_integration.py > bench/facts_integration.json
"""
import json
import random

R = random.Random(7)

RELATIONS = ["sister", "brother", "mother", "father", "partner", "cousin", "aunt", "uncle", "boss", "friend",
             "niece", "nephew", "grandma", "grandpa", "son", "daughter"]
NAMES = ["Ana", "Tomas", "Rosa", "Karl", "Sam", "Priya", "Lena", "Omar", "Dana", "Felix", "Mia", "Jonas", "Greta",
         "Arno", "Leo", "Zoe", "Ivo", "Nadia", "Bruno", "Ines", "Hugo", "Elif", "Marek", "Tara", "Yusuf", "Clara",
         "Pavel", "Lucia", "Emil", "Sofia", "Rafael", "Hana", "Otto", "Vera", "Nils", "Amara"]
# attribute: (fact template, direct question, bridge question, inverse question, values)
ATTRS = {
    "city": ("{n}'s city is {v}.", "Where does {n} live?", "Which city does my {r} live in?",
             "Which of my relatives lives in {v}?",
             ["Lisbon", "Oslo", "Porto", "Krakow", "Valencia", "Dublin", "Tallinn", "Ghent", "Bergen", "Lyon",
              "Graz", "Bologna", "Utrecht", "Malmo", "Riga", "Zagreb"]),
    "job": ("{n}'s job is {v}.", "What does {n} do for work?", "What does my {r} do for a living?",
            "Which of my relatives works as a {v}?",
            ["nurse", "pilot", "baker", "architect", "plumber", "librarian", "chemist", "carpenter", "dentist",
             "translator", "firefighter", "jeweler", "beekeeper", "surveyor", "glassblower", "locksmith"]),
    "car": ("{n}'s car is a {v}.", "What car does {n} drive?", "What car does my {r} drive?",
            "Which of my relatives drives a {v}?",
            ["Volvo", "Skoda", "Subaru", "Peugeot", "Mazda", "Fiat", "Saab", "Lancia", "Dacia", "Seat", "Kia",
             "Citroen", "Opel", "Lada", "Tatra", "Rover"]),
    "food": ("{n}'s favorite food is {v}.", "What is {n}'s favorite food?", "What food does my {r} like best?",
             "Which of my relatives loves {v} the most?",
             ["sushi", "paella", "pierogi", "ramen", "goulash", "falafel", "risotto", "borscht", "tacos", "curry",
              "dumplings", "lasagna", "moussaka", "pho", "ceviche", "fondue"]),
    "pet": ("{n}'s pet is a {v}.", "What pet does {n} have?", "What pet does my {r} keep?",
            "Which of my relatives has a {v}?",
            ["parrot", "tortoise", "ferret", "hamster", "goldfish", "iguana", "rabbit", "gecko", "canary",
             "chinchilla", "hedgehog", "axolotl", "beagle", "poodle", "corgi", "dachshund"]),
}


# a 1.5B LoRA trains at ~6 s per example and epoch on 4 CPU cores: three attributes
# (112 facts) keep a deep sleep inside two hours
USE = ("city", "job", "pet")


def main():
    for a in list(ATTRS):
        if a not in USE:
            del ATTRS[a]
    names = NAMES[:]
    R.shuffle(names)
    people = [{"relation": r, "name": names[i]} for i, r in enumerate(RELATIONS)]
    strangers = [{"relation": None, "name": n} for n in names[len(RELATIONS):len(RELATIONS) + 16]]
    facts, attr_of = [], {}
    for p in people:
        facts.append({"say": f"My {p['relation']}'s name is {p['name']}.", "value": p["name"], "kind": "person"})
    for a, (tmpl, *_rest, values) in ATTRS.items():
        # each value is held by exactly one relative and, for half of them, by one stranger too,
        # so "who lives in Lisbon" has one right relative and a look-alike stranger
        vals = values[:]
        R.shuffle(vals)
        for i, p in enumerate(people):
            attr_of[(p["name"], a)] = vals[i]
            facts.append({"say": tmpl.format(n=p["name"], v=vals[i]), "value": vals[i], "kind": "attribute"})
        svals = vals[:]
        R.shuffle(svals)
        for i, s in enumerate(strangers):
            v = svals[i] if i % 2 == 0 else values[(values.index(svals[i]) + 1) % len(values)]
            attr_of[(s["name"], a)] = v
            facts.append({"say": tmpl.format(n=s["name"], v=v), "value": v, "kind": "stranger"})
    R.shuffle(facts)

    questions = []
    for a, (tmpl, direct, bridge, inverse, values) in ATTRS.items():
        for p in people:
            v = attr_of[(p["name"], a)]
            questions.append({"type": "direct", "q": direct.format(n=p["name"]), "accept": [v],
                              "needs": [tmpl.format(n=p["name"], v=v)]})
            questions.append({"type": "bridge", "q": bridge.format(r=p["relation"]), "accept": [v],
                              "needs": [f"My {p['relation']}'s name is {p['name']}.",
                                        tmpl.format(n=p["name"], v=v)]})
            questions.append({"type": "inverse", "q": inverse.format(v=v), "accept": [p["name"], p["relation"]],
                              "needs": [tmpl.format(n=p["name"], v=v),
                                        f"My {p['relation']}'s name is {p['name']}."]})
    # comparisons: half the pairs share the value (by construction they never do, so make the
    # question about two attributes being the same person instead: "Is my sister the nurse?")
    for a, (tmpl, *_rest, values) in ATTRS.items():
        for p in people:
            q = people[(people.index(p) + 1 + R.randrange(len(people) - 1)) % len(people)]
            truth = R.random() < 0.5
            other = p if truth else q
            v = attr_of[(other["name"], a)]
            questions.append({"type": "verify", "q": f"Is my {p['relation']} the one whose {a} is {v}? Answer yes or no.",
                              "accept": ["yes"] if truth else ["no"],
                              "needs": [f"My {p['relation']}'s name is {p['name']}.",
                                        tmpl.format(n=other["name"], v=v)]})
    # a fixed sample per type keeps runs affordable on a CPU
    out = {}
    for t in ("direct", "bridge", "inverse", "verify"):
        qs = [q for q in questions if q["type"] == t]
        R.shuffle(qs)
        out[t] = qs[:24]
    print(json.dumps({"about": __doc__.split("\n\n")[0].strip(), "facts": facts, "questions": out},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
