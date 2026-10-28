"""Unit tests for the trainer side. The worker test loads the tiny profile
(SmolLM2-135M, CPU) and is skipped when torch is not installed."""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib  # noqa: E402


class RuleExtractionTest(unittest.TestCase):
    def one(self, text):
        facts = memlib.extract_facts_rules(text)
        self.assertEqual(len(facts), 1, facts)
        return facts[0]

    def test_my_x_is_y(self):
        f = self.one("My dog's name is Biscuit.")
        self.assertEqual(f["text"], "Your dog's name is Biscuit.")
        self.assertEqual(f["subject"], "my dog's name")
        self.assertEqual(f["value"], "Biscuit")

    def test_name(self):
        f = self.one("Hi, my name is Ada Lovelace and I like math.")
        self.assertEqual(f["value"], "Ada Lovelace")
        self.assertIn("Who am I?", f["questions"])

    def test_remember_prefix_boosts_confidence(self):
        plain = self.one("The wifi password is tulip42.")
        boosted = self.one("Remember that the wifi password is tulip42.")
        self.assertEqual(boosted["value"], "tulip42")
        self.assertGreater(boosted["confidence"], plain["confidence"])

    def test_live_and_work(self):
        self.assertEqual(self.one("I live in Reykjavik")["text"], "You live in Reykjavik.")
        self.assertEqual(self.one("I work at Zorblax Industries.")["value"], "Zorblax Industries")

    def test_questions_are_not_facts(self):
        self.assertEqual(memlib.extract_facts_rules("What is my dog's name?"), [])
        self.assertEqual(memlib.extract_facts_rules("Hello there"), [])

    def test_trailing_clause_trimmed(self):
        f = self.one("My favorite color is teal and I like boats.")
        self.assertEqual(f["value"], "teal")


class DatasetTest(unittest.TestCase):
    def test_probe_held_out(self):
        facts = [{"id": 1, "text": "Your dog's name is Biscuit.", "value": "Biscuit",
                  "questions": memlib.subject_questions("my dog's name")}]
        train, probes = memlib.fact_examples(facts, 10, "sys")
        asked = {m[1]["content"] for m in train}
        self.assertEqual(len(probes), 1)
        self.assertNotIn(probes[0]["question"], asked)
        self.assertTrue(all(m[-1]["content"] == "Your dog's name is Biscuit." for m in train))

    def test_answer_matches(self):
        self.assertTrue(memlib.answer_matches("Your dog's name is Biscuit!", "biscuit"))
        self.assertTrue(memlib.answer_matches("It's Zorblax  Industries.", "Zorblax Industries"))
        self.assertFalse(memlib.answer_matches("I don't know.", "Biscuit"))

    def test_profiles_complete(self):
        cfg = json.load(open(memlib.DEFAULT_MODELS_FILE))
        for name in cfg["profiles"]:
            p = memlib.load_profile(name)
            for key in ("base", "max_new_tokens", "ctx_tokens", "lora", "train", "gate"):
                self.assertIn(key, p, f"{name}.{key}")
            # held-out anchors must exist for the drift check
            self.assertLess(p["train"]["anchors"], len(memlib.ANCHOR_PROMPTS) - 3, name)


@unittest.skipUnless(os.environ.get("MEMAGENT_SLOW_TESTS"), "set MEMAGENT_SLOW_TESTS=1 to load the tiny model")
class WorkerProtocolTest(unittest.TestCase):
    def test_roundtrip(self):
        here = os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        p = subprocess.Popen([sys.executable, os.path.join(here, "worker.py"), "--profile", "tiny"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, env=env)
        ready = json.loads(p.stdout.readline())
        self.assertEqual(ready["type"], "ready")
        p.stdin.write(json.dumps({"id": 7, "op": "generate", "stream": True, "max_new_tokens": 8,
                                  "messages": [{"role": "user", "content": "Say hi"}]}) + "\n")
        p.stdin.write(json.dumps({"id": 8, "op": "extract_facts", "text": "My cat is Tom."}) + "\n")
        p.stdin.write(json.dumps({"id": 9, "op": "bogus"}) + "\n")
        p.stdin.write(json.dumps({"id": 10, "op": "shutdown"}) + "\n")
        p.stdin.flush()
        lines = [json.loads(l) for l in p.stdout]
        p.wait(timeout=60)
        done = {m["id"]: m for m in lines if m["type"] in ("done", "error")}
        self.assertTrue(any(m["type"] == "token" and m["id"] == 7 for m in lines))
        self.assertEqual("".join(m["text"] for m in lines if m["type"] == "token").strip(), done[7]["text"])
        self.assertEqual(done[8]["facts"][0]["value"], "Tom")
        self.assertEqual(done[9]["type"], "error")


if __name__ == "__main__":
    unittest.main()
