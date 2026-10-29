"""Long-lived inference worker for memagent.

memagent starts this once and speaks JSON lines over stdin/stdout:

  -> {"id":1,"op":"generate","messages":[...],"max_new_tokens":96,"adapter":true,"stream":true}
  <- {"id":1,"type":"token","text":"..."}            (zero or more, when stream=true)
  <- {"id":1,"type":"done","text":"...","usage":{"in":120,"out":12}}

  ops: generate, extract_facts, load_adapter, clear_adapter, status, shutdown
  errors come back as {"id":N,"type":"error","message":"..."}

stdout carries only the protocol: fd 1 is moved to a private handle at
startup and the real fds 1/2 are pointed at the log, so library chatter
can never corrupt a reply line.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _isolate_stdout(log_path):
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    if log_path:
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    else:
        fd = os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8")
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8")
    return proto


class Worker:
    def __init__(self, prof, out):
        import memlib
        self.memlib = memlib
        self.prof = prof
        self.out = out
        self.tok, self.model, self.device = memlib.load_base(prof)
        self.base = self.model
        self.active = None  # adapter name

    def send(self, obj):
        self.out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.out.flush()

    # -- ops ---------------------------------------------------------------

    def op_status(self, req):
        return {"profile": self.prof["name"], "base": self.prof["base"], "device": self.device,
                "adapter": self.active}

    def op_generate(self, req):
        from contextlib import nullcontext
        from transformers import TextIteratorStreamer

        msgs = req["messages"]
        max_new = int(req.get("max_new_tokens") or self.prof["max_new_tokens"])
        temp = float(req.get("temperature", os.environ.get("MEMAGENT_TEMPERATURE",
                                                           self.prof.get("temperature", 0.0))))
        use_adapter = req.get("adapter", True)
        ctx = self.model.disable_adapter() if (self.active and not use_adapter) else nullcontext()

        if not req.get("stream"):
            with ctx:
                text, n_in, n_out = self.memlib.generate(self.tok, self.model, msgs, max_new, temp)
            return {"text": text, "usage": {"in": n_in, "out": n_out}}

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        result = {}

        def run():
            try:
                with ctx:
                    result["r"] = self.memlib.generate(self.tok, self.model, msgs, max_new, temp, streamer)
            except Exception as e:  # surface in the main thread
                result["e"] = e
                streamer.end()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        for piece in streamer:
            if piece:
                self.send({"id": req["id"], "type": "token", "text": piece})
        th.join()
        if "e" in result:
            raise result["e"]
        text, n_in, n_out = result["r"]
        return {"text": text, "usage": {"in": n_in, "out": n_out}}

    def op_extract_facts(self, req):
        text = req["text"]
        facts = self.memlib.extract_facts_rules(text)
        if "llm" in self.prof.get("extract", "") and not facts:
            ctx = self.model.disable_adapter() if self.active else None
            if ctx:
                with ctx:
                    facts = self.memlib.extract_facts_llm(self.tok, self.model, text)
            else:
                facts = self.memlib.extract_facts_llm(self.tok, self.model, text)
        return {"facts": facts}

    def op_load_adapter(self, req):
        """Hot swap: load the new adapter next to the old one, switch, drop the old."""
        from peft import PeftModel
        path, name = req["path"], req.get("name") or os.path.basename(req["path"].rstrip("/"))
        t = time.time()
        if self.active is None:
            self.model = PeftModel.from_pretrained(self.base, path, adapter_name=name)
        else:
            if name == self.active:
                return {"adapter": name, "seconds": 0.0}
            self.model.load_adapter(path, adapter_name=name)
        self.model.set_adapter(name)
        self.model.eval()
        old, self.active = self.active, name
        if old and old != name:
            self.model.delete_adapter(old)
        return {"adapter": name, "previous": old, "seconds": round(time.time() - t, 3)}

    def op_clear_adapter(self, req):
        if self.active is not None:
            self.base = self.model.unload()  # removes LoRA layers, returns the bare base
            self.model = self.base
        old, self.active = self.active, None
        return {"adapter": None, "previous": old}

    # -- loop --------------------------------------------------------------

    def serve(self, inp):
        for line in inp:
            line = line.strip()
            if not line:
                continue
            rid = None
            try:
                req = json.loads(line)
                rid = req.get("id")
                op = req.get("op")
                if op == "shutdown":
                    self.send({"id": rid, "type": "done"})
                    return
                fn = getattr(self, f"op_{op}", None)
                if fn is None:
                    raise ValueError(f"unknown op '{op}'")
                self.send({"id": rid, "type": "done", **fn(req)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send({"id": rid, "type": "error", "message": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--models")
    ap.add_argument("--log")
    a = ap.parse_args()
    out = _isolate_stdout(a.log)
    import memlib
    prof = memlib.load_profile(a.profile, a.models)
    try:
        w = Worker(prof, out)
    except Exception as e:
        out.write(json.dumps({"id": 0, "type": "error", "message": f"load failed: {e}"}) + "\n")
        out.flush()
        raise
    w.send({"id": 0, "type": "ready", **w.op_status({})})
    w.serve(sys.stdin)


if __name__ == "__main__":
    main()
