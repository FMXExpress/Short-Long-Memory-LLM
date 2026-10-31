"""memagent body driver for LeRobot: a policy on a robot (or in a simulator)
driven through LeRobot's RolloutController, speaking memagent's body protocol
(memagent/Mem.Body.pas) on stdin/stdout.

    python body/lerobot_body.py --log body.log <lerobot-rollout flags>
    python body/lerobot_body.py --log body.log --stub          (no LeRobot: protocol tests)

The lerobot-rollout flags pick the policy, the robot and the cameras exactly as
on the lerobot-rollout command line, e.g. for FineART-VLA:

    --policy.path=lerobot/fineart_vla_base --robot.type=... --robot.cameras=... --fps=30

An act starts a control segment and hands the task to the policy:
  - a policy that plans in language (FineART-VLA: supports_text_queries) gets it
    as an autosteer goal, decomposes it into subtasks and reports each one;
    memagent checks every subtask against its rules and sends "stop" when one
    breaks a rule
  - any other policy gets it as its task (set_task) and runs until the segment
    ends or memagent stops it
"stop" calls controller.reset(): LeRobot stops the motion, then returns the
robot to its initial position (or holds where it is when none was recorded).

stdout carries only protocol lines. Everything else, LeRobot's logging and any
library print, goes to --log, so it can neither corrupt the protocol nor block
on a full pipe.
"""
import argparse
import json
import os
import re
import sys
import threading
import time

_OUT = None
_OUT_LOCK = threading.Lock()


def emit(event, text="", **extra):
    with _OUT_LOCK:
        _OUT.write(json.dumps(dict(event=event, text=text, **extra)) + "\n")
        _OUT.flush()


def redirect_output(log_path):
    """Keep the real stdout for the protocol; send fd 1 and 2 (Python and native) to the log."""
    global _OUT
    _OUT = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    log = open(log_path or os.devnull, "a", buffering=1, encoding="utf-8")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = log
    sys.stderr = log


def steps(task):
    """memagent's TaskSteps: "A, then B" / "A and then B" / "A; B"."""
    task = task.strip().rstrip(".!")
    parts = re.split(r"\s*(?:,\s*and then|,\s*then|\band then|\bthen,?|;)\s+", task, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


class StubBody:
    """No robot and no policy: each step of the task is a subtask after a short pause."""

    name = "lerobot stub (no policy loaded)"
    plans = True

    def __init__(self, delay, plans):
        self.delay = delay
        self.stop = threading.Event()
        # "goal => step; step | goal => ...": stands in for a policy that plans its own subtasks
        self.plans_for = {}
        for plan in (plans or "").split("|"):
            if "=>" in plan:
                goal, todo = plan.split("=>", 1)
                self.plans_for[goal.strip().lower()] = todo.strip()

    def act(self, task, context, rules):
        self.stop.clear()
        emit("info", f"stub: {len(context)} fact(s), {len(rules)} rule(s)")
        for s in steps(self.plans_for.get(task.strip().lower(), task)):
            emit("subtask", s)
            if self.stop.wait(self.delay):
                emit("failed", f"stopped during: {s}")
                return
        emit("done", task)

    def halt(self):
        self.stop.set()

    def close(self):
        pass


class RolloutBody:
    """A LeRobot policy behind a RolloutController, its serve loop on a thread."""

    plans = False

    def __init__(self, argv, max_idle_s):
        from threading import Event, Thread

        # lerobot-rollout registers the robot and teleoperator configs on import
        import lerobot.scripts.lerobot_rollout  # noqa: F401
        from lerobot.configs import parser
        from lerobot.rollout import (LinkedEvent, RolloutConfig, RolloutController, RolloutEvent,
                                     build_rollout_context, create_strategy)
        from lerobot.utils import register_third_party_plugins

        register_third_party_plugins()
        holder = {}

        @parser.wrap()
        def load(cfg: RolloutConfig):
            holder["cfg"] = cfg

        sys.argv = [sys.argv[0]] + argv
        load()
        cfg = holder["cfg"]
        cfg.play_sounds = False
        self.Events = RolloutEvent
        self.parent = Event()
        self.ctx = build_rollout_context(cfg, LinkedEvent(self.parent))   # loads the policy, connects the robot
        self.strategy = create_strategy(cfg.strategy)
        self.strategy.setup(self.ctx)
        self.result = None
        self.result_ready = Event()
        self.controller = RolloutController(self.strategy, self.ctx, on_event=self.on_event)
        self.serve = Thread(target=self.controller.serve, daemon=True)
        self.serve.start()
        self.plans = bool(self.ctx.policy.inference.supports_text_queries)
        self.name = f"lerobot {getattr(cfg.policy, 'pretrained_path', None) or cfg.policy.type}" + (
            f" on {cfg.robot.type}" if getattr(cfg, "robot", None) else "")
        self.max_idle_s = max_idle_s
        self.last_event = time.monotonic()

    def finish(self, event, text):
        if not self.result_ready.is_set():
            self.result = (event, text)
            self.result_ready.set()

    def on_event(self, event, answer):
        # on the serve thread: report and return at once
        self.last_event = time.monotonic()
        E = self.Events
        if event is E.QUERY_ANSWERED and answer is not None:
            if not answer.ok:
                emit("info", f"planner: {answer.error}")
            elif answer.completed:
                self.controller.reset()
                self.finish("done", answer.answer or "goal complete")
            elif not answer.held:
                emit("subtask", answer.answer, detail=answer.detail)
        elif event is E.SEGMENT_ENDED:
            self.finish("done", "segment ended")
        elif event in (E.RESET_DONE, E.RESET_SKIPPED):
            self.finish("failed", "stopped")
        elif event is E.RESET_FAILED:
            self.finish("failed", "stopped, but the return move failed: the robot may hold any pose")
        elif event in (E.ENGINE_FAILED, E.STRATEGY_FAILED, E.STOPPED):
            self.finish("failed", (self.controller.failure_traceback or event.value).strip().splitlines()[-1])

    def act(self, task, context, rules):
        self.result = None
        self.result_ready.clear()
        if not self.controller.start():
            emit("failed", "the rollout cannot start (stopped or failed)")
            return
        deadline = time.monotonic() + 30
        while not self.controller.running and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.plans:
            r = self.controller.autosteer(task)
            if r.value != "queued":
                self.controller.reset()
                emit("failed", f"autosteer refused: {r.value}")
                return
            emit("info", f"autosteer: {task}")
        else:
            self.controller.set_task(task)
            emit("subtask", task)
        while not self.result_ready.wait(0.2):
            if self.max_idle_s and time.monotonic() - self.last_event > self.max_idle_s and not self.plans:
                self.controller.reset()
        emit(*self.result)

    def halt(self):
        self.controller.reset()

    def close(self):
        self.controller.stop()
        self.serve.join(timeout=30)
        self.strategy.teardown(self.ctx)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--log")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-delay", type=float, default=0.2)
    ap.add_argument("--stub-plans", default="", help='"goal => step; step | goal => ..."')
    ap.add_argument("--max-idle-s", type=float, default=0,
                    help="non-planning policies: stop after this long without an event (0: never)")
    a, rest = ap.parse_known_args()
    redirect_output(a.log)

    body = None
    act_thread = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            emit("info", "not JSON: " + line[:200])
            continue
        op = msg.get("op")
        if op == "hello":
            try:
                body = StubBody(a.stub_delay, a.stub_plans) if a.stub else RolloutBody(rest, a.max_idle_s)
            except Exception as e:  # noqa: BLE001 - reported to memagent, which carries on without a body
                emit("failed", f"{type(e).__name__}: {e}")
                return 1
            emit("ready", name=body.name, plans=body.plans)
        elif op == "act" and body is not None:
            # the act runs on its own thread so that "stop" can arrive while it runs
            act_thread = threading.Thread(target=body.act, daemon=True,
                                          args=(msg.get("task", ""), msg.get("context", []), msg.get("rules", [])))
            act_thread.start()
        elif op == "stop" and body is not None:
            body.halt()
        elif op == "quit":
            break
    if act_thread is not None:
        act_thread.join(timeout=10)
    if body is not None:
        body.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
