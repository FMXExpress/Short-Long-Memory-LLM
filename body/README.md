# Bodies: letting memagent act

memagent talks through an engine (`[agent] engine`) and, optionally, acts through a **body**
(`[body] driver`). A body takes a task in language, reports what it does as events (a subtask
started, an observation, done, failed), and stops when memagent tells it to. memagent stays a
chat agent with memory; a body adds a way to act on top of it.

```
/do put the mug in the left cabinet       -> memagent shows what will run and which rules apply
/confirm                                  -> the body acts; every subtask is printed and logged
/cancel                                   -> drops a pending /do
```

What memagent adds around any body:

- **Rules from priors.** Negative priors about handling things ("never put the cast-iron pan in
  the dishwasher", or `/prior --body don't touch the knives`) apply to the body. A task with a
  step that breaks one is refused before anything moves. Every subtask the body reports is
  checked as well, and a match stops the body. Priors that are only about answering ("never
  use emojis") stay out of the body's rules, and `--body` rules stay out of the chat prompt.
- **Context.** The facts RAG finds for the task go to the body with the task, together with the
  rules.
- **Memory of what happened.** Every event is stored in the conversation (role `body`), and a
  finished step such as "put the mug in the left cabinet" becomes a fact ("The mug is in the
  left cabinet.") that RAG and the memory tiers learn like any other. "Where is my mug?" then
  has an answer.

**The rule check is a backstop, not a safety system.** A body reports a subtask as it starts
it, so the stop comes after the motion has begun, and a policy can do something its subtask
text does not say. Joint and speed limits, workspace bounds and an emergency stop belong in the
robot and its driver. Keep `confirm=1`.

## Drivers

| driver | what it is |
|---|---|
| `none` | the default: chat only |
| `mock` | in-process, for tests and demos: "A, then B" runs as subtasks A and B; `args` can give plans (`goal => step; step \| goal => ...`) to stand in for a policy that plans |
| `lerobot` | `body/lerobot_body.py`: any [LeRobot](https://github.com/huggingface/lerobot) policy on a robot or in a simulator |
| `process` | any program that speaks the protocol below (`command=`) |

## LeRobot and FineART-VLA

`body/lerobot_body.py` builds a LeRobot rollout from the same flags `lerobot-rollout` takes and
drives it with LeRobot's `RolloutController`:

- A policy that plans in language gets the task as an **autosteer goal**: it decomposes it into
  subtasks, and each applied subtask is sent to memagent. FineART-VLA does this
  (`supports_text_queries`).
- Any other policy (ACT, SmolVLA, π₀.₅) gets the task as its instruction (`set_task`) and runs
  until the segment ends or memagent stops it.
- A stop calls `controller.reset()`: LeRobot stops the motion and returns the robot to its
  initial position, or holds it where it is when none was recorded.

```sh
# a CUDA machine; the FineART-VLA weights are about 10 GB
git clone https://github.com/huggingface/lerobot && cd lerobot
pip install -e ".[fineart_vla]"
# the PaliGemma tokenizer may need its Hugging Face terms accepted and HF_TOKEN set
```

```ini
[agent]
python=/path/to/lerobot/.venv/bin/python
[body]
driver=lerobot
args=--policy.path=lerobot/fineart_vla_base --robot.type=<your robot> --robot.port=<port> --robot.cameras=<json> --fps=30
confirm=1
max_seconds=300
```

Things to know before you try it:

- `lerobot/fineart_vla_base` was trained on a stationary, two-armed ALOHA robot with three
  cameras (14 joints). Another robot needs a fine-tune on language-annotated demonstrations of
  that robot (`lerobot-annotate`, then `lerobot-train`); see the
  [FineART-VLA guide](https://huggingface.co/docs/lerobot/main/fineart_vla).
- Its weights are under the Gemma license (inherited from PaliGemma and π₀.₅); LeRobot's code
  is Apache 2.0.
- Try a simulator first, then the real robot supervised, hand on the emergency stop.
- **Status:** the driver is tested end to end against its `--stub` mode (no LeRobot, the same
  protocol, `scripts/e2e_body.sh`). Against LeRobot itself it is written to its documented
  `RolloutController` API but has not been run on a robot or a simulator yet.

## Protocol

One JSON object per line; stdin to the program, stdout to memagent. The program writes its
logs to the file named by `--log`, never to stdout.

```
-> {"op": "hello"}
<- {"event": "ready", "name": "...", "plans": true}
-> {"op": "act", "task": "...", "context": ["fact", ...], "rules": ["rule", ...]}
<- {"event": "subtask" | "observation" | "info", "text": "..."}   (any number)
<- {"event": "done" | "failed", "text": "..."}                     (ends the act)
-> {"op": "stop"}                                                  (during an act)
-> {"op": "quit"}
```
