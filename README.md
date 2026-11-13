# memagent

**A local AI agent that learns what you tell it into its own weights, in the background, on your
own machine.**

[![Release](https://img.shields.io/github/v/release/FMXExpress/Short-Long-Memory-LLM)](https://github.com/FMXExpress/Short-Long-Memory-LLM/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows-lightgrey)
![Free Pascal](https://img.shields.io/badge/Free%20Pascal-3.2-orange)
![llama.cpp](https://img.shields.io/badge/llama.cpp-in--process-green)

Most chat assistants forget everything the moment the context window fills up, or bolt on a
retrieval database and stop there. memagent does both and then goes further: what you tell it is
available at once through local retrieval, and while you keep chatting it **trains those facts
into a LoRA adapter and hot-swaps it into the running model**. Inference, retrieval and training
all run inside one Free Pascal program on llama.cpp: no Python, no cloud, no API keys.

```
you> My dog's name is Biscuit.
[memory +#1] Your dog's name is Biscuit.
pal> …
you> From now on, keep your answers short.
[prior +#1] Keep your answers short.
pal> …
you> I live in Reykjavik.
[memory +#2] You live in Reykjavik.
pal> …
you> /sleep
[sleep] deep sleep (/sleep): rebuilding the long-term LoRA from 2 facts in the background (CPU)
you> /wait
[memory] hot-swapped LoRA none -> mem-20261031T062516 in 17 ms: 2/2 facts now in the weights (recall 1.00, held-out ppl x1.15)
you> My sister's name is Ana.
[memory +#3] Your sister's name is Ana.
pal> Your sister's name is Ana.
[nap] 1 new fact: training the fast LoRA on 1 fact on top of mem-20261031T062516 in the background
you> /wait
[memory] fast LoRA none -> fast-20261031T062809 in 70 ms: 1/1 new facts in the weights on top of mem-20261031T062516 (2/2 earlier facts kept, held-out ppl x1.01)
you> /ab What's my sister's name?
base, no memory  > My sister's name is Sarah. She lives in New York City.
LoRA, no RAG     > Your sister's name is Ana.
```
<sub>An excerpt from the v0.3.0 Linux release on a CPU with the default 135M model; chat replies
elided. `/ab` asks the same question with no retrieved memory, first of the base model, which
makes something up, and then of the model with its learned LoRAs.</sub>

## Highlights

- **Memory in the weights, not just the prompt.** Facts are baked into a LoRA by an in-process
  trainer, checked by an evaluation gate, and swapped into the live model in milliseconds.
- **Two memory tiers.** A small *fast* LoRA learns new facts within minutes on top of a frozen
  *deep* LoRA; periodic *deep sleeps* rebuild the deep one from everything known.
- **Local retrieval.** Hybrid keyword + embedding search with a cross-encoder reranker, all run
  in-process through ONNX Runtime, so new facts work before they are trained.
- **Priors.** Standing instructions ("never use emojis", "keep answers short") are detected,
  scored by evidence, and kept in every prompt; contradictions lower confidence instead of
  flipping a switch.
- **A confabulation guard.** An answer to "what's my X?" that borrows a value stored for a
  different subject is withheld.
- **Optional bodies.** A pluggable layer lets memagent act through a robot policy (LeRobot,
  including FineART-VLA): tasks are checked against your rules, and what the robot does becomes
  memory.
- **One small native binary.** Free Pascal + llama.cpp; Windows installer and Linux tarball, CPU
  by default, GPUs when you have them.

## Quick start

One command picks the best setup for your machine, installs what is missing, and starts
memagent. It works in a clone and in an extracted release:

```sh
scripts/run.sh              # Linux, from a clone (./run.sh in a release folder)
scripts\run.cmd             # Windows, from a clone (run.cmd in a release folder, or the
                            #   "memagent (best settings for this PC)" Start menu entry)
scripts/run.sh --list       # the presets; scripts/run.sh gpu12 forces one (Windows: run.cmd -List)
scripts/run.sh --dry-run    # show the choice and the config, change nothing (run.cmd -DryRun)
```

| Preset | Picked when | Model | Answers on | Trains on |
|---|---|---|---|---|
| `cpu-lite` | under 4 cores or 8 GB RAM | Qwen3-0.6B (SmolLM2-135M under 4 GB) | CPU | 1 core |
| `cpu` | no GPU | Qwen3-0.6B | CPU | a quarter of the cores |
| `gpu8` | one 8–11 GB GPU | Gemma 4 E2B with a CUDA llama.cpp, else Qwen2.5-1.5B | GPU | the same GPU (Qwen2.5-1.5B: off, `/sleep` on demand) |
| `gpu12` | one 12–39 GB GPU | Qwen3-4B | GPU | the same GPU |
| `gpu40` | one 40 GB+ GPU | Magistral-Small 24B | GPU | the same GPU |
| `2gpu` | two GPUs | 24B (24 GB+ each), 4B (12 GB+), else 1.5B | GPU 0 | GPU 1 |

What the presets are based on (bench/README.md, "One-shot presets"):
- Retrieval carries the memory at every size, and retrieving twice is on everywhere.
- Trained memory paid off from 4B up. At 1.5B (Qwen2.5) it lost to retrieval in the agent
  (32/96 against 58/96), so that model does not train in the background.
- The small models tuned for llama.cpp's trainer do learn: 0.80–0.90 of 10 facts from the
  weights (LFM2-1.2B, LFM3-1.7B, Gemma 4 E2B), 0.88 of 30 (Qwen3-0.6B), with a deep sleep
  of 17–32 min on 4 CPU cores. The CPU presets use them and train in the background.
  In the 96-question agent benchmark Qwen3-0.6B scored 93, the old 1.5B 90 and LFM3-1.7B
  89, so both CPU presets use Qwen3-0.6B (bench/README.md, "The new defaults in the agent").
- The 24B trains gently, because its first settings learned every fact and broke the model.
- On Windows, and anywhere llama.cpp was built without CUDA, GPU presets answer through the
  Python engine (PyTorch with CUDA), which the launcher installs.

Or download a release:

The latest release is on the
[Releases page](https://github.com/FMXExpress/Short-Long-Memory-LLM/releases):

| Platform | Download | Run |
|---|---|---|
| Windows x64 | `memagent-<ver>-windows-x64-setup.exe` (per-user, no admin) or the portable `.zip` | Start menu → memagent |
| Linux x64 | `memagent-<ver>-linux-x64.tar.gz` | `./memagent` (needs `libsqlite3` and OpenSSL 3, present on most distributions) |
| Linux x64, NVIDIA GPU | `memagent-<ver>-linux-x64-cuda.tar.gz` | `./run.sh` (answers and trains on the card; needs the NVIDIA driver 525+, the CUDA libraries are included) |

memagent is a console app: start it and type. The first start downloads the default model
(SmolLM2-135M, 270 MB) and the retrieval models; after that it works offline. Your memory lives
in `data/` next to the program.

Things to try:

```
My dog's name is Biscuit.              tell it facts in plain language
Where do I live?                       ask; answers use retrieved memory and the trained LoRAs
/sleep                                 bake everything into the deep LoRA now (it also happens on its own)
/ab What's my dog's name?              compare the base model with the memory LoRAs, no retrieval
From now on, answer in one sentence.   a standing instruction becomes a prior
/facts  /priors  /status  /help        see what it knows and why
```

## How it works

```
                    ┌─────────────────────────────── one turn ──────────────────────────────┐
  you ─────────────►│ extract facts & priors ─► retrieve memories (FTS5 + embeddings + rerank)│
                    │            │                         │                                │
                    │            ▼                         ▼                                │
                    │  persona + priors + memories + recent turns ─► base model + LoRAs ──► reply
                    └────────────┬──────────────────────────────────────────────────────────┘
                                 │ new facts (SQLite)
                                 ▼
          ┌──────────────── background trainer (llama.cpp, in-process) ────────────────┐
          │ fast run:   new facts + replay of old ─► fast LoRA, trained on the frozen deep │
          │ deep sleep: every fact ─► fresh deep LoRA (fast LoRA emptied)                  │
          │ eval gate:  recall from the weights alone + drift on held-out prompts           │
          └──────────────────────────────── hot swap between turns ───────────────────────┘
```

memagent keeps several kinds of memory, each doing what it is good at:

| Layer | Holds | Available | Mechanism |
|---|---|---|---|
| Conversation | recent turns, a rolling summary | immediately | the prompt |
| Retrieval | every message and fact | immediately | SQLite FTS5 + MiniLM embeddings, fused and reranked |
| Fast LoRA | facts since the last deep sleep | within minutes | trained whenever the trainer is free |
| Deep LoRA | every fact | after a deep sleep | rebuilt from the base model; corrections and `/forget` reach the weights |
| Priors | how you want to be answered | immediately | evidence-weighted standing instructions in every prompt |

Retrieval acts as a write-ahead log for the weights: a fact can be retrieved as soon as you say
it, and once an adapter answers its held-out probe question without any context, the fact is
marked *baked*. Training pauses between windows while you are being answered, so the background
work never slows the chat. A new adapter replaces the old one only if it passes the gate.

The full design, from the llama.cpp patch to the retrieval cascade, is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Features

### Long-term memory that trains itself

- Facts come from your messages: rules for the small model, plus LLM extraction on larger ones.
  They are deduplicated, and a new value for the same subject supersedes the old one ("My dog's
  name is Max").
- Training data is built from each fact from several sides (on the default profile:
  paraphrased questions, short and full answers, yes/no checks against other facts' values,
  fill in the blank), plus anchor examples of the model's own answers, which keep its general
  behaviour intact.
- Every adapter is gated on recall (each fact answered from the weights alone) and on drift
  (perplexity on held-out prompts against the model before training).
- `/rollback` restores the previous adapter; every adapter is kept on disk with a manifest.

### Priors: standing instructions

"From now on keep answers short", "never use emojis", "I prefer code examples" or
`/prior <text>` become priors. A prior's confidence is counted, not guessed: each message counts
once for or against it, so a remark is not re-weighed on every turn. Contradictory priors never
both reach the prompt, and a contradicted prior is deprecated rather than deleted. `/priors`
shows each one with its evidence, and `/prompt <text>` shows exactly what the model will see.

### Retrieval and the confabulation guard

Retrieval finds facts by meaning ("Where is my home?" finds "You live in Reykjavik"). The
cross-encoder that reranks the results also decides whether a new fact corrects an old one;
embeddings alone confuse "my sister's name" with "my brother's name". Fine-tuning makes a model
eager to answer from memory, so a licensing guard withholds an answer to "what's my horse's
name?" that uses the dog's name.

### Bodies: acting in the world (optional)

`[body] driver` connects memagent to something that acts. `/do <task>` shows the plan, and
`/confirm` runs it:

- **Drivers:** `mock`, `lerobot` (any [LeRobot](https://github.com/huggingface/lerobot) policy
  on a robot or in a simulator) and `process` (any program speaking a small JSON-lines protocol).
- **FineART-VLA:** it plans its own subtasks in language, and memagent checks each one against
  your rules (priors such as "never put the cast-iron pan in the dishwasher") and stops the
  robot on a match.
- **Memory:** completed steps such as "put the mug in the left cabinet" become facts.
- **Safety:** the rule check is a backstop, not a safety system. See
  [body/README.md](body/README.md).

### Model profiles

| Profile | Base model | Runs on |
|---|---|---|
| `tiny` (default) | SmolLM2-135M-Instruct | any CPU; a deep sleep over 30 facts takes minutes |
| `small` | Qwen2.5-1.5B-Instruct | one 8 GB+ GPU, or a patient CPU |
| `medium` | Qwen3-4B-Instruct (4-bit) | two 16–24 GB GPUs |
| `large` | Magistral-Small 24B (4-bit) | two 24 GB+ GPUs |
| `qwen3-14b` | Qwen3-14B (4-bit, 9 GB) | one 16 GB+ GPU; learns into its weights in-process (30 facts: 0.967 answered from the weights) |
| `gpt-oss-20b` | gpt-oss-20b (MXFP4, 12 GB) | one 16 GB+ GPU; learns in-process, attention only (10 facts: 0.75 from the weights) |
| `qwen3.8-27b` | Qwen3.8-27B (4-bit, 17 GB) | one 24 GB+ GPU; **chat-only**: memory is stored facts and messages, retrieved into each prompt (nothing can train its Gated DeltaNet layers yet) |
| `qwen3-0.6b` | Qwen3-0.6B (Q8_0, 0.8 GB) | any CPU, fastest (29 t/s on 4 cores); learns in-process (30 facts: 0.883 from the weights) |
| `lfm2-1.2b` | LFM2-1.2B (Q8_0, 1.2 GB) | any CPU; learns in-process (10 facts: 0.80 from the weights, drift 1.02) |
| `lfm3-1.7b` | LFM3-1.7B (Q4_K_M, 1.1 GB) | any CPU (21 t/s on 4 cores); learns in-process (10 facts: 0.85 from the weights, 1.00 with retrieval) |
| `gemma4-e2b` | Gemma 4 E2B (Q4_0, 2.8 GB) | slower on a CPU (11 t/s on 4 cores); learns in-process (10 facts: 0.90 from the weights, drift 0.95) |

Everything model-specific lives in `models.json`. Where chat and training run is `[gpu] layout`
(default `auto`: no card → `cpu`, one card → `one-gpu`, two → `two-gpus`):

| layout | chat | training |
|---|---|---|
| `cpu` | CPU | CPU |
| `chat-gpu-train-cpu` | GPU | CPU (small models: fast answers, the CPU trains in the background) |
| `chat-cpu-train-gpu` | CPU | GPU |
| `two-gpus` | GPU `gpu` | GPU `second` |
| `one-gpu` | GPU | the same GPU, waiting while a turn is answered |
| `one-gpu-idle` | GPU | the same GPU, only after `idle_min` minutes without a message (and inside `night`, when set); the chat model is unloaded so training has the whole card, and the next message stops the run |

`[consolidate] train_base = auto` trains on a full-precision copy of the model when it fits
on the training device, else on the model as loaded (on the chat card, the chat engine's own
copy). The startup line says what was chosen and why. `tiny` and `small` train in-process on
llama.cpp; `medium` and `large` still train with the Python trainer.

## Results

Measured on the default `tiny` model (135M parameters, CPU) with 30 facts and their held-out
questions, worded differently from anything in training. Methodology and every run are in
[bench/README.md](bench/README.md).

| | Accuracy, weights only | Accuracy, weights + retrieval | Confabulation, weights only |
|---|---|---|---|
| Base model | 0.00 | 0.37 (retrieval alone) | 0.00 |
| Memory LoRA (dev set) | **0.93** | **0.97** | 0.70 |
| Memory LoRA (held-out set) | **0.93** | **1.00** | 0.80 |

- **In the full agent** (retrieval, LoRAs and the guard together, measured end to end with the
  earlier training defaults), confabulation on never-stated questions is 0.20. The guard is
  what brings it down from the raw LoRA's 0.70–0.80.
- **Recall:** every fact passes the recall gate (1.00).
- **Drift:** held-out perplexity rises by 1.2–1.5×, well inside the 3.0 gate.
- **Training:** a deep sleep over 30 facts takes about 14 minutes on 4 CPU cores. Fast runs
  over a few new facts take a minute or two.

## Commands

| Command | |
|---|---|
| `/facts`, `/remember <text>`, `/forget <id>` | list, add or remove facts |
| `/recall <query>` | what retrieval would put in the prompt, with scores |
| `/dream [all]`, `/review`, `/keep <id>` | check facts against the messages they came from; list or restore the ones set aside |
| `/import <path or source> [--llm] [--dry-run]` | take what you said in other chats into memory: ChatGPT and claude.ai exports, Claude Code, Codex, Gemini CLI, Pi / OpenClaw, OpenCode (below) |
| `/ab <question>` | base model vs memory LoRAs, both without retrieval |
| `/sleep`, `/nap`, `/wait` | deep sleep now, fast run now, wait until memory has caught up |
| `/adapter`, `/rollback` | adapter history; go back to the previous deep LoRA |
| `/priors`, `/prior [--body] <text>`, `/prior drop\|restore <id>` | standing instructions |
| `/prompt <text>` | the exact system prompt a message would get |
| `/do <task>`, `/confirm`, `/cancel` | act through a body |
| `/status`, `/help`, `/quit` | |

### Importing other chats

`/import` reads what you told other assistants, and memory learns it as if you had told
memagent:

| Source | `/import ...` |
|---|---|
| ChatGPT | the `conversations.json` from Settings → Data controls → Export data (unzipped) |
| claude.ai | the `conversations.json` from Settings → Privacy → Export data (unzipped) |
| Claude Code | `claude-code` (reads `~/.claude/projects`), or one session's `.jsonl` |
| Codex CLI | `codex` (`~/.codex/sessions`) |
| Gemini CLI | `gemini` (`~/.gemini/tmp`) |
| OpenCode | `opencode` (`~/.local/share/opencode`) |
| Pi / OpenClaw | a session `.jsonl` |

What gets kept:
- **Your own turns only.** They go in the message log at the time you wrote them, so
  "yesterday" resolves to the right day. Facts come out of them the same way as from a
  chat. The assistants' replies are left out, and so are subagent transcripts and text a
  harness added (command echoes, reminders, environment blocks).
- **No secrets.** API keys, tokens, private keys and the value after "password" or "token"
  become `[redacted]`, and a turn that held one gives no fact.
- **Fast by default.** The rule extractor runs on every turn: 123 messages took 4 seconds.
  `--llm` also asks the model where the rules found nothing, at 5–8 s a message on
  `small`. `--dry-run` lists what would be read.
- **Each conversation once.** Running `/import codex` again picks up only new sessions.

The ChatGPT, Claude Code, Pi and OpenCode readers follow
[PasClaw](https://github.com/FMXExpress/PasClaw)'s importers.

Command-line options: `--profile <name>`, `--engine auto|llama|python`, `--data <dir>`,
`--config <ini>`, `--offline`, `--provision`. Everything else is in
[`memagent.ini`](memagent.ini), which documents each setting.

## Build from source

```sh
sudo apt install fpc cmake                 # Free Pascal 3.2+, CMake for llama.cpp
make                                       # build/memagent
make llama                                 # llama.cpp b11985 + the LoRA-training patch, and the tiny GGUF model
./build/memagent                           # first start downloads the retrieval models
```

Optional:
- `LLAMA_CUDA=1 make llama` builds the CUDA backend.
- `pip install -r trainer/requirements.txt` installs the Python trainer and engine, needed for
  the `medium` and `large` profiles.
- `make package VERSION=x.y.z` builds a release tarball. The Windows installer is built by
  [`.github/workflows/release.yml`](.github/workflows/release.yml).

## Tests

```sh
make test                  # Pascal and Python unit tests: extraction parity, retrieval, priors, TLS
make test-e2e-llama        # the real binary on CPU: recall, consolidations, corrections, /forget, /rollback
make test-tiers            # a deep sleep, then a fast LoRA trained on top of it; both answer from the weights
scripts/e2e_priors.sh      # priors picked up, confirmed, contradicted and kept consistent in the prompt
scripts/e2e_body.sh        # bodies: rules refuse a task and stop a subtask; finished steps become facts
```

Every release is built and smoke-tested on Linux and Windows before it is published. The
smoke test downloads the models, chats, trains, and hot-swaps.

## Project layout

```
memagent/        the agent, in Free Pascal (Mem.Agent, Mem.Store, Mem.Train, Mem.Engine.Llama, Mem.Priors, Mem.Body, ...)
trainer/         Python trainer and engine (optional), GGUF export
body/            body drivers: LeRobot (FineART-VLA and other policies)
bench/           benchmarks and their results
packaging/       llama.cpp training patch, Inno Setup installer
scripts/         setup, packaging and end-to-end tests
docs/            ARCHITECTURE.md and a sample transcript
legacy/          the earlier Python prototypes
```

## Status and limitations

memagent is a research project that works end to end. It is not a finished product.

- **Model size:** the default 135M model is chosen so it runs anywhere. It learns facts
  reliably but follows instructions loosely, so priors and fluent answers pay off from `small`
  up.
- **Untested profiles:** the `medium` and `large` profiles have not been tuned on real
  hardware yet.
- **Confabulation:** a fine-tuned model still answers some never-stated questions with stored
  values. The guard catches most of this in the agent.
- **Bodies:** the LeRobot driver follows LeRobot's documented API but has not yet run on a
  robot. FineART-VLA was trained on one robot (ALOHA), needs a CUDA GPU, and its weights are
  under the Gemma license.

## Background

memagent's design borrows from recent work on continual learning in language models:

- Behrouz et al., *Language Models Need Sleep* ([arXiv 2606.03979](https://arxiv.org/abs/2606.03979)):
  wake/sleep consolidation into low-rank parameters.
- O'Neill et al., *Can a Language Model Continually Learn Facts in its Weights?*
  ([arXiv 2607.11020](https://arxiv.org/abs/2607.11020)): varied examples, and sequential LoRA
  writes.
- Hatakeyama-Sato et al., *Teaching Specific Scientific Knowledge into Large Language Models
  through Additional Training* ([arXiv 2312.03360](https://arxiv.org/abs/2312.03360)): text
  augmentation and LoRA target layers.

## License

memagent is released under the [MIT License](LICENSE). It builds on
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), ONNX Runtime (MIT), SQLite (public
domain) and OpenSSL (Apache 2.0). The optional LeRobot driver uses
[LeRobot](https://github.com/huggingface/lerobot) (Apache 2.0). Model weights are downloaded on
first use and come under their own licenses; see each model's page.
