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

Download the latest release from the
[Releases page](https://github.com/FMXExpress/Short-Long-Memory-LLM/releases):

| Platform | Download | Run |
|---|---|---|
| Windows x64 | `memagent-<ver>-windows-x64-setup.exe` (per-user, no admin) or the portable `.zip` | Start menu → memagent |
| Linux x64 | `memagent-<ver>-linux-x64.tar.gz` | `./memagent` (needs `libsqlite3` and OpenSSL 3, present on most distributions) |

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

Everything model-specific lives in `models.json`. With two GPUs, `[gpu] infer=0` and
`[gpu] train=1` keep inference and training on separate devices. `tiny` and `small` train
in-process on llama.cpp; `medium` and `large` still train with the Python trainer.

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
| `/ab <question>` | base model vs memory LoRAs, both without retrieval |
| `/sleep`, `/nap`, `/wait` | deep sleep now, fast run now, wait until memory has caught up |
| `/adapter`, `/rollback` | adapter history; go back to the previous deep LoRA |
| `/priors`, `/prior [--body] <text>`, `/prior drop\|restore <id>` | standing instructions |
| `/prompt <text>` | the exact system prompt a message would get |
| `/do <task>`, `/confirm`, `/cancel` | act through a body |
| `/status`, `/help`, `/quit` | |

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
