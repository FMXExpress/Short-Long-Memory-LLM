# Plan: Always-On Agent with Baked-In LoRA Memory (Free Pascal)

## 1. Goal

Turn this repo's Python RAG + LoRA chatbot (`v2/cragchat.py`) into an
**always-on agent** that runs on one machine with two GPUs:

| GPU | Role | Runs |
|---|---|---|
| GPU 0 | **Inference**, always on | the base model plus the current memory LoRA, served to the agent |
| GPU 1 | **Consolidation** | trains the next memory LoRA from recent facts and conversations, then hands it off for a hot swap |

The agent loop, tools, sessions, RAG and scheduling are written in **Free
Pascal**. Python is kept only where nothing comparable exists outside it:
LoRA training (PEFT/Unsloth) and a LoRA-capable inference server (vLLM).

## 2. Memory model

Memory is split into four tiers, ordered by how quickly each one changes:

| Tier | What it is | Lifetime | Where it lives |
|---|---|---|---|
| **Priors** | base weights, system prompt, persona (`SOUL.md`, `USER.md`) | fixed | GPU 0 weights and the prompt |
| **Short-term** | the live context window, compacted into summaries as it fills | minutes to hours | the PasClaw session JSON |
| **Mid-term (RAG)** | distilled facts and session transcripts, searched with FTS5 BM25 plus vectors | until baked, then kept as a backup | `facts.db`, `sessions/.search.db` |
| **Long-term (parametric)** | the facts trained into a LoRA adapter | until the next rebuild | `adapters/mem-<timestamp>/` |

**Core idea: RAG works as the write-ahead log for the LoRA.** A new fact
goes into `facts.db` straight away and reaches the model through retrieval
and prompt injection. Once a trained adapter can be shown to know the fact
(it passes a probe, see §5.4), the fact is marked `baked_in = <adapter id>`
and stops using space in the prompt. It stays searchable, so a regression
can fall back to it. Prompt use therefore stays bounded while total
knowledge grows.

## 3. Choosing the base: PasClaw or localpal

| | **PasClaw** (FMXExpress/PasClaw) | **localpal** (FMXExpress/localpal) |
|---|---|---|
| Compiler | **FPC 3.2+** (`make`); also Delphi | Delphi only (FireDAC, Win64 DLLs) |
| Agent loop and tools | full: tool loop, MCP, subagents, hooks, steering | none; chat only |
| Memory | FTS5 plus vector `memory_search`, `session_search`, **`facts.db` fact store with distillation and dedup**, nightly `maintain` | SQLite chat history |
| Local model | `vllm`, `ollama`, `lmstudio` provider entries (OpenAI protocol) | in-process llama.cpp through llama-cpp-delphi |
| Scheduling | `cron` with catch-up and the reserved `@maintain` skill | none |

**Decision: build on PasClaw.** It already compiles with FPC, and it
already has most of the short-term and mid-term tiers: compaction, the
distiller that writes `TFact` rows (`PasClaw.Memory.Distill`), the
SQLite fact store with supersede and expiry (`PasClaw.Memory.Facts`),
hybrid retrieval, and cron. localpal is useful only as a reference for
running llama.cpp in-process (see §4, option B), and its vendored bindings
target Delphi and Windows.

Approach: run PasClaw as the agent. Upstream a small set of patches to it
(§6.2) and keep everything specific to memory consolidation in this repo.

## 4. Inference and hot swap (GPU 0)

### Option A: vLLM with runtime LoRA loading (recommended for v1)

```sh
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
CUDA_VISIBLE_DEVICES=0 vllm serve <BASE_MODEL> \
  --enable-lora --max-lora-rank 64 --max-loras 2 --port 8000
```

Hot swap, with no restart and no dropped requests:

1. `POST /v1/load_lora_adapter {"lora_name":"mem-20261028T1400","lora_path":"/srv/adapters/mem-20261028T1400"}`
2. Switch the model name PasClaw requests to `mem-20261028T1400`, taking
   effect from the next turn. A turn already in flight finishes on the old
   adapter.
3. After a grace period: `POST /v1/unload_lora_adapter {"lora_name":"mem-<previous>"}`

`--max-loras 2` keeps the old and new adapters resident together, which
makes the swap atomic from the agent's side. Rolling back is the same
model-name switch in reverse.

### Option B: llama.cpp in-process from FPC (v2 and later, the pure-Pascal goal)

llama.cpp's C API loads adapters at runtime (`llama_adapter_lora_init`,
then attaches or detaches one on a context). An FPC unit can bind these
calls directly. That removes Python from serving, but each adapter must
first be converted from PEFT to GGUF (`convert_lora_to_gguf.py`, which
runs in Python on the training side, so this is acceptable). `llama-server`
on its own can only rescale adapters that were preloaded at startup; it
cannot load a new file, so it does not meet the hot-swap requirement.

Keep the agent unaware of which option is in use: it only ever sees an
OpenAI-compatible endpoint and a model name. Option B can then replace
Option A without changing the agent.

## 5. Consolidation pipeline (GPU 1)

### 5.1 Cadence

| Job | When | What |
|---|---|---|
| **Incremental** | hourly, and only when at least N new facts exist | continue training the current adapter on the **delta** (facts not yet baked) plus a **replay sample** of baked facts, to limit forgetting |
| **Rebuild** | nightly | train a **fresh adapter from base** on the full active fact set. This is the only clean way to apply deletions, supersessions and expiries, because a LoRA cannot be reliably un-trained one fact at a time |

### 5.2 Dataset builder

Inputs: active rows in `facts.db` (`superseded = 0` and not expired), plus
selected session excerpts. This reuses the existing logic in
`v2/cragchat.py`:

- `augment_training_data_with_fact_based_scaling` and
  `generate_question_variations`: create K paraphrased Q→A pairs per fact
  (the "fact-based scaling" from the knowledge-injection paper already
  cited in v2)
- each superseded fact adds an *update* example ("X used to be A; it is
  now B") so the newer value wins
- a fixed **anchor set** of general instruction data, around 10–20% of
  each batch, to protect the priors

The builder writes `datasets/<run-id>.jsonl` using the model's own chat
template, so train-time and serve-time prompts match.

### 5.3 Trainer

`trainer/train_memory_lora.py` is a QLoRA job on GPU 1, using Unsloth or
PEFT and TRL with hyperparameters taken from v2 (r = 64, α = 16, all
projection layers, lr 5e-4). It writes
`adapters/mem-<ts>/{adapter_model.safetensors, adapter_config.json, manifest.json}`.
The manifest records the base model hash, the fact IDs covered, the
dataset hash and the parent adapter.

### 5.4 Eval gate (nothing is promoted without passing)

- **Recall probe**: for each newly covered fact, ask a held-out paraphrase
  **with no RAG context**, then score the answer by string or embedding
  match. Facts that pass are marked `baked_in`.
- **Regression**: perplexity on the anchor set, plus a small fixed
  benchmark. Reject the adapter if it worsens either by more than X%.
- **Old-fact retention**: a sample of previously baked facts. If retention
  drops below its threshold, reject the adapter and schedule a rebuild.

The probe runs on GPU 1: vLLM with the candidate adapter, or HF
`generate`. GPU 0 is never used for evaluation.

### 5.5 Promote

The promoter loads the adapter into GPU 0's server (§4), switches the
model name, updates `facts.baked_in` for the facts that passed, and appends
to `adapters/history.jsonl`. It keeps the last N adapters for rollback.

## 6. Components to build

### 6.1 In this repo

```
memd/                    FPC: consolidation orchestrator daemon (no ML math)
  memd.lpr
  Memd.Scheduler.pas     hourly or nightly decision; threshold on new facts
  Memd.Dataset.pas       reads facts.db + sessions, writes dataset JSONL
  Memd.Trainer.pas       launches the trainer process on GPU 1, tails its log, enforces a timeout
  Memd.Eval.pas          runs the recall, regression and retention probes
  Memd.Swap.pas          vLLM load/unload calls; sets PasClaw's active model; rollback
  Memd.Registry.pas      adapters/history.jsonl and manifests
trainer/                 Python (the only Python left)
  train_memory_lora.py   CLI: --dataset --base --parent --out ; exit code reports the outcome
  convert_to_gguf.sh     for Option B
  requirements.txt
server/
  vllm-gpu0.service      systemd unit for Option A
  pasclaw.service        the agent (gateway + channels)
  memd.service
legacy/                  existing Python scripts kept for reference (cragchat, memos, ...)
```

Paraphrase generation for 5.2 calls the **GPU 0 model through its HTTP
endpoint** from Pascal, so the dataset builder does not need Python.

### 6.2 Patches to upstream to PasClaw (kept small, all general-purpose)

1. A `facts.baked_in TEXT` column. Injection into the system prompt skips
   baked facts; `memory_search` still returns them.
2. **A hot model switch:** a `POST /v1/config/model` endpoint, or a watched
   `active_model` file, so `memd` can retarget the agent without restarting
   it.
3. A `memory_distill_enabled` default of on in the profile used for this
   agent. Configuration only, no code change.
4. Optionally, a `pasclaw facts export --since` command, so `memd` never
   has to know the database schema.

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | Pick the base model and confirm VRAM fit on both GPUs (see §9) | a bf16 or AWQ build serves on GPU 0 **and** QLoRA trains on GPU 1 |
| M1 | PasClaw (built with FPC) talks to vLLM on GPU 0; distillation fills `facts.db` | a chat produces fact rows |
| M2 | `trainer/train_memory_lora.py` plus the dataset builder (port v2's fact scaling) | a manually triggered run produces an adapter |
| M3 | Eval gate and manual hot swap (`memd swap <id>`) | a fact that RAG has never seen is answered with the adapter loaded |
| M4 | `memd` daemon: nightly rebuild; `baked_in` bookkeeping; rollback | runs unattended for a week |
| M5 | Hourly incremental runs with a replay buffer | retention stays above threshold over 7 days |
| M6 (optional) | Option B: FPC llama.cpp binding with in-process adapter swap; vLLM removed | Python is used only on GPU 1 |

## 8. Risks and mitigations

- **Hallucinated or wrong facts get trained in.** Train only on distilled
  facts that have a confidence score, never on raw model output. Gate on
  confidence, and optionally require user confirmation for facts of kind
  `user`. A nightly rebuild from the fact store means a corrected row is
  fixed in the weights by the next morning.
- **Catastrophic forgetting.** Mitigated by the replay buffer, the anchor
  set, the retention gate and the nightly rebuild from base.
- **The adapter learns style rather than facts.** Mitigated by paraphrase
  scaling, probes run without RAG context, and keeping the fact in RAG
  until its probe passes.
- **Base model mismatch between training (nf4) and serving (bf16/AWQ).**
  Use the same checkpoint for both, and record the base hash in the
  manifest. The swapper refuses an adapter whose hash does not match.
- **A training run hangs or runs out of memory.** `memd` enforces a timeout
  and an exit-code contract. GPU 0 is never affected, because the agent
  keeps the previous adapter.

## 9. Open decisions

1. **Base model and VRAM.** v2 uses Magistral-Small 24B. In bf16 that is
   about 48 GB, which does not fit a single 24 GB card. Options: a 24 GB
   card serving an AWQ or GPTQ build of 24B; or an 8B–14B model (e.g.
   Qwen3) that fits bf16 serving on GPU 0 and QLoRA on GPU 1. The VRAM on
   each GPU decides this.
2. **Hourly incremental runs: v1 or later?** Recommendation: nightly
   rebuild only for v1 (M4), then add hourly runs (M5) once the eval gate
   has been shown to work.
3. **Which channel the agent runs on** (CLI, Telegram, gateway web UI,
   ...). This does not affect the memory design.
