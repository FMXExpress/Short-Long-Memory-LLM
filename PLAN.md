# Plan: "memagent", a Free Pascal Agent Demo with Short- and Long-Term Memory

## Status (2026-10-28)

M1–M5 are implemented and tested end to end on the `tiny` profile (SmolLM2-135M, CPU):
`make test` runs the unit tests and the worker protocol test, and `make test-e2e` runs the real
binary through two consolidations, a correction, `/forget` and `/rollback`.

Changes from the plan below:

- **Fact extraction:** rule-based extraction always runs. LLM extraction, conversation
  summaries and tool calls are switched on per profile (`extract`, `summarize`, `tools` in
  `models.json`), because a 135M model cannot produce reliable JSON.
- **Training:** a plain PyTorch loop on PEFT instead of TRL, so there are fewer moving
  parts and fewer version pins.
- **Drift check:** anchors are split into a trained-on set and a held-out set, and the gate
  checks perplexity on the held-out set only.
- **Message RAG:** skips any message whose fact is already baked, so `/ab` really does show
  recall coming from the weights.

Added after M5:

- **Local retrieval in Pascal:** a MiniLM embedder and an ms-marco cross-encoder reranker, run
  through ONNX Runtime. They came from PasClaw's localvector and memory units, now part of
  `memagent/` as `Mem.*`.
- **Hybrid search:** FTS5 and cosine results are fused with RRF, then reranked.
- **Fact matching:** deciding that a new fact is about an existing subject needs a cosine shortlist
  plus confirmation from the cross-encoder.
- **Verifying HTTPS downloader:** `Mem.Https` checks certificates and hostnames, tunnels through a
  proxy and loads OpenSSL 3. FPC 3.2.2's stock handler does none of these.

Full test pass from a fresh clone, plus fixes found by it (2026-10-29):

- **What ran:** a fresh clone, empty caches, `make`, `--provision`, the unit, retrieval and TLS suites,
  e2e, the resilience test, and the `small` profile (Qwen2.5-1.5B) on CPU for chat and training.
- **Fixed: a rollback was undone on restart.** A leftover `ready.json` re-promoted the adapter
  the user had rolled back from. `ready.json` is now cleared once it is handled, and adapters
  already in the history are never re-promoted.
- **Fixed: non-ASCII text was corrupted.** Everything that passed through fpjson was stored as
  Latin-1 ("Zoë" became invalid UTF-8 in memory.db). `Mem.Utf8` and `cwstring` now make UTF-8
  the process codepage.
- **Fixed: LLM fact extraction.** Pretty-printed JSON was dropped, and values the model invented
  were accepted ("The current time is 14:30"). Values must now occur in the user's own words.
- **Fixed: tool calls.** Bare-JSON tool calls without `<tool>` tags were ignored, a missing
  `args` field would have been dereferenced as nil, and `remember` claimed "stored" even when
  nothing was stored.
- **Fixed: duplicate facts.** A fact from the user's message and the same fact from a tool call
  were both stored. Facts are now taken from the user's message before the reply, and a repeat
  (same value, text cosine ≥ 0.9) is skipped.
- **Fixed: recall by relation.** Facts are now embedded and reranked as "subject: text", so
  "where does my sister live?" finds "Ana moved to Lisbon".
- **Fixed: the summary prompt.** The instructions moved into a system turn; small models had
  been echoing them back as headings.
- **Fixed: `small` training.** It ran out of memory at 14 GB on CPU, so gradient checkpointing
  is now enabled for `small`, `medium` and `large` (peak 8.7 GB). Its defaults overfit and
  failed the drift gate (1.75×), so they are now 4 epochs at lr 3e-4: recall 1.00, drift 1.45×.
- **Fixed: `Mem.Https` ignored `NO_PROXY`.**

Not yet exercised: the CUDA and 4-bit paths and the `small`/`medium`/`large` profiles, because
this build environment has no GPU. The code paths exist (`memlib.load_base`), but they need a
first run on real GPUs.

## 1. Goal

A **small, standalone demo**: a Free Pascal console agent whose memory has
two tiers.

- **Short-term:** the chat context, plus simple RAG over past messages and
  facts.
- **Long-term:** facts *baked into a LoRA*, retrained in the background on
  GPU 1 and **hot-swapped** into the model running on GPU 0.

**v1 split:**

- **FPC owns the agent:** the loop, prompts, memory store, RAG, fact
  bookkeeping, scheduling and the UI.
- **Python owns the tensors**, and FPC just launches it as processes:
  - a **long-lived inference worker** (`worker.py`, GPU 0). FPC starts it
    once with `TProcess` and sends JSON lines over stdin/stdout. The model
    stays loaded, and swapping in a new LoRA is a single PEFT call.
  - a **one-shot trainer** (`consolidate.py`, GPU 1). FPC runs it like a
    shell command and reads its exit code.

A Python process per turn would reload the model every time, which takes
10–60 s, so the inference worker is kept running. Both scripts are mostly
a refactor of the existing `v2/cragchat.py`.

Later, the worker can be replaced by in-process llama.cpp (localpal style,
see §9) **without touching the agent**, because the agent only talks to an
`IEngine` interface.

```
 ┌─────────────── memagent (FPC console app) ───────────────────────┐
 │ agent loop · priors · short-term context · SQLite FTS5 RAG       │
 │ fact table · scheduler · /commands                               │
 └──────┬──────────── stdin/stdout JSON lines ─────────┬────────────┘
        │ TProcess (long-lived)                          │ TProcess (one-shot, exit code)
 ┌──────▼──────── worker.py  (GPU 0) ───────┐   ┌──────▼──── consolidate.py (GPU 1) ──────┐
 │ HF model 4-bit + PEFT                    │   │ facts → paraphrase dataset → QLoRA       │
 │ generate (streamed) · extract_facts      │   │ → recall eval gate → adapters/mem-<ts>/  │
 │ load_adapter / set_adapter / no_adapter  │◄──┤ manifest.json                            │
 └──────────────────────────────────────────┘   └──────────────────────────────────────────┘
```

## 2. Memory tiers

| Tier | Implementation | Changes |
|---|---|---|
| **Priors** | base weights + `priors.md` system prompt | never |
| **Short-term** | the last N turns kept verbatim in context; older turns folded into a running summary | every turn |
| **RAG** | SQLite FTS5 over `messages` and `facts`; top-k hits injected as a `Relevant memory:` block | every turn |
| **Long-term (LoRA)** | the PEFT adapter active in the worker | hourly or nightly, or on `/sleep` |

**Core idea: RAG is the write-ahead log for the LoRA.** A new fact is
usable straight away through RAG. Once a trained adapter answers the fact's
probe question *with no RAG context*, the fact gets
`baked_in = '<adapter id>'` and is no longer injected (it stays
searchable). The demo shows facts moving from the prompt into the weights.

## 3. Worker protocol (FPC ↔ `worker.py`)

The protocol is one JSON object per line. Requests go to stdin; replies and
streamed tokens come back on stdout; logs go to stderr, which FPC tails into
`worker.log`. stdout carries **protocol only**, so the worker redirects
`print` and library chatter to stderr at startup.

```jsonc
// → requests
{"id":1,"op":"generate","messages":[{"role":"system","content":"..."},...],
 "max_new_tokens":512,"temperature":0.7,"adapter":true}
{"id":2,"op":"extract_facts","text":"<user+assistant turn>"}
{"id":3,"op":"load_adapter","path":"adapters/mem-20261028T1400","name":"mem-20261028T1400"}
{"id":4,"op":"unload_adapter","name":"mem-20261028T0300"}
{"id":5,"op":"status"}
{"id":6,"op":"shutdown"}

// ← replies
{"id":1,"type":"token","text":"Hel"}          // streamed, zero or more
{"id":1,"type":"done","text":"Hello ...","usage":{"in":812,"out":40}}
{"id":2,"type":"done","facts":[{"text":"User's dog is named Rex","kind":"user","confidence":0.9}]}
{"id":3,"type":"done","active":"mem-20261028T1400"}
{"id":9,"type":"error","message":"..."}
```

Worker internals (refactored from `v2/cragchat.py`):

- loads `BASE_MODEL_ID` in 4-bit with `CUDA_VISIBLE_DEVICES=0`, wrapped in
  `PeftModel` once the first adapter exists
- applies the chat template with `tokenizer.apply_chat_template`
- streams with `TextIteratorStreamer`, the same as v2
- **hot swap:** `model.load_adapter(path, adapter_name=name)`, then
  `model.set_adapter(name)`, then `model.delete_adapter(old)`
- **`"adapter": false`** (used by `/nolora` and the eval gate) generates
  inside `with model.disable_adapter():`
- requests are handled one at a time, so a swap can never happen in the
  middle of a generation

## 4. The agent loop (FPC)

For each user turn:

1. **Promote check:** if `adapters/ready.json` names an adapter newer than
   the active one, send `load_adapter`, then `unload_adapter` for the old
   one, and set `baked_in` from the manifest.
2. **Retrieve:** an FTS5 query built from the user message, against facts
   that are not yet baked and against messages; take the top k.
3. **Build messages:** priors, summary, retrieved memory, recent turns,
   then the user message.
4. **Generate:** send `generate` and print `token` events as they arrive.
5. **Tools (minimal):** if the reply contains
   `<tool>{"name":...,"args":...}</tool>`, run the tool, append its result,
   and loop (at most 4 times). Demo tools: `remember(fact)`,
   `memory_search(query)`, `now()`.
6. **Persist:** write the turn to `messages`.
7. **Extract facts:** send `extract_facts`, insert the results while
   skipping exact and FTS-near duplicates, and drop rows below the
   confidence threshold.
8. **Compact:** if the context is over budget, summarise the oldest turns
   (a `generate` call) into `summary`.

Console commands: `/facts`, `/forget <id>`, `/sleep` (consolidate now),
`/adapter` (show the current one and its history), `/rollback`, `/nolora`
(ask the same question with the adapter off and RAG off, for an A/B),
`/quit`.

## 5. Consolidation job (`consolidate.py`, GPU 1)

memagent runs it as a plain command and waits for it on a background
thread:

```
CUDA_VISIBLE_DEVICES=1 python trainer/consolidate.py \
    --db memory.db --out adapters/ --base unsloth/Qwen3-4B-Instruct-2507-bnb-4bit
```

The trigger is a timer (hourly, when at least N new facts exist), a nightly
run, or `/sleep`. Only one run can happen at a time, enforced with a lock
file. Steps:

1. **Dataset:** all active facts (not superseded or forgotten). For each
   fact, K Q→A paraphrases plus one held-out probe paraphrase, reusing
   `augment_training_data_with_fact_based_scaling` and
   `generate_question_variations` from `v2/cragchat.py`. A fixed slice of
   generic chat data is mixed in to protect the priors.
2. **Train:** QLoRA with PEFT/TRL using v2's hyperparameters (r = 64,
   α = 16, all projection layers, lr 5e-4). For the demo, always retrain
   **from base on the full fact set**. With a small model and a few hundred
   facts this takes minutes, and it makes `/forget` and corrections work
   with no un-training logic. Incremental training with a replay buffer is
   a later optimisation.
3. **Eval gate** (on GPU 1, with the freshly trained model still in
   memory): run the probe questions with no context, then check recall and
   do a quick perplexity check on the generic slice.
4. **Publish:** save `adapters/mem-<ts>/` (the PEFT folder) plus
   `manifest.json` (base id, fact IDs that passed, metrics), then
   atomically rename `ready.json` to point at it.

Exit codes: **0** means promoted, **2** means rejected by the gate (the
current adapter stays), **1** means error. The last stdout line is a JSON
summary that memagent shows in the console.

## 6. FPC side of the process plumbing

```pascal
(* Long-lived worker: one TProcess, pipes kept open for the whole run. *)
FWorker := TProcess.Create(nil);
FWorker.Executable := FConfig.PythonExe;
FWorker.Parameters.Add('trainer/worker.py');
FWorker.Environment.Add('CUDA_VISIBLE_DEVICES=0');
FWorker.Options := [poUsePipes];
FWorker.Execute;

(* Request = one JSON line on stdin; replies are read line by line from
   Output until a "done" or "error" with the matching id. A reader thread
   drains Stderr into worker.log so the pipe never fills and blocks. *)
```

- JSON uses `fpjson` + `jsonparser`, which ship with FPC.
- Line framing is done with a small buffered reader over `FWorker.Output`
  (`ReadLn` cannot be used on a `TInputPipeStream`).
- **Restarting the worker** (it crashed or ran out of memory): memagent
  restarts it and replays `load_adapter` for the active adapter. A turn
  that fails is reported, and the agent keeps running.
- The trainer is launched as a one-shot `TProcess` with
  `poWaitOnExit`, on a `TThread` so chat stays responsive.

## 7. Layout

```
memagent/
  memagent.lpr             entry point + REPL
  Mem.Engine.pas           IEngine interface + TPyWorkerEngine (TProcess / JSON lines)
  Mem.Store.pas            sqldb SQLite: messages, facts (FTS5), summary, adapters
  Mem.Agent.pas            turn loop (§4), prompt assembly, tool parsing
  Mem.Consolidate.pas      scheduler, lock, one-shot trainer launch, ready.json watch
  Mem.Config.pas           memagent.ini (python path, model id, thresholds, cadence)
  Makefile                 fpc -Mobjfpc / -Mdelphi; Linux first
trainer/
  worker.py                §3, inference worker (GPU 0)
  consolidate.py           §5, training job (GPU 1)
  memlib.py                code shared by both: model loading, chat template, fact scaling (from v2)
  requirements.txt         torch, transformers, peft, trl, bitsandbytes, datasets
legacy/                    current Python prototypes (cragchat, memos, ...)
```

`memory.db` schema (fields kept close to PasClaw's fact record so a later
port is mechanical):

```
messages(id, session, role, content, created_at)          + FTS5
facts(id, text, kind, confidence, source_msg, created_at,
      superseded, forgotten, baked_in)                     + FTS5
summary(session, text, upto_msg)
adapters(id, path, created_at, n_facts, recall, ppl, status)
```

SQLite is opened in WAL mode. memagent is the only writer; `consolidate.py`
only reads `facts`, and its results come back through the manifest.

## 8. Milestones

| # | Deliverable | Demo moment |
|---|---|---|
| M1 | `worker.py` + `TPyWorkerEngine`: streamed chat from FPC | FPC console chatting with the GPU 0 model |
| M2 | `Mem.Store` + short-term context + summary + FTS5 RAG | it recalls something said 50 turns ago through RAG |
| M3 | `extract_facts` + `/facts` `/forget` | facts appear as you chat |
| M4 | `consolidate.py` + `/sleep` + hot swap | `/nolora` versus LoRA on a fact with RAG turned off |
| M5 | automatic schedule, eval gate, `baked_in`, `/rollback`, worker restart | leave it running; facts migrate from the prompt into the weights |

## 9. Later

- **No Python at inference time:** add `TLlamaEngine` behind `IEngine`, a
  thin FPC binding to a pinned `llama.h` (localpal's design, but its
  vendored llama-cpp-delphi is Delphi-only and uses the removed
  `llama_lora_adapter_*` names; current names are `llama_adapter_lora_init`
  and `llama_set_adapters_lora`). The trainer then also runs
  `convert_lora_to_gguf.py`. The agent does not change.
- **PasClaw:** `IEngine` becomes a local provider, `facts` maps to
  `PasClaw.Memory.Facts` (plus a `baked_in` column), and consolidation
  becomes a cron skill next to `@maintain`.

## 10. Open decisions

1. **Demo model.** Recommendation: a **Qwen3-4B-class instruct model** in
   4-bit, so a consolidation run takes minutes. Magistral 24B from v2 also
   works with 24 GB or more on each GPU, but every run is far slower.
2. **Platform.** Linux first, since FPC, CUDA and bitsandbytes are simplest
   there.
3. **Default cadence.** `/sleep` on demand, hourly when there are at least
   10 new facts, and a nightly rebuild.
