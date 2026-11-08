# Benchmarks

| Benchmark | Command | Measures |
|---|---|---|
| memory | `python3 bench/bench_memory.py --profile tiny [--models M] [--facts F] --out R` | trains a memory LoRA on the facts with the production trainer, then asks each fact's held-out natural questions under four conditions: base model, base model + the fact in the prompt (perfect retrieval), LoRA, LoRA + fact. Also measures interference (another fact's value in the answer) and confabulation: questions about things never stated, with and without the look-alike fact a retriever would return ("near"). |
| retrieval | `build/bench_retrieval cache bench/facts.json [filler_facts] [filler_msgs]` (`BENCH_SWEEP=1` for threshold sweeps, `BENCH_MODE=name` for a single configuration) | the production RAG pipeline (`Mem.Recall`) in several configurations. Reports hit@1, hit@3 (the right fact among the `rag_k`=3 injected), MRR, wrong@1, noise (unknown questions that still get a fact injected) and latency per stage. |
| agent latency | `MEMAGENT_TIMING=1 build/memagent` | per-turn stages: recall (embed / search / rerank), fact extraction, time to first token, generation, tools, summary. `/status` shows the averages. |

Data: `facts.json` is the **dev** set, used for tuning. `facts_heldout.json` is the **test** set:
written before any tuning and used only to verify. Neither set's questions appear among the
trainer's question templates (checked), and the augmentation word pools in `trainer/memlib.py`
share no words with either set's unknown questions.

## Results log

Tiny profile (SmolLM2-135M), CPU, 30 dev facts, 60 held-out questions, 20 unknown questions.

| run | LoRA | LoRA+RAG | base+RAG | interference (LoRA) | confab. LoRA | confab. LoRA+near | drift | train s |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.783 | 0.933 | 0.367 | 0.067 | 0.70 | 0.80 | 1.78 | 412 |
| E1: names + negatives 0.5 + RAFT 0.5 | 0.617 | 0.917 | 0.367 | 0.033 | 0.65 | 0.73 | 1.42 | 504 |
| E2: names only | 0.767 | 0.900 | 0.367 | 0.100 | 0.60 | 0.73 | 1.50 | 408 |

E1 finding: at 135M parameters, the abstention examples teach "unfamiliar wording → abstain"
rather than "unknown subject → abstain". The LoRA refused 16 of 60 questions about facts it knew,
and it still confabulated whenever the wording looked familiar ("What's my horse's name?" →
"Paint"). This matches Gekhman et al. (EMNLP'24): fine-tuning on new knowledge raises
hallucination, and a model's sense of what it knows is hard to train. Negatives and RAFT stay off
for `tiny`.

E2 finding: the generic name-question shapes give no accuracy gain on their own (0.767 vs 0.783,
within run-to-run noise), so they stay off.

Confabulation is the open problem: with the LoRA, 60-70% of questions about things never stated
get a stored value as the answer, and 73-80% when a look-alike fact is also in the prompt. The
base model alone confabulates 0% (it refuses or rambles). Supervised examples did not fix this
at 135M (E1). The next candidates are retrieval-side filtering, or preference training (DPO) on
the larger profiles.

Prompt experiment H1 (baseline adapter, evaluation only): the memory header "use an item only
if it answers the question, otherwise say you haven't been told" gave LoRA+RAG 0.95 (+0.02,
noise) and confabulation with a look-alike present of 0.87 (worse than 0.80). The 135M model
does not follow the instruction, so the header is unchanged.

## Other findings from this round

- **Extraction:** the rule extractor's value accuracy went from 0.77 to 1.00 on dev and is 1.00
  on held-out. The fixes strip leading "a/on/called" from values and stop sentence splits at
  "Dr.".
- **RAG policy:** LoRA+RAG beats the LoRA alone by 15 points, and retrieval is capped at `rag_k`
  either way, so baked facts now stay retrievable (`rag_include_baked=1`).
- **Retrieval thresholds (applied):** re-swept with the 20 unknown questions. The new defaults
  are a cosine floor of 0.25 (was 0.35) and ms-marco `rerank_min` 1e-4 (was 0.005). hit@3 goes
  from about 0.75 to 0.967 on dev and 0.975 on held-out; noise goes from about 0.45 to 0.85.
  Recall wins because missing the fact costs 15 points of accuracy on every known question,
  while a look-alike in the prompt adds about 10 points of confabulation, and only on questions
  about unknown things.
- **Reranker batching:** the float model (ms-marco) scores in one padded session run and matches
  per-pair scores exactly. The int8 model (bge) scores per pair, because its dynamic activation
  quantization makes batched scores depend on their batch-mates (up to 0.17 apart).

## Profiling round (2026-10-29)

**Retrieval at scale** (`bench_retrieval cache bench/facts.json <facts> <messages>`, MiniLM +
ms-marco):

| stored | before | after |
|---|---|---|
| 30 facts | 23 ms/query | 26 ms/query |
| +500 facts, +2,500 messages | **1,210 ms**/query (search 1,187) | 8 ms search |
| +2,000 facts, +10,000 messages | — | 65 ms/query (embed 8, search 36, rerank 21) |

The cause was the "skip messages whose fact is baked" correlated subquery on `facts.source_msg`,
which had no index. gprof showed only about 0.5 s of CPU in our own code; the rest was inside
SQLite. Fixes: an index on `facts(source_msg)`, plus a top-k selection in place of the full
insertion sort. Quality numbers are unchanged. Search now grows by about 3 ms per 1,000 stored
rows; an in-memory vector cache is the next step if that ever matters.

**Live turn latency** (`MEMAGENT_TIMING=1`, CPU, 5 scripted turns):

| profile | first token before | first token after | cause |
|---|---|---|---|
| tiny | 0.5 s | 0.5 s | — |
| small (Qwen2.5-1.5B) | **7.9 s** avg (up to 11 s) | **2.5 s** avg | the model fact-extraction pass (5–8 s) ran before the reply, even for questions and greetings |

Fixes:
- The model extraction pass now runs after the reply. The instant rule pass stays before it.
- Messages with no statement (questions, greetings) skip the model extraction pass entirely.
- A tool-stored fact gets the user's own framing ("my sister") when the later extraction repeats it.

**Other:**
- Messages that only ask a question are no longer recalled as memories; an earlier identical
  question echoed back into the prompt.
- A configuration error (for example a missing models.json) prints a message and exits with
  code 2 instead of an unhandled-exception dump.

## Small profile (Qwen2.5-1.5B, CPU), dev set

| run | LoRA | LoRA+RAG | base+RAG | interference (LoRA) | confab. LoRA | confab. LoRA+near | drift | train s |
|---|---|---|---|---|---|---|---|---|
| small baseline | 0.883 | 1.000 | 0.967 | 0.067 | 0.55 | 0.87 | **1.61 (gate fails)** | 1811 |
| small, epochs 2 + DPO (β 0.1, lr 1e-4, 3 epochs, 60 pairs) | 0.633 | 0.933 | 0.967 | 0.083 | 0.35 | **0.13** | 1.55 | 6465 |
| **small, epochs 2** (ablation, no DPO) | **0.883** | **1.000** | 0.967 | 0.067 | 0.55 | 0.53 | **1.37 (passes)** | 1151 |

- The 1.5B model reads its memory block almost perfectly (0.97, against 0.37 for the 135M model).
  For real-sized models RAG carries most of the weight; the LoRA adds recall when retrieval misses.
- Confabulation remains the problem: 0.55, and 0.87 with a look-alike fact in the prompt.
- The `small` hyperparameters were tuned on 3 facts (4 epochs, lr 3e-4) and drift too far with
  30 facts (1.61 > 1.5): ten times the optimizer steps. The gate rejects that adapter, as it should,
  but the defaults need to scale with the number of facts.

## DPO abstention stage

`train.dpo` in models.json (off by default) runs Direct Preference Optimization after SFT. The
pairs are: unknown subject (sometimes with a same-shaped stored fact in the memory block) → chosen
"You haven't told me that yet.", rejected = the model's own answer; and known fact → chosen the
fact, rejected the abstention. The reference is the SFT model, scored once. The manifest records
pair accuracy, abstention on held-out unknown subjects, and refusals of known facts.

| tiny run | LoRA | LoRA+RAG | confab. LoRA | confab. LoRA+near | drift | DPO pair acc. | abstains on unseen unknown |
|---|---|---|---|---|---|---|---|
| baseline | 0.783 | 0.933 | 0.70 | 0.80 | 1.78 | — | — |
| DPO β0.1, lr 2e-4, 3 epochs, 60 pairs | 0.750 | 0.867 | 0.70 | 0.73 | 1.94 | 0.93 | 0/6 |
| anchor_ratio 0.3 (anchors replayed) | 0.833 | 0.883 | 0.75 | 0.73 | **2.36** | — | — |
| **epochs 4** (was 8) | **0.800** | 0.917 | 0.70 | 0.73 | **1.21** | — | — |

Tiny DPO finding: the pairs are learned (0.93) but nothing generalizes. The model abstains on 0 of
6 unknown subjects it was not trained on, confabulation is unchanged, and accuracy drops. The same
result as E1: a 135M model memorizes the contrast instead of learning a boundary between known and
unknown. DPO stays off for `tiny`; `small` is the real test.

Anchor-replay finding: rejected. Replaying the 8 anchor answers up to a 30% share raised drift
(2.36, against 1.78): more steps, overfit to those exact anchors, and no protection on the held-out
prompts. Drift appears to track total optimizer steps instead. The next test is fewer epochs.

Epochs finding (applied): with 4 epochs instead of 8, drift falls from 1.78 to 1.21 and training
time from 412 s to 186 s, while LoRA accuracy holds (0.80 vs 0.78). Drift tracks total optimizer
steps, so `tiny` now defaults to 4 epochs. Caveat: steps still grow with the number of facts, so a
much larger fact set will need fewer epochs or a lower learning rate. The drift gate is the safety
net.

## Licensing guard (Mem.Guard), offline evaluation on recorded answers

`build/bench_guard cache bench/facts.json bench/results/<run>.json` re-scores a memory benchmark's
answers with the guard applied. Rule: for a "my X" question, an answer that contains stored values
is licensed only if one of them belongs to a fact about X. "About X" means a shared content word,
or a bge-reranker-base score of at least 0.03 for (question, "subject: text").

| answers from | condition | accuracy | confabulation |
|---|---|---|---|
| tiny, 4 epochs | LoRA | 0.800 → 0.767 | 0.70 → **0.25** |
| tiny, 4 epochs | LoRA + RAG (look-alike) | 0.917 → 0.867 | 0.73 → **0.27** |
| small baseline | LoRA | 0.883 → 0.850 | 0.55 → **0.20** |
| small baseline | LoRA + RAG (look-alike) | 1.000 → 0.933 | 0.87 → **0.33** |

How the scoring was chosen (`gvar`/`gdiag` diagnostics):
- ms-marco cannot separate real paraphrases from look-alikes at any threshold.
- bge on (question, "subject: text") at 0.03 blocks 15/15 dev and 14/15 held-out look-alikes,
  while keeping 22/27 dev and 16/17 held-out of the guarded known questions.
- Scoring the question against the fact text alone, or subject against subject, separates worse.

The remaining false blocks are synonyms bge misses (girl/daughter, boss/manager,
employer/"where I work"). The 0.03 threshold was chosen on dev; held-out answers from a trained
adapter are still to be scored.

## DPO on small (1.5B)

DPO statistics: pair accuracy 0.95. It abstains on 1/6 unseen unknown subjects (tiny: 0/6) and
refuses 0% of known facts.

What it buys, at the system level (RAG is always on in the agent): with a look-alike fact in the
memory block, confabulation falls from **0.87 to 0.13**, on unknown questions whose subjects never
appeared in the DPO pairs. With the guard on top it is **0.07**. LoRA+RAG accuracy costs 7 points
(1.00 → 0.93).

What it costs: LoRA-alone accuracy falls from 0.88 to 0.63. Only 3 of the misses are refusals;
the rest are generic answers ("You drive a car."), which points at weaker fact memorization. That
result is confounded, because epochs (4 → 2) and DPO changed together; the ablation (epochs 2, no
DPO) is queued. Drift 1.55 still narrowly fails the 1.5 gate.

## Held-out verification (facts_heldout.json: 20 facts, 40 questions, 20 unknown; never used for tuning)

| tiny, current defaults | dev | held-out |
|---|---|---|
| extraction value accuracy | 1.00 | 1.00 |
| LoRA | 0.80 | 0.80 |
| LoRA + RAG | 0.917 | 0.925 |
| base + RAG | 0.367 | 0.30 |
| drift | 1.21 | 1.26 |

Guard on the held-out answers (threshold 0.03 chosen on dev):

| condition | without guard | with guard |
|---|---|---|
| known, LoRA | 0.800 | 0.800 (0 correct answers blocked) |
| known, LoRA + RAG | 0.925 | 0.925 (0 blocked) |
| confabulation, LoRA | 0.60 | **0.20** |
| confabulation, LoRA + look-alike | 0.87 | **0.27** |
| confabulation, base + look-alike | 0.33 | **0.00** |

The dev-tuned settings carry over to unseen facts, and on held-out the guard costs no accuracy at all.

## Decision: small = 2 epochs + guard; DPO stays off

The ablation separates the two changes:
- **2 epochs (instead of 4):** the same LoRA accuracy (0.883), now passing the drift gate
  (1.37 against 1.61), look-alike confabulation down from 0.87 to 0.53, and 36% faster training.
  This is the new `small` default.
- **DPO:** it causes the LoRA-alone drop (0.88 → 0.63) and costs 5.6× the training time.

With the guard on, the system-level comparison:

| small | LoRA+RAG accuracy | confab. LoRA | confab. LoRA+look-alike |
|---|---|---|---|
| 2 epochs + guard | **0.933** | **0.15** | **0.067** |
| 2 epochs + DPO + guard | 0.867 | 0.20 | 0.067 |

DPO adds nothing once the guard runs and costs accuracy, so it stays implemented but off. The
`medium`/`large` hyperparameters (8 epochs, from v2) are untested; the same step-driven drift is
likely, so run `scripts/profile_smoke.sh medium --train` on a GPU before relying on them.

## End-to-end system benchmark (bench/bench_agent.py, the real binary)

The tiny profile with the current defaults, CPU, 30 dev facts told as plain sentences, then 60
held-out questions and 20 unknown questions, asked before and after one `/sleep`:

| | before sleep (RAG) | after sleep (LoRA + RAG + guard) |
|---|---|---|
| accuracy | 0.317 | **0.800** |
| confabulation | 0.00 | 0.20 |

- All 30 facts were extracted from conversation.
- Consolidation learned all 30 facts (recall 1.00, drift 1.33).
- 2.7 s per turn (first token 1.4 s); the whole run took 732 s, covering 190 turns and one
  training run.
- After sleep, the system matches LoRA-alone accuracy (0.80) rather than oracle LoRA+RAG (0.92).
  Real retrieval plus the guard's few false blocks cost about 12 points; that gap is the next target.
- The confabulation rate (0.20) matches the guard's offline prediction (0.25).

### Gap analysis and a rejected guard extension

The 12 misses after sleep break down into 5 guard false blocks (employer, girl/daughter,
bicycle/bike, boss/manager, dental checkup/dentist appointment) and 7 model errors, mostly
look-alike facts mixed up (hometown answered with where-I-live, gym answered with the workplace).
The guard made 9 correct blocks in the same run.

Tried: a second license path, bge scoring the subjects against each other ("my bicycle" / "my
bike") at 0.977 or more.
- **On the pairs it was tuned on:** 9/10 synonyms pass and 0/14 look-alikes pass.
- **On 31 fresh pairs (validation):** 11/16 synonyms pass, but 2/15 look-alikes leak
  (grandfather/grandmother 0.991, favorite song/band 0.984).

The margin is 0.003 and the trade is roughly even (about +5 points of accuracy against about +6
points of confabulation), so it is rejected. The guard stays as it is.

## Engines: Python worker vs in-process llama.cpp (2026-10-29)

Same build, same 30 facts, tiny profile, 4-core CPU, temperature 0
(`bench_agent.py --engine python|llama`; results in `agent-tiny-python.json`, `agent-tiny-llama.json`).

| Engine | accuracy before / after sleep | confabulation after | consolidation | avg turn | first token | wall |
|---|---|---|---|---|---|---|
| python (Transformers fp32) | 0.317 / 0.80 | 0.20 | 30/30, recall 1.00, drift 1.33 | 1548 ms | 823 ms | 523 s |
| llama (GGUF f16, 3 threads) | 0.35 / 0.80 | 0.20 | 30/30, recall 1.00, drift 1.33 | **1293 ms** | 973 ms | **386 s** |

- Quality is the same: the same LoRA (trained in PyTorch, exported to GGUF) gives the same
  after-sleep scores. Before sleep 3 of 80 answers differ (f16 vs fp32 arithmetic); single
  prompts checked by hand give token-identical replies.
- Thread count matters more than anything else on CPU (5 chat turns, avg ms/turn): 1 thread 1455,
  2 threads 832, **3 threads 625**, 4 threads 1506 (the Python engine: 1054). With every core
  taken, ggml's spinning workers fight the agent thread and ONNX Runtime, so the default is now
  all cores but one.
- A `q8_0` base (145 MB vs 270 MB) is faster again at 2 threads (570 vs 832 ms) but changes the
  greedy outputs; tiny stays f16, `medium`/`large` convert to `q8_0`.
- The hot swap is a GGUF load: ~20 ms on tiny. The GGUF export adds 3–4 s to a consolidation (`gguf_seconds` in the manifest).
- `make test-e2e-llama` passes (two consolidations, correction, /forget, /rollback, non-ASCII).

## In-process LoRA training on llama.cpp (2026-10-30)

`[consolidate] trainer=llama` (`Mem.Train` + `packaging/llama-patches/0001-lora-training.patch`)
against the Python trainer, same 30 facts, tiny profile, 4-core CPU, llama engine
(`bench_agent.py --engine llama --trainer llama|python`; `agent-tiny-llama-train.json`):

| Trainer | recall (probes) | held-out drift | after-sleep accuracy | confabulation | training time |
|---|---|---|---|---|---|
| python (PyTorch + PEFT) | 1.00 | 1.33x | 0.80 | 0.20 | 132 s |
| llama (in-process) | 1.00 | **1.22x** | 0.75 | 0.20 | 340 s |

`make test-e2e-llama` passes on the in-process trainer: two consolidations (3 then 5 facts), a
correction, `/forget`, `/rollback`; hot swaps 14-40 ms.

How it got there: the first in-process runs memorised 4 facts but drifted 5x, and diverged at 30
facts (loss rising every epoch, drift 236x). Each suspect was ruled out against PyTorch on the same
model and data:

| Check | Result |
|---|---|
| packed examples isolated (B's loss alone vs after A) | identical (2.434083 both) |
| one SGD step, one example: LoRA gradients vs PEFT | cosine 1.000000 on all 420 tensors, rel. error <= 4.7e-4 (q/k LoRA B compared after undoing GGUF's RoPE row permutation) |
| one step, two packed examples | cosine 1.000000, rel. error <= 7e-5 |
| PyTorch replaying Mem.Train's exact windows | converges (0.80 -> 0.008, drift 1.19x): the data is right |
| step-by-step losses, same windows, same initial adapter | equal for 2 steps, then llama's updates ~14% larger, growing |
| step 1's gradient in-session vs from a fresh session | in-session = fresh + step 0's gradient, cosine 1.0000 |

Root cause: ggml gives parameters persistent gradient accumulators when graphs are rebuilt per step
(llama.cpp's training path) but only reset them for `opt_period > 1`; with the default 1 every
step trained on the sum of all gradients so far. The patch zeroes them after each optimizer step;
after that llama.cpp tracks PyTorch to 1e-4 in loss over 12 AdamW steps with clipping.

Other measurements behind the design:

- **f32 training copy:** the backward pass through f16 base weights is 6.4x slower on the CPU
  (14.8 vs 2.3 s per 256-token step); the copy is made once (1.6 s for tiny) next to the GGUF.
- **Window = one profile batch:** each optimizer step sees about `batch` examples, the number of
  steps per epoch consolidate.py's learning rate is tuned for (3 examples per step drifted 5x).
- **Masked rows:** before the cross-entropy fix, prompt and padding rows (all-zero labels) got
  `softmax` as gradient and flattened the model into "acknow acknow..." within one epoch.
- `medium`/`large` stay on the Python trainer: their f32 copies would be 16 GB / 96 GB.

## Varied training examples and adapter layers (2026-10-31)

`python3 bench/bench_memory.py --profile tiny --trainer llama [--set key=value ...]`: the memory
benchmark on the in-process trainer and engine (`build/bench_llama`), as the agent runs by default.
The question: does training each fact from several sides fix confabulation, as O'Neill et al. 2026
([arXiv 2607.11020](https://arxiv.org/abs/2607.11020)) and Hatakeyama-Sato et al. 2023
([arXiv 2312.03360](https://arxiv.org/abs/2312.03360)) suggest? Until now every fact was trained on
the same question templates, all answered with the fact text word for word. `examples=varied`
(`Mem.Examples`, `memlib.varied_qas`) adds short answers, the value asked from the other side, yes/no
checks (one with another fact's value), fill in the blank and two facts together.

30 facts, 60 held-out questions, 20 never-stated questions; recall is 1.00 in every run.

| run | LoRA | LoRA+RAG | interference (LoRA) | confab. LoRA | confab. LoRA+near | drift | train s |
|---|---|---|---|---|---|---|---|
| A: templates ×6, all-linear (before) | 0.750 | 0.883 | 0.083 | 0.80 | 0.80 | **1.29** | 481 |
| B: varied ×12 | 0.900 | **0.983** | 0.067 | 0.90 | 0.87 | 1.80 | (shared CPU) |
| **C: varied ×12, v/o/gate/up (new default)** | **0.933** | 0.967 | **0.050** | **0.70** | 0.93 | 1.49 | 826 |
| D: varied ×12, lr 1e-3 | 0.900 | 0.967 | 0.050 | 0.75 | 0.93 | 1.85 | 870 |
| E: varied ×6 | 0.900 | **0.983** | 0.050 | **0.70** | 0.80 | 1.63 | 460 |
| F: varied ×6, v/o/gate/up | 0.883 | **0.983** | 0.050 | **0.70** | 0.93 | 1.78 | **422** |

Findings:
- Varied examples are worth 13-18 points of accuracy from the weights alone, at the same number of
  examples (E) or twice as many (B, C), and they lower interference.
- They do not fix confabulation: 0.70-0.90 against 0.80 is within noise (one question is 0.05). The
  adapter still answers a question about something never stated with a stored value; the guard
  (Mem.Guard) remains what stops it in the agent.
- Adapters on v/o/gate/up only (Hatakeyama-Sato's best layers, no q/k/down) gave the best accuracy
  and the lowest drift of the varied runs at 12 examples (C); at 6 (F) they did not help.
- Half the learning rate (D) did not lower drift.
- Drift is measured on 4 held-out anchor prompts and moves by ±0.2 between similar runs, so the
  drift column ranks only coarsely. All runs pass the 3.0 gate.

Verified on the held-out set (`--facts bench/facts_heldout.json`: 20 facts, 40 questions, 20
unknown; never used for tuning):

| held-out | LoRA | LoRA+RAG | interference (LoRA) | confab. LoRA | confab. LoRA+near | drift | train s |
|---|---|---|---|---|---|---|---|
| templates ×6, all-linear (before) | 0.850 | 0.750 | 0.050 | 0.85 | 0.80 | 1.15 | 346 |
| varied ×12, v/o/gate/up | **0.925** | **1.000** | 0.075 | 0.80 | 0.80 | 1.48 | 589 |

Accuracy holds up off the dev set; with the fact in the prompt as well, the varied adapter
answers every held-out question. Interference 0.075 vs 0.050 is one question.

`tiny` now trains with `examples: varied`, 12 examples per fact, adapters on
`v_proj, o_proj, gate_proj, up_proj` (run C). The other profiles keep templates until measured.

## Memory loop, iteration 1: extraction coverage (2026-10-31)

A fact that is never stored can't be retrieved or trained, so the first thing to measure is how
many facts get stored at all. The old benchmark's "extraction 1.00" came from facts written to fit
the rules ("my X is Y"). `bench/facts_natural.json` holds 40 dev and 40 held-out facts phrased the
way people say them ("I'm allergic to peanuts", "My son Leo just turned seven"), all written
before any tuning. `bench/chatter.json` holds 40 chat lines that state no durable fact, to catch
false memories. Both are scored by `build/bench_extract`, which runs the Pascal extractor the agent
uses.

| rule extractor (tiny profile) | dev coverage | held-out coverage | junk facts from chat |
|---|---|---|---|
| before | 0.225 | 0.250 | 0/40 |
| + 28 rules from the dev misses | 1.000 | 0.675 | 19/40 |
| **+ value checks (adopted)** | **1.000** | **0.650** | **0/40** |

- **New rules** (`memlib.EXTRA_RULES`, generated into `memagent/Mem.ExtractRules.inc`): pets and
  people with names, allergies, birthplace, job, diet, car, moves, likes and dislikes, what to call
  me, languages, instruments, training goals, devices, teams, schools, counts of children and
  siblings, age, where relatives live, commutes, upcoming events, things being learned, weekly
  habits, coffee and tea, where I grew up, a partner's job. Also: "my son Leo …" (a relation and
  a name), "my birthday's on …", and "I've been working at …".
- **Value checks:**
  - Without them, 19 of 40 chat lines became facts ("I love you" → "You love you."; "I moved
    to the next question" → "You live in next question.").
  - Every new rule now rejects a value that starts with a pronoun or filler word.
  - Rules for names and places (birthplace, moves, names, teams, schools, cars, devices)
    require a proper noun.
- **Honesty about held-out:** rules were written only from dev misses. Phrasings seen only in
  the held-out set ("I've got a …", "we call our …", "I can't stand …", "I turned 41") were left
  out on purpose, and they make up most of the 14 held-out misses. That is the generalization
  limit of a rule extractor. The `small` profile and up add LLM extraction on top, which a later
  iteration should measure on the same sets.
- **No regressions:** the original benchmark facts still extract 30/30 and 20/20. The Pascal and
  Python extractors agree on all 698 parity cases, now including the natural and chatter sets.

## Memory loop, iteration 2: memory over time (2026-10-31)

`python3 bench/bench_stream.py` drives the real binary through sessions:
- each session tells it 10 facts; the last session also corrects two earlier ones
- `/nap` trains the fast LoRA, then a checkpoint asks one held-out question per fact so far,
  plus 6 never-stated questions, twice: through the full agent and from the weights alone
  (`/weights`)
- `/sleep` runs a deep sleep, and a final checkpoint follows the last one

Accuracy is split into the latest session's facts ("new") and earlier ones ("old", retention).

**Baseline** (v0.3.0 defaults, 30 facts, 3 sessions, `stream-tiny-baseline.json`):

| checkpoint | facts | full agent: new / old | weights only: new / old | confabulation, agent / weights |
|---|---|---|---|---|
| session 1 | 10 | 0.60 / – | 0.80 / – | 0.00 / 1.00 |
| session 2 | 20 | 0.90 / 0.90 | 1.00 / 1.00 | 0.17 / 0.33 |
| session 3 (+2 corrections) | 30 | 0.60 / 0.75 | 0.80 / 0.75 | 0.50 / 0.50 |
| after the final deep sleep | 30 | – / 0.93 | – / 1.00 | 0.17 / 0.67 |

Findings:
1. **The full agent scored below the weights alone** at three of four checkpoints. The main
   cause is the guard withholding right answers, in 4 to 6 questions per checkpoint: "Who is my
   employer?" (the fact is "where I work"), "Who is my boss?" ("my manager"), "What is my girl
   called?" ("my daughter"). No word matched, and the cross-encoder scored the pairs under its
   threshold.
2. **"Fast" runs are not fast:** 645 s and 841 s, against 634 s for a deep sleep over 20 facts.
   They train the new facts plus an equal replay sample, with two adapters in the graph.
3. **A wasted deep sleep:** `/sleep` right after the first run rebuilt the same 10 facts (341 s).
4. **Retention holds:** old-fact accuracy from the weights stays at 1.00 through session 2, and
   every fast run kept all of its replayed facts.

**Fix 1, guard synonyms:** `Mem.Guard` matches words that name the same thing in a question
about the user (work/employer/company/salary, boss/manager, girl/daughter, boy/son,
mother/mom/mum, father/dad, partner/wife/husband/…, kid/child, home/live/house, car/vehicle).
Re-scored offline (`build/bench_guard`) on the recorded memory-benchmark answers:

| | correct answers withheld | confabulation, LoRA → with guard |
|---|---|---|
| dev, before | 3–4 | 0.70 → 0.25 |
| dev, after | 0 | 0.70 → 0.25 |
| held-out, before / after | 0 / 0 | 0.80 → 0.25 / 0.80 → 0.25 |

None of the look-alikes the guard blocks (horse/dog, aunt/sister, stepmother/mother,
parrot/cat) shares a group.

**Fix 2:** a deep sleep is skipped when the deep LoRA already holds every fact (nothing in the
fast tier, nothing forgotten or changed).

## Memory loop, iteration 3: what training time is spent on (2026-10-31)

Per-phase timings (now in every adapter's `manifest.json` under `phases`) for a deep sleep over
10 facts and a fast run over 10 new facts:

| run | total | training | anchors | probes | other |
|---|---|---|---|---|---|
| deep sleep, 10 facts | 365 s | 326 s (89%) | 28 s | 8 s | ~4 s |
| fast run, 10 new + 10 replayed | 675 s | 634 s (94%) | 24 s | 12 s | ~4 s |

There is no large fixed overhead: cost is examples × epochs, about 0.64 s per example pass on
tiny (4 cores). A fast run cost twice a deep sleep of the same size because its replayed facts
got full example sets, although they are already in the frozen deep LoRA and are there only so
the new adapter learns to leave them alone. Replayed facts now get `max(2, per_fact div 3)`
examples (4 on tiny).

Verified on the stream benchmark (`stream-tiny-loop3.json`, together with the iteration 2 fixes):

| | baseline | loop 3 |
|---|---|---|
| total training | 3825 s | 3077 s (−20%) |
| fast runs | 645 s, 841 s | 495 s, 561 s |
| weights only, old facts at session 3 | 0.75 | 0.80 |
| weights only, after the final deep sleep | 1.00 | 1.00 |
| full agent: session 1 / session 3 / final | 0.60 / 0.70 / 0.93 | 0.70 / 0.80 / 0.97 |

Retention is unchanged or better, and the full agent gains 3–10 points from the guard fix.
Confabulation moved by one question; 6 never-stated questions per checkpoint make each one
0.17.

**Drift grows with memory.** Deep-sleep drift was 1.27×, 1.41× and 1.76× at 10, 20 and 30 facts,
identical in both runs because training is deterministic. At that rate the 3.0 gate would start
rejecting deep sleeps around 60–80 facts, and memory would stop updating. The anchor set is
fixed at 8 examples while fact examples grow to 360. That is the next iteration.

## Memory loop, iteration 4a: retrieval of facts already in the weights (2026-10-31)

Several wrong full-agent answers in the stream baseline came from retrieval putting a misleading
fact in the prompt, while the weights alone were right. `rag_include_baked=1` was chosen when the
LoRA was much weaker (+15 points then). With the stronger varied-example LoRA, does leaving baked
facts out of retrieval help?

| full agent, `stream-tiny-loop3` vs `stream-tiny-loop4a-nobaked` | session 1 | session 2 | session 3 | final |
|---|---|---|---|---|
| baked facts retrieved (default) | 0.70 | 0.90 | 0.80 | 0.967 |
| baked facts left out | 0.70 | 0.95 | 0.867 | 0.933 |

The difference is a few questions in each direction (+1, +2, −1), and weights-only scores are
identical because training is deterministic. **Inconclusive: the default stays.** Retrieving
baked facts is also the safety net for a fact the weights lose.

## Memory loop, iteration 4b: drift against the number of anchors (2026-10-31)

Drift (perplexity ratio on general text) is what the gate rejects, and it grew with memory. Two
ways to bring it down, both on the 30-fact dev set (`memory-tiny-llama-G*`, `-H*`):

- **Fewer, larger training steps** (`max_windows=16`): rejected. Drift went from 1.49 to 1.83, and
  confabulation from the weights rose to 0.90.
- **More general-text anchors in training.** The first try reported 16 anchors as drift 1.09, but
  that measurement was flawed. The held-out drift prompts were whatever anchors were left over, so
  they changed with the anchor count. Both trainers now hold out a fixed set (the last 8 of the
  32 anchor prompts) and train on the first `anchors` of the other 24 (commit a6ad994):

| anchors | accuracy LoRA / +RAG | interference | confab. (weights) | drift | recall | train s |
|---|---|---|---|---|---|---|
| 8 (old default) | 0.933 / 0.967 | 0.050 | 0.70 | 1.39 | 1.00 | 1074 |
| 16 | 0.917 / 0.983 | 0.033 | 0.70 | 1.22 | 1.00 | 1147 |
| **24** | 0.917 / 0.967 | 0.067 | 0.70 | **1.01** | 1.00 | 1137 |

Drift goes down steadily with more anchors, to almost none at 24. Accuracy and interference move
by one question at most, and the run takes about 6% longer. **Tiny now trains with 24 anchors.**

Held-out facts (`memory-tiny-llama-heldout-anchors{8,24}.json`), never used for tuning:

| anchors | accuracy LoRA / +RAG | interference | confab. (weights) | drift | train s |
|---|---|---|---|---|---|
| 8 | 0.925 / 1.000 | 0.075 | 0.80 | 1.25 | 733 |
| 24 | 0.925 / 0.975 | 0.025 | 0.80 | **0.96** | 819 |

Same picture: drift goes from 1.25 to 0.96 with the same accuracy, LoRA alone.

Stream benchmark (`stream-tiny-loop4b-anchors24.json`). It hit the 2-hour run limit during the
final question round, so the final checkpoint has 28 of 30 facts and no unknown questions.
Every training run had finished:

| | deep drift at 10 / 20 / 30 facts | fast drift, sessions 2 / 3 | agent accuracy, checkpoints 1–4 | weights, old facts after session 3 | train s |
|---|---|---|---|---|---|
| 8 anchors (`loop3`) | 1.27 / 1.41 / **1.76** | 1.01 / 1.37 | 0.70 / 0.90 / 0.80 / 0.967 | 0.80 | 3077 |
| 24 anchors | 1.01 / 1.05 / **1.12** | 1.04 / 1.08 | 0.70 / 0.95 / 0.80 / 0.964 | 0.60 | 4784 |

(The two rows measure drift on different held-out prompts: loop3 predates the fixed set. What
compares is the growth: +0.49 across 20 more facts before, +0.11 now.) Drift no longer climbs
toward the 3.0 gate as memory grows, and the agent's accuracy is unchanged. Two costs to watch:

- With the fast LoRA stacked on the deep one after session 3, the weights alone kept 12 of 20
  earlier facts instead of 16. The full agent was unaffected (0.80 both times), and the deep
  sleep restored all of them (1.00).
- Training time rose 55%. Part of that is the extra anchor examples. Part of it is that the
  retrieval benchmarks of iteration 6 ran on the same CPU during this run, so this number is not
  clean.

## Memory loop, iteration 5: confabulation that gets past the guard (2026-10-31)

In the stream baseline (`stream-tiny-loop3`), the full agent still answered 4 of 6 never-stated
questions with a stored value after session 3. The guard only checks questions about "my X", and
4 of those 5 leaks had no "my": "Which school did I go to?" got Teo, "What motorbike do I own?" got
Rocket. The fifth was "When is my partner's birthday?" getting March 3rd, which was licensed
because "birthday" matched the user's own birthday.

**Owner rule (applied).** When the question names an owner ("my partner's birthday"), a word
match only licenses facts about the same owner. Offline on six recorded answer sets (dev and
held-out, `bench_guard`): no correct answer blocked on any set, and LoRA confabulation after the
guard 0.20 to 0.15 on the newest dev set. The other sets are unchanged.

**Guarding every question: rejected.** The guard was widened to any question, using its content
words as the subject ("What do I drive?" gives "drive"). That cut confabulation after the guard
from 0.20 to 0.10, but blocked 12 of 60 correct dev answers. A paraphrase shares no word with its
fact ("Who am I dating?" against "my partner's name"), and licensing has to come from a model:

- bge-reranker scores those real paraphrases 0.000 to 0.027 and the look-alikes 0.0006 to 0.004.
  No threshold separates them.
- Licensing a fact only when retrieval ranks it first for the question (minilm cosine) keeps
  100 of 107 correct uses, but stops just 3 of 13 confabulations. The rest are true nearest
  neighbours: motorbike gets the bike, high school gets the university, garage code gets the
  office door code. The closest stored fact is the wrong one, and nothing lexical or rank-based
  sees that.

What is left is a model's knowledge boundary, which SFT negatives and DPO did not teach at 135M
(above). The candidates are the larger profiles, or answering such near-miss questions with the
stored fact named ("You told me your bike is Rocket; you haven't mentioned a motorbike"). The
second one changes the answer style and needs its own benchmark.

## Memory loop, iteration 6: retrieval when memory fills up with look-alikes (2026-10-31)

The earlier scale test (profiling round) only timed retrieval, and its filler facts were about
shipping crates, which share nothing with personal facts. A long-lived memory fills up with facts
about other people that have the user's own shapes. `BENCH_FILLER=personal` generates those:
40 owners ("my neighbor's", "Tom's") times 20 attributes ("dog's name", "employer", "favorite
band"). Production configuration (MiniLM + FTS + ms-marco, pool 8, rag_k 3), dev facts:

| look-alikes stored | hit@1 | hit@3 | wrong@1 | hit@1 | hit@3 | wrong@1 |
|---|---|---|---|---|---|---|
| | **before** | | | **after** | | |
| 0 | 0.900 | 0.967 | 0.067 | 0.900 | 0.967 | 0.067 |
| 100 | 0.883 | 0.950 | 0.083 | | | |
| 800 | 0.783 | 0.883 | 0.200 | 0.883 | 0.967 | 0.100 |
| 2,000 | 0.733 | 0.833 | 0.233 | 0.867 | **0.967** | 0.117 |

Every miss was rank 0: the right fact never reached the 8-candidate pool. "Who is my employer?"
got "my manager's employer", "my neighbor's employer" and so on. A bigger pool did not help
(16 and 32 both gave hit@3 0.833 at 800), because ms-marco ranks the other people's facts higher.

**Owner-aware search (applied).** A fact whose subject names an owner the question does not
mention (`Mem.Guard.ForeignOwner`: "my manager's employer" for "Who is my employer?"; a
possessive followed by "name", as in "my partner's name", is the fact's topic and not an owner)
ranks after the others:

1. The keyword list and the vector list are each searched 4 times deeper.
2. Foreign-owner facts move to the end of each list, and each list is cut back to the pool size
   before rank fusion. With nothing foreign, fusion is exactly as before. A first version that
   moved them after fusion changed the fusion itself and lost a question at 0 fillers.
3. After reranking, they move to the end again.

Held-out facts: hit@3 0.975 before and after, at 0 and 2,000 look-alikes. At 2,000, noise (unknown
questions that still get a fact injected) went from 0.70 to 0.90. The pool now holds the user's own
nearest facts instead of only other people's, and some pass the reranker. That matches the
0.85–0.90 the dev set already has, and the guard covers it. Search time at 2,000 look-alikes is
about 35 ms (it was 25 ms), within the 150–250 ms query total, which the reranker dominates.

## Memory loop, iteration 7: replay every deep fact in fast runs (2026-10-31)

Iteration 4b lost old facts from the weights after session 3's fast run (weights-only old retention
0.60). The losses were sibling swaps: the sister got the partner's name, the son got the other
child's. The fast run then replayed 12 of the 20 deep facts. This run replays all of them
(`bench_stream.py --set replay=0`). There were 18 because the two corrected facts were in the new set.
Everything else was as in `stream-tiny-loop4b-anchors24`.

| session 3, after the fast run | replay 12 (default) | replay all |
|---|---|---|
| agent: new / old | 0.80 / 0.80 | 0.70 / 0.75 |
| weights only: new / old | 0.90 / 0.60 | **1.00 / 0.70** |
| fast run drift (ppl ratio) | 1.083 | 1.144 |
| after the final deep sleep: agent / weights | 0.964 / 1.000 | 0.967 / 0.967 |

Full replay fixed the targeted swaps in the weights: the sister is Ana again, and the son is Teo.
The agent, however, lost three other answers to swaps of the same kind, for example "home city"
answered with Porto. The net change is two questions up and two down out of 30. That is within
one-question noise, and drift went up. **The default stays `replay: 1`** and the setting stays
available. Sibling interference at 135M is a capacity limit that replay ratios don't move. Only the
deep sleep cleans it up, and it does that in both runs.

## Integration experiment: does sleep teach the model to combine facts? (small profile, 2026-10-31)

Complementary-learning-systems theory says slow learning into the weights pays off in
knowledge that *links* separately learned facts: transitive and relational inference
improves after sleep (Ellenbogen et al. 2007, Lau et al. 2010). The earlier benchmarks
only ask one stored fact back, which is what retrieval is best at. This one asks
questions that need two facts together:

- **Facts.** 112 facts from `gen_integration.py`:
  - 16 "My sister's name is Ana." facts.
  - 48 attribute facts about those people (city, job, pet).
  - 48 attribute facts about strangers. Half of them share a value with a relative.
- **Questions.** 96 questions, 24 of each type:
  - direct: "Where does Ana live?"
  - bridge: "Which city does my sister live in?"
  - inverse: "Which of my relatives is a nurse?"
  - verify: "Is my sister the one whose city is Lisbon?"
- **Runner.** `bench_integration.py` runs the real `memagent` and splits every score by
  whether retrieval put all the facts a question needs into the prompt (found / missed).

**Retrieval only** (qwen2.5-1.5B, rag_k 3), all 96 questions:

| type | correct | retrieval found every needed fact |
|---|---|---|
| direct | 24/24 | 100% |
| bridge | 3/24 | 12.5% |
| inverse | 7/24 | 0% |
| verify | 10/24 (12 is chance) | 83% |

Retrieval cannot do a two-hop question: the question names "my sister", so it finds
"Your sister's name is Ana", but not "Ana's city is Lisbon". In 15 of the 24 bridge
questions the model then picked the wrong stored value ("Hana lives in Bologna", but
Bologna is Sofia's city): a **binding error**, the right kind of item attached to the
wrong person.

**Two bugs this found, both fixed:**

- *Subject merge corrupted memory.* Merging duplicate subjects treated "Tomas's job" as
  a duplicate of "Tomas's city", and "my grandpa's name" as a duplicate of "my grandma's
  name". 14 of 112 facts were superseded. `DistinctSubjects` in `Mem.Guard` now vetoes
  a merge when the owner or the attribute differs: 112/112 facts are kept.
- *The guard licensed bound-to-someone-else answers.* The guard checked that the value
  was stored, not whose it was. The new binding check (`Unbound` in `Mem.Guard`) blocks
  an answer whose value is stored only under other owners. Offline, on the
  retrieval-arm answers, wrong stored values dropped from 17 to 8. It blocked 9 answers,
  none of them correct. The facts and held-out guard benchmarks are unchanged.
- The rule extractor also dropped third-person facts ("Ana's city is Lisbon."). It has a
  named-possessive rule now, in Python and Pascal, and passes 710/710 extract cases.

**Deep sleep could not learn 112 facts within the drift gate.** Each sleep took about
70 min on 4 CPU cores:

| sleep | recall | drift (ppl ratio) | gate (0.7 / 1.5) |
|---|---|---|---|
| lr 3e-4 (profile default), 3 paraphrases | 0.37 | 1.47 | rejected: recall |
| lr 1e-3, 24 anchors | 0.93 | 2.07 | rejected: drift |
| same, gate drift raised to 3 for this experiment only | 0.93 | 2.07 | promoted |

All 71 recall failures in the first sleep were binding errors, and so were all 8 in the
second. The weights learned the values (the items) before they learned whose value is
whose (the bindings). That is the associative-deficit pattern in human memory: item
memory is learned fast, relational binding is learned slowly. It also shows the
stability–plasticity dilemma on a small model: learning the bindings costs too much
drift.

**Retrieval plus the deep LoRA**, the 53 questions it finished before the 2-hour task
limit stopped it (direct, bridge, and the first 5 inverse questions; no verify):

| type | retrieval only | retrieval + deep LoRA |
|---|---|---|
| direct | 24/24 | 23/24 |
| bridge | 3/24 | 1/24 |
| inverse (5) | 2/5 | 4/5 |
| all 53 | 29 | 28 |

- **No integration gain.** Bridge got worse, not better. The one new bridge hit ("Your
  cousin's city is Lisbon") sits beside two new "You haven't told me that yet." answers.
- **The inverse gains are mostly not real.** The answers list several people, and the
  scorer accepts any list that contains the right name. Two of the four mix in a
  stranger or an invented relation ("Priya's cousin's name is Lucia").
- **The drift is visible in the answers.** 33 of 53 answers run on and invent new
  "User: / Assistant:" turns until the token limit. They average 563 characters,
  against 25 for retrieval only. The model lost its turn format: that is what a drift
  ratio of 2.07 looks like. Each answer took about 2.2 min, against 26 s. The
  weights-only arm did not run.

**Verdict for the 1.5B profile.** On CPU, consolidation cannot buy integration within a
usable drift budget. The binding errors that sink retrieval on two-hop questions are
the same errors the weights make. The gains that are real came from the store and the
guard (subject-merge veto, binding check). The deep tier stays a backup for facts
retrieval misses, not a way to reason across facts. The decisive version of this
experiment needs a model large enough to learn the bindings at drift ≤ 1.5, which means
a GPU.

Results: `results/integration-small-partial.json` (the 53 paired answers, the sleep
manifest, timings) and `results/integration-small-retrieval-answers.json` (all 96
retrieval-only answers).

## Edited facts, GPU deep sleep, retrieve-twice and model size (2026-11-06/07)

Same integration set: 112 facts, 96 questions in four types.
- **Agent rows** run the real memagent (llama.cpp, guard on).
- **GPU rows** run `bench_integration_hf.py` (transformers, no guard) on an RTX A6000.

GPU rows feed every model the agent's own retrieved facts (one hop), so the sizes compare
like for like. Compare GPU rows with each other, not with agent rows.

### Writing facts by editing instead of training (`trainer/edit_facts.py`)

The edit follows MEMIT (Meng et al. 2023):
- **Target:** for each person, find the hidden-state change after layer 8 that makes the
  model say their facts.
- **Solve:** spread that change over the MLP output weights of layers 4–8 in one solve,
  `dW = R K^T (lam C + K K^T)^-1`. C comes from 100k tokens of wikitext.
- **Export:** save dW as a LoRA, which the agent loads like any other adapter.

One bug mattered. The name sat at position 0 in some prompts, and Qwen's attention-sink
activation there is huge and nearly the same for every name. With it fixed:

| Model | lam | Recall (no retrieval) | Drift | Wrong person's value | Time on GPU |
|---|---|---|---|---|---|
| 1.5B | 15000 | 0.39 | 1.004 | 31 | 7 min |
| 1.5B | 500 | 0.83 | 1.016 | 11 | 7 min |
| 4B | 500 | 0.94 | 1.010 | 2 | 12 min |

λ 150 and 50 gave the same as 500.

In the agent (1.5B):
- **Retrieval + edit:** 41/96, against 44/96 for retrieval alone.
- **Edited weights alone:** 26/96. Only 10 of 24 direct questions, because an edit
  written as "Ana's city is…" does not answer "Where does Ana live?".

Edits are fast and barely move the model, but they do not help combine facts.

### Deep sleep on a GPU (the Python trainer, `trainer/consolidate.py`)

| Model | Settings | Recall | Drift | Time |
|---|---|---|---|---|
| 1.5B | profile (6 wordings × 2 epochs, lr 3e-4) | 1.00 | 1.53 | 116 s |
| 1.5B | lr 1e-3, 24 anchors, 3 wordings | 0.94 | 2.08 | 81 s |
| 4B | profile (8 epochs, lr 5e-4, 4-bit) | 1.00 | 1.39 | 14 min |
| 24B | old profile (8 epochs, lr 5e-4, 4-bit) | 1.00 | **15.96** | 52 min |
| 24B | **2 epochs, lr 2e-4** (now the profile) | 1.00 | 1.04 | 15 min |

The same 1.5B deep sleep on 4 CPU cores with llama.cpp reached 0.37 recall in 70 minutes.

### Does the trained memory help? (GPU rows)

| Model | Arm | direct | bridge | inverse* | verify | all |
|---|---|---|---|---|---|---|
| 1.5B | retrieval | 24 | 4 | 19 | 16 | 63 |
| 1.5B | retrieval + sleep | 24 | **18** | 12 | 0 | 54 |
| 4B | retrieval | 23 | 4 | 20 | 22 | 69 |
| 4B | retrieval + sleep | 24 | **20** | 15 | 18 | **77** |
| 24B | retrieval | 24 | 3 | 21 | 24 | 72 |
| 24B | retrieval + sleep (old settings) | 9 | 2 | 3 | 0 | 14 |
| 24B | retrieval + sleep (2 epochs, 2e-4) | 24 | **23** | 10 | 5 | 62 |

\* Inverse is inflated for retrieval: the scorer accepts answers that list several names.

**Fact-combining works from 1.5B up.** Retrieval brings "Your sister's name is Hana", the
weights know Hana's city, and the model answers. Neither manages alone: the weights alone
get 2–3 of 24 bridge questions.

**Training also teaches a format.** Every training example is question → fact sentence. So
"Is my friend the one whose city is Porto? Answer yes or no." gets "Your friend's city is
Porto.": right, but not a yes or no. Verify fell to 0 at 1.5B and to 5 at 24B; the 4B
mostly kept it. The next fix is training data with yes/no and which-relative questions,
plus anchors that ask for short answers.

### Retrieve twice (`[retrieval] hops=2`, Mem.Recall.RecallHops)

The second hop follows each name in the first hits. "Your brother's name is Yusuf" leads to a
search for "Yusuf …", which returns "Yusuf's city is Ghent". Agent rows, 1.5B, binding-guard
build:

| Hops | Both facts in the prompt: bridge / inverse / verify | Correct |
|---|---|---|
| 1 | 12.5% / 0% / 83% | 40/96 |
| 2 | **100% / 71% / 88%** | **58/96** |

It needs no training and costs about 70 ms per extra hop. It is now the default.

The GPU-trained 1.5B memory in the agent, two hops, rebuilt binding-guard build:

| Arm | direct | bridge | inverse | verify | all |
|---|---|---|---|---|---|
| retrieval | 24 | 12 | 6 | 16 | **58/96** |
| retrieval + GPU deep sleep | 24 | 4 | 4 | 0 | 32/96 |

With one hop it loses too: 33/96 against 40/96 for retrieval, with verify 0/24 and 36 answers
withheld by the guard.

At 1.5B the trained memory makes the agent worse. Its fact-sentence answers fail verify, and
the guard withheld 40 of them. So the 1.5B presets do not train in the background.

### What changed because of this

| Change | Why |
|---|---|
| `[retrieval] hops=2` is the default | 40 → 58 of 96 on 1.5B |
| Large profile trains 2 epochs at lr 2e-4 | 8 epochs at 5e-4 learned every fact and drifted ×16, so the gate would have thrown away every deep sleep |
| Medium gate allows drift up to 1.5 | Its best run, the top score of the day (77/96), drifted 1.39 and failed the old 1.3 limit |
| The agent warns after two rejected runs in a row | It names drift (lower epochs or lr) or recall (raise epochs or wordings), so rejected background training no longer wastes compute unnoticed |
| `scripts/run.sh` (Linux) and `run.cmd` / `run.ps1` (Windows) pick a preset from CPU, RAM and GPUs | See the README quick start |
| 1.5B presets do not train in the background | Trained memory scored 32/96 against 58/96 for retrieval in the agent |

### One-shot presets: what is measured and what is not

**Measured:**
- Recall, drift and training time for every model above.
- 24B on one 48 GB card: QLoRA training peaks at 23.7 GB. The Q4 model takes about 14 GB to
  answer, plus about 5 GB for a 16k context. So answering and training at once fits in
  about 43 GB.

**Not measured:**
- How much memory the 1.5B (gpu8) and 4B (gpu12) presets use while answering and training
  on one card.
- Two GPUs, and hours of continuous background training. The next test streams facts in
  while the trainer loops, and tracks answers and drift.

**Windows:** `run.ps1` is written for Windows PowerShell 5.1. It was not run in this session.
The release workflow dry-runs it on a Windows runner for every pull request.

## The answer guard as a decision-model readout (Jev, 2026-11-07)

The agent's guard caused many of today's losses. It withheld 29 answers with retrieval
and 36–40 with trained memory, and many of them were right ("Your grandma keeps a
hedgehog"). The question was whether a decision model judges better.

**The test set** (`bench/jev/make_cases.py`):
- 120 answers that the 1.5B, 4B and 24B models really gave in the GPU runs.
- Bridge, inverse and verify questions, 20 right and 20 wrong of each, labelled strictly.
  An inverse answer must name only the right relative; a verify answer must say yes or no
  correctly.
- Allowing everything scores 50%.

**The two guards** (`memagent/tests/jev_guard.lpr`):
- **Current guard:** `Mem.Guard.Licensed`, a reranker plus binding rules, checking against
  all 112 stored facts, as in the agent.
- **Decision readout:** in JevBan's format (`Ban.Local`). The state holds the two-hop
  retrieved facts and the question and answer, as data. One forward pass reads the logits
  of the option letters; there is no generation.

| Guard | Right kept | Wrong blocked | Accuracy | AUC | CPU, 4 cores |
|---|---|---|---|---|---|
| current guard | 28/60 | 37/60 | 0.54 | — | ~0.1 s |
| readout with the 1.5B chat model | 44/60 | 23/60 | 0.56 (0.68 at p≥0.8) | 0.65 | 3.3 s |
| **OpenJev-Flash-9B, p≥0.5** | 54/60 | 42/60 | 0.80 | **0.93** | 11.5 s |
| **OpenJev-Flash-9B, p≥0.9** | 47/60 | 55/60 | **0.85** | | 11.5 s |
| current guard AND OpenJev p≥0.9 | 21/60 | 59/60 | 0.67 | | |
| current guard, OpenJev p≥0.9 can overrule a block | 54/60 | 33/60 | 0.72 | | |

AUC measures how well the score separates right answers from wrong ones; 0.5 is chance.

| By type (OpenJev, p≥0.5) | Current guard | OpenJev | AUC |
|---|---|---|---|
| bridge | 0.40 (kept 3/20) | **1.00** (20/20 kept, 20/20 blocked) | 1.00 |
| inverse | 0.62 | 0.65 | 0.78 |
| verify | 0.60 | 0.75 | 0.93 |

**Two-fact answers.** OpenJev judged all 40 correctly. It resolves "your grandma" → Nils →
Nils's pet from the facts in its state, which the current guard's rules miss.

**Verify mistakes.** Most are answers like "Your daughter's city is Valencia." They are true
but don't say yes or no, so the strict labels call them wrong; Jev scored them 0.8–0.9. Two
are real misses: "Your sister's pet is a rabbit" (Rosa's rabbit) scored 0.82, and "Your
nephew's pet is a tortoise" (Lena's) scored 0.88.

**Inverse answers.** These are the weak spot for both guards. A list of several people is
hard to judge from 3–6 facts.

**The 1.5B chat model.** As a judge it is barely better than the current guard, so the
calibrated decision model is what makes the difference, not the readout trick itself.

**Verdict.** OpenJev-Flash-9B is the better guard by a wide margin: 0.85 against 0.54. The
cost is 11.5 s per answer on 4 CPU cores; on a GPU it is well under a second. The next step
is a `Mem.Decide` unit that runs the guard as a readout, behind `[guard] judge=jev`:
- Use it in the GPU presets, where the 9B fits beside the chat model on a 12 GB+ card.
- Keep the current guard on CPU-only machines.
- The same readout can also decide whether a new fact replaces an old one (the merge bug)
  and which candidate facts go into the prompt.

### Built in: `[judge]` (Mem.Decide)

**Settings:** `[judge] model=openjev-flash-9b` loads the judge beside the chat model, on the
answering GPU when there is one.
- **Guard:** an answer to a question about the user's people is shown only when
  P(supported) ≥ `guard_min` (0.9). The judge reads the facts that turn's prompt had.
- **Merges:** the rules still propose that a new fact supersedes an old one. The judge must
  agree that it is the same person and the same attribute (`merge_min` 0.9); otherwise
  both facts are kept.
- **Launchers:** they turn the judge on for GPU presets with a CUDA llama.cpp and 16 GB or
  more. Everywhere else it is off.

**Live, small profile, CPU:**
- "Which city does my sister live in?" → "Lisbon." (allowed).
- "Ana's city is Porto" replaced "Ana's city is Lisbon" (the judge agreed).
- An unknown fact got no invented answer.
- Each judgement took about 35 s with the chat model on the same 4 cores. That is why
  it is off on CPUs.

**One fix came out of it.** A plain statement ("My sister's name is Ana.") had gone through
the guard too, and the judge withheld the reply (p 0.87). Statements now skip the guard.

**From Hightouch's write-up on LLM judges for identity resolution (Nov 2026):**
- **Over-merging is the failure that matters.** Their identities merge records the way
  memagent merges facts, so the judge here may only *veto* a merge. An unsure judge keeps
  both facts.
- **Calibration over raw accuracy.** OpenJev is calibrated on this set too:
  - p < 0.1: 25 answers, all wrong.
  - p ≥ 0.97: 21 answers, all right.
  - 0.3–0.9: close to a coin flip (24 answers, 10 right).
- **Escalation.** They route that middle band to a stronger model. Locally there is
  none: routing it to the rule guard lowered accuracy (0.80 against 0.85), so the guard
  withholds there, which is the safe side.
- **Threshold.** Tune it on a labelled set of your own: 0.9 is best here (0.85, against
  0.80 at 0.5).
- **Next:** a labelled set of merge decisions, built from look-alikes as in their hard set
  (same-name people, twins, an updated value against another attribute), to tune
  `merge_min` the same way.

### Tuning the merge judge (75 labelled pairs)

**The test set** (`bench/jev/make_merge_cases.py`) follows Hightouch's hard set.
- **36 pairs that should merge:** a new value, a corrected name, the same slot reworded.
- **39 look-alikes that must not:**
  - another attribute or another person;
  - grandma / grandpa and other look-alike relations;
  - a person and that person's pet or thing;
  - me and my partner;
  - the same first name for different people;
  - a fact that adds rather than replaces.

`memagent/tests/merge_judge.lpr` runs the agent's rules (`SameSubject`: subject cosine,
`DistinctSubjects`, cross-encoder) and OpenJev-Flash-9B on each pair.

| Policy | Wrong merges (fact lost) | Missed merges (duplicate kept) |
|---|---|---|
| rules alone (before the judge) | 6 / 39 | 4 / 36 |
| judge alone, p ≥ 0.9 | 1 | 0 |
| judge alone, p ≥ 0.95 | 0 | 2 |
| rules + judge veto, merge_min 0.5–0.9 | **0** | 4 |
| rules + veto + judge rescue (cos ≥ 0.4, p ≥ 0.97) | **0** | **0** |

- **The rules' wrong merges** were "added, not replaced" (4: "You also like pizza" overwrote
  "Your favorite food is sushi") and "a person and their thing" (2). The judge scored all
  six at 0.00–0.03, so its veto removes them.
- **Right merges** scored 0.92–1.00. `merge_min` 0.9 keeps all of them with margin; 0.95
  would lose two corrections.
- **Same first name, different people** is the judge's weak spot: 0.79 and 0.95. The rules
  never proposed those, and `DistinctSubjects` blocks them in the rescue path. That is why
  the judge must not decide a merge alone below 0.97.
- **The rescue** merges reworded slots the rules miss ("Your dog's name is Rex" → "Your dog
  is called Max"). The judge scored all 12 reworded pairs 0.98–0.99.
- **A caveat:** rescue_min and rescue_cos were tuned on these same 75 pairs, and the closest
  negative (0.95) is near the 0.97 bar. A larger set should confirm them.
- **Speed:** about 32 s per judgement on 4 CPU cores. One judgement per new fact that has a
  candidate, so it is affordable on a GPU and slow on a CPU.

### The yes/no habit: the fix, not yet measured

**The problem.** After deep sleep, models answer yes/no questions with a fact sentence:
"Is my friend the one whose city is Porto? Answer yes or no." gets "Your friend's city is
Porto." Verify fell to 0/24 at 1.5B and 5/24 at 24B. Every training example had one form,
question → fact sentence, so that is the form the model learned to give.

**The fix** (both trainers, `memlib` and `Mem.Examples` / `Mem.Train`):
- **Varied examples for every profile.** Small, medium and large now use them, as tiny
  already did. Each fact becomes a full answer, a short answer, the value asked from the
  other side, yes/no checks, fill in the blank, two facts together, and "It's <value>."
- **Short yes/no examples.** On even rounds the yes/no checks take the form users ask in:
  "Is Ana's city Lisbon? Answer yes or no." → "Yes.", and "Is Ana's city Porto? …" → "No.",
  using another fact's value of the same attribute when there is one.
- **Yes/no anchors.** Six general yes/no prompts ("Is Paris the capital of France? Answer
  yes or no.") come first among the anchors, answered by the base model itself, so every
  profile rehearses answering yes or no. The eight held-out drift prompts are unchanged.

**Not measured.** The GPU rerun of deep sleep at 1.5B, 4B and 24B, with the sleep and
weights arms of the integration benchmark, has not been done. Until it is, the effect on
verify scores, and on drift and recall, is unknown.

### Derived facts (`[agent] derive_facts`, `TMemStore.SyncDerived`)

**What they are.** A relation fact ("Your mother's name is Mia.") joined with a fact about
that person ("Mia's job is nurse.") is stored as one more fact: "Your mother Mia's job is
nurse." It is a plain join, no model, after AI-native memory (2406.18312), A-MEM
(2502.12110) and GraphRAG (2404.16130).

**Result.** Agent rows, 1.5B, retrieve twice, binding-guard build:

| | direct | bridge | inverse | verify | all |
|---|---|---|---|---|---|
| two hops | 24 | 12 | 6 | 16 | 58/96 |
| **two hops + derived facts** | 24 | **22** | **13** | **19** | **78/96** |

- **Why it works.** Questions that needed two linked facts become one-fact lookups, and the
  small model reads the answer off: "Mia works as a nurse."
- **The guard.** It withheld 11 answers, against 29 without derived facts, because answers
  now match a stored fact.
- **The remaining misses.** Two bridge answers abstained, and one said "He pilots airplanes."
  for "pilot".
- **Overall.** Retrieval changes alone took this agent from 40 to 78 of 96, with no
  training. That is ahead of the GPU-trained 4B memory's 77 (Python benchmark, a different
  harness).
- **Default.** Derived facts stay on.

## Checking what won (2026-11-07/08)

Each change below was run against the measurement it was meant to move, and kept only if
it moved it.

### Every fact in the prompt: lost (`[agent] rag_all`)

The baseline long-context papers argue for: no retrieval, all 112 facts in the prompt.
Agent rows, 1.5B, `integration-small-allfacts.json`:

| | direct | bridge | inverse | verify | all | time |
|---|---|---|---|---|---|---|
| every fact in the prompt | 20 | 12 | 4 | 17 | 53/96 | 6092 s |
| two hops + derived facts | 24 | 22 | 13 | 19 | 78/96 | about 2400 s |

With every fact in front of it, the small model binds values to the wrong people, and it is
2.5 times slower. `rag_all` stays off by default.

### A guard licence for derived facts: won (`Mem.Guard.DerivedLicenses`)

Ten of the eleven inverse failures in the derived-facts run were correct answers that the
rule guard withheld. "Which of my relatives works as a nurse?" answered "Your sister Ana."
names nobody in the question, so the binding check could not license it. A derived fact
now licenses an answer when the question carries its value and the answer names its
person or relation, or the other way round. `integration-small-derive-guard.json`:

| | direct | bridge | inverse | verify | all |
|---|---|---|---|---|---|
| two hops + derived facts | 24 | 22 | 13 | 19 | 78/96 |
| **+ derived-fact licence** | **24** | **22** | **22** | **20** | **88/96** |

The eight misses left:
- **Yes/no errors by the model (4).** For example, "Is my aunt the one whose city is
  Utrecht?" answered "No".
- **Abstained with the facts stored (2).**
- **Right but marked wrong by the scorer (1).** "He pilots airplanes." for "pilot".
- **A wrong answer that got through (1).** "Lena lives in Porto." when the answer is Greta.

### The merge judge on pairs it was not tuned on: kept, with a known cost

The thresholds (veto 0.9, rescue 0.97 at cos 0.4) were picked on the 75 pairs above. To test
them, `make_merge_cases_heldout.py` builds 68 new pairs: new names and values, rewordings
not in the tuning set, and two new trap kinds (a person against their pet, and a past
value "used to live in" against the current one).

| | wrong merges (true fact lost) | missed merges (duplicate kept) | cost (5 per wrong) |
|---|---|---|---|
| rules alone | 18 of 30 | 7 of 38 | 97 |
| **agent: rules + veto 0.9 + rescue** | **0 of 30** | **14 of 38** | **14** |
| judge alone, p ≥ 0.95 | 0 of 30 | 8 of 38 | 8 |

- **The veto holds on new data.** It makes no wrong merges, where the rules overwrite a
  person with their pet 8 times and a current city with a past one 6 times.
- **The rescue does not carry over.** 7 of the 8 reworded attributes were missed, though
  the judge gave each p ≥ 0.98 and their cosines are 0.74 to 0.94. The blocker is
  `DistinctSubjects`. Its synonym table (`Related` in Mem.Guard) groups "job" with "work",
  but not with "profession", "occupation" or "trade", and "city" with nothing. So "Oskar's
  job" against "Oskar's profession" reads as two different attributes, and the rescue is
  never asked.
- **Not fixed here.** Adding those words is one line, but it would be tuned on these very
  pairs, leaving no held-out set. The next step is to add the groups and measure them on a
  third set. Meanwhile, a missed merge only keeps a duplicate line, and nothing is lost.
