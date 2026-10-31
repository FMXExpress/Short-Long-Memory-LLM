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
