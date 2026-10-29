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

Tiny DPO finding: the pairs are learned (0.93) but nothing generalizes. The model abstains on 0 of
6 unknown subjects it was not trained on, confabulation is unchanged, and accuracy drops. The same
result as E1: a 135M model memorizes the contrast instead of learning a boundary between known and
unknown. DPO stays off for `tiny`; `small` is the real test.

Anchor-replay finding: rejected. Replaying the 8 anchor answers up to a 30% share raised drift
(2.36, against 1.78): more steps, overfit to those exact anchors, and no protection on the held-out
prompts. Drift appears to track total optimizer steps instead. The next test is fewer epochs.
