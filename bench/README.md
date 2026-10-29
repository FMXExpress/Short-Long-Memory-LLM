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
| baseline | 0.783 | 0.933 | 0.367 | 0.067 | see below | see below | 1.78 | 412 |
| E1: names + negatives 0.5 + RAFT 0.5 | 0.617 | 0.917 | 0.367 | 0.033 | 0.65 | 0.73 | 1.42 | 504 |
| E2: names only | 0.767 | 0.900 | 0.367 | 0.100 | 0.60 | 0.73 | 1.50 | 408 |

E1 finding: at 135M parameters, the abstention examples teach "unfamiliar wording → abstain"
rather than "unknown subject → abstain". The LoRA refused 16 of 60 questions about facts it knew,
and it still confabulated whenever the wording looked familiar ("What's my horse's name?" →
"Paint"). This matches Gekhman et al. (EMNLP'24): fine-tuning on new knowledge raises
hallucination, and a model's sense of what it knows is hard to train. Negatives and RAFT stay off
for `tiny`.

E2 finding: the generic name-question shapes give no accuracy gain on their own (0.767 vs 0.783,
within run-to-run noise), so they stay off. The baseline is being re-scored with the 20-question
unknown set so its confabulation numbers are comparable.

## Other findings from this round

- **Extraction:** the rule extractor's value accuracy went from 0.77 to 1.00 on dev and is 1.00
  on held-out. The fixes strip leading "a/on/called" from values and stop sentence splits at
  "Dr.".
- **RAG policy:** LoRA+RAG beats the LoRA alone by 15 points, and retrieval is capped at `rag_k`
  either way, so baked facts now stay retrievable (`rag_include_baked=1`).
- **Retrieval sweep** (`BENCH_SWEEP=1`): a cosine floor of 0.25 instead of 0.35, and ms-marco
  `rerank_min` 1e-4 instead of 0.005, raise hit@3 from 0.75 to about 0.97. Near-miss noise is
  structural (e.g. "horse's name" lands next to "dog's name"). The defaults are not changed yet;
  that needs a re-run with the 20 unknown questions.
- **Reranker batching:** the float model (ms-marco) scores in one padded session run and matches
  per-pair scores exactly. The int8 model (bge) scores per pair, because its dynamic activation
  quantization makes batched scores depend on their batch-mates (up to 0.17 apart).
