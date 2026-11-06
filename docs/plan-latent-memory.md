# Plan: a learned memory reader (after Memorilla, arXiv 2610.21375)

## What the paper does

Memorilla (Fallahpour et al., Stanford, Oct 2026) keeps the language model frozen and
never writes facts into it:

- **Encode once.** A frozen embedding model encodes every stored document once. Only new
  or changed documents are re-encoded.
- **Read per question.** A small trained module reads *all* stored embeddings for each
  question: a Perceiver, with cross-attention conditioned on the question and two read
  blocks.
- **Inject.** It emits K = 16 soft tokens, which are put in front of the frozen
  decoder's input.

Only the module is trained: about 194M parameters, with Qwen3-8B as the decoder and
Qwen3-Embedding-4B as the encoder. Training used 3.3M examples on 4 H100s.

The result that matters here is on FactKG, by the number of facts a claim combines.
Top-50 RAG falls from 78.7 to 51.8 between 1 and 4 combined facts, while Memorilla
stays between 80 and 89. The paper puts this down to retrieval never fetching some of
the evidence, and our bridge questions fail the same way. On the integration benchmark,
retrieval found both facts for 12.5% of bridge questions and 0% of inverse questions,
and the deep LoRA did not make up for it (bench/README.md).

Neither the post nor the paper covers the following, and our memory needs all of them:

- **Deleting one fact.** The paper only says a store can be deleted, with no mechanism.
- **Drift.** The paper does not measure it. Our LoRA path shows it plainly at 1.5B, but
  the frozen decoder should avoid it altogether.
- **Provenance for the guard.** Soft tokens cannot be cited.
- **Training cost at our scale.** Gradients must flow through the frozen decoder,
  and the paper only reports runs on H100s.

## What it changes in this project

The LoRA tiers write facts into the weights. The edit experiment (trainer/edit_facts.py)
writes them faster and with less drift. Both still leave the read path as top-k
retrieval, and that is where two-hop questions fail. The paper's idea, a reader trained
to look across the whole store, attacks that read path directly and leaves the store
exact: forgetting deletes a row, and nothing drifts.

## Phases

Each phase has a measurable gate on the integration benchmark
(bench/bench_integration.py) before the next one starts.

### 0. Baseline: two-hop retrieval without learning (1 day, CPU)

Retrieve, then retrieve again with the names and values of the first hits added to the
query: "my sister" → "Your sister's name is Ana" → "Ana's city is Lisbon". The guard
keeps working unchanged, because every fact still enters the prompt as text.

- **Gate:** bridge coverage from 12.5% to 60% or more; inverse questions improve too;
  direct questions stay at 24/24.
- **Why first:** any learned reader has to beat this cheap baseline, and the paper
  does not compare against one.

### 1. A learned reader that selects facts (1–2 weeks, CPU)

The Memorilla module, used as a retriever rather than a soft-token source.

- **How it works:** query-conditioned cross-attention with two read blocks over the
  embeddings the store already keeps (the `emb` column, bge-small/MiniLM). It scores
  each fact and puts the top facts into the prompt as text, as today.
- **Training:** a cross-entropy loss on which facts each question needs. Training needs
  no gradient through the decoder, so a module of a few million parameters trains on
  4 CPU cores.
- **Training data:** synthetic stores from a generalized `gen_integration.py`: random
  people, relations, attributes and look-alike strangers, with direct, bridge, inverse
  and verify questions and their `needs`. Names, values and question templates for
  evaluation are held out from training.
- **Runtime:** export to ONNX and run it in the agent through the existing
  onnxruntime (Mem.Onnx). Facts still enter the prompt as text, so the guard and
  forgetting are unchanged.
- **Gate:** it beats phase 0 on bridge and inverse coverage on held-out templates and
  names, with no loss on direct questions or on the look-alike benchmark
  (bench_retrieval, 2,000 look-alikes, hit@3 ≥ 0.95).

### 2. Soft-token memory, the paper's own method (pilot on the tiny profile)

The module's output goes into the decoder as K soft tokens, as in the paper.

- **Engine work:** `llama_batch` already has an `embd` field (Mem.Llama.pas), so
  Mem.Engine.Llama can feed the K vectors before the prompt tokens.
- **Pilot:** train against the tiny profile (SmolLM2-135M) on CPU, with the decoder
  frozen and gradients flowing only into the module. At 135M a step takes a fraction of
  a second, so tens of thousands of examples fit in a few hours.
- **1.5B:** needs a GPU. On these CPUs the backward pass through 1.5B takes seconds per
  example.
- **Provenance:** the module's read attention over facts gives citations for the
  guard. Keep the text of the top-attended facts in the prompt too: the paper's own
  analysis suggests text retrieval and latent memory together beat either alone.
- **Gate:** on tiny, soft tokens plus text beat phase 1 on the integration questions
  with no regression on the guard benchmark. Otherwise stop at phase 1.

### 3. How it fits the tiers

- **Read path:** the store plus the learned reader becomes the main read path.
- **Weights:** the weight tiers (edit or LoRA) remain a backup for facts the reader
  misses.
- **Sleep:** sleep could fine-tune the reader on the user's own store, which is cheap
  and causes no decoder drift, instead of retraining the decoder.

## Risks

- **Synthetic-to-real gap.** A reader trained on templated facts may not handle
  naturally phrased ones. Evaluate on the natural-phrasing set from memory loop 1.
- **Embedder capacity.** Small embedders may not separate "Ana's city" from "Ana's pet"
  well enough for a reader to bind them. Fallback: Qwen3-Embedding-0.6B through ONNX.
- **Soft tokens on a small decoder.** A 135M–1.5B decoder may not use them well. The
  paper ran only Qwen3 4B and 8B decoders.
