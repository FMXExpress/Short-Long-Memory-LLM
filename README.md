# memagent: Short/Long-Term Memory Agent (Free Pascal)

An always-on agent with two tiers of memory:

- **Short-term:** the chat context, plus RAG over earlier messages and facts: SQLite FTS5 plus a
  local embedding model, fused by reciprocal rank, then re-scored by a local cross-encoder
  reranker. Both models run in-process from Pascal through ONNX Runtime.
- **Long-term:** facts *baked into a LoRA*. A background trainer (on the second GPU, or the CPU)
  learns the facts, the new adapter is checked against an eval gate, and it is **hot-swapped**
  into the running model between turns.

The agent (loop, prompts, memory store, RAG, scheduling, adapter bookkeeping) is **Free Pascal**.
The model runs behind one interface (`TEngine`), with two interchangeable engines
(`[agent] engine=` or `--engine`):

| Engine | What runs the model | Python at inference time |
|---|---|---|
| `python` (default) | `trainer/worker.py`: Transformers + PEFT, long-lived, JSON lines over stdin/stdout | yes |
| `llama` | llama.cpp **inside the agent process** (`Mem.Llama` binding, `Mem.Engine.Llama`): GGUF base model, GGUF LoRA hot-swapped with `llama_set_adapters_lora`, fact extraction ported to Pascal (`Mem.Extract`) | **no** |

Training is Python with either engine, as a one-shot process the agent launches on `[gpu] train`:
`trainer/consolidate.py` turns facts into a paraphrase dataset, trains a LoRA, checks it against a
recall/drift gate and writes `data/adapters/mem-<ts>/` (plus `adapter.gguf` for the llama engine).

**RAG is the write-ahead log for the LoRA.** A new fact is available straight away through RAG.
Once an adapter answers the fact's held-out probe question with no context, the fact is marked
`baked_in` and is no longer put in the prompt. Every consolidation retrains from the base model
on all active facts, so corrections (`My dog's name is Max`) and `/forget` reach the weights
without any un-training logic.

## Model profiles

`models.json` holds everything model-specific. Pick one with `profile=` in `memagent.ini`, or
with `--profile`:

| Profile | Base | Where it runs |
|---|---|---|
| `tiny` | SmolLM2-135M-Instruct | CPU. Development and CI; a consolidation takes about 2 min |
| `small` | Qwen2.5-1.5B-Instruct | one 8 GB+ GPU |
| `medium` | Qwen3-4B-Instruct (4-bit) | two 16–24 GB GPUs |
| `large` | Magistral-Small 24B (4-bit) | two 24 GB+ GPUs, e.g. rented A100/H100 |

The `tiny` profile only extracts facts with rules ("my X is Y", "I live in", "I work at",
"remember that the X is Y"). Bigger profiles add LLM fact extraction, conversation
summaries and tool calls.

## Local retrieval: embedder + reranker (Pascal, ONNX Runtime)

| Stage | Model (default) | What it does |
|---|---|---|
| keyword | SQLite FTS5 / BM25 | exact-word matches |
| vector | all-MiniLM-L6-v2, 384-d, about 90 MB | finds facts by meaning: "Where is my home?" finds "You live in Reykjavik", which shares no word with it |
| fuse | reciprocal rank fusion | merges the two ranked lists into a candidate pool (`pool=8`) |
| rerank | ms-marco-MiniLM-L-6-v2 cross-encoder | re-scores each (query, candidate) pair together; drops the irrelevant ones |

The same two models decide whether a new fact is a **correction** of an existing one ("what my dog
is called" vs "my dog's name"). Cosine similarity alone is not safe for this: MiniLM scores "my
sister's name" vs "my brother's name" (0.85) *higher* than a real paraphrase (0.75). So cosine only
shortlists candidates, and the cross-encoder has to confirm the match. `memagent/tests/retrieval_test.lpr`
checks these cases.

```sh
build/memagent --provision   # ONNX Runtime + embedder + reranker into ./cache (about 180 MB)
```

Without provisioning, memagent falls back to a dependency-free lexical embedder (hashed words and
character n-grams). It tolerates typos but does not understand meaning, and it is never used to
merge facts. Options live in the `[retrieval]` section of `memagent.ini`:
`embed_model=minilm|bge-small|mxbai` and
`rerank_model=ms-marco-minilm|bge-reranker-base|bge-reranker-v2-m3`. `/recall <query>` shows
what the RAG stage would retrieve, with its scores.

**TLS / OpenSSL.** Downloads use `Mem.Https`, a verifying TLS handler on FPC's OpenSSL bindings.
It checks the certificate chain and hostname, which FPC 3.2.2's stock handler does not. It also
tunnels through `HTTPS_PROXY` and loads OpenSSL 3. Linux and macOS use the system OpenSSL 1.1 or 3.
On **Windows**, put `libssl-3-x64.dll` and `libcrypto-3-x64.dll` next to `memagent.exe`; they ship
with curl for Windows and with Lazarus.

## Confabulation guard and abstention training

Fine-tuning facts into a LoRA makes the model eager to answer from memory. Asked "What's my
horse's name?", it replies with the dog's name. Two measured countermeasures:

- **Licensing guard** (`Mem.Guard`, `[guard]` in memagent.ini, on when bge-reranker-base is
  provisioned). For a question about "my X", an answer that uses a stored value belonging to a
  different subject is replaced with "You haven't told me that yet." In the live agent it blocks
  aunt → the sister's name and parrot → the cat's name, and lets dog, sister and cat through. In
  the benchmarks, confabulation goes from 0.70 to 0.25 (tiny) and from 0.55 to 0.20 (small), for
  3-7 points of accuracy.
- **DPO abstention stage** (`train.dpo` in models.json, off by default): preference pairs
  "abstain > the model's own wrong answer" and "fact > abstain". On the 1.5B profile it cuts
  confabulation with a look-alike in memory from 0.87 to 0.13 (0.07 with the guard). On the 135M
  profile it does not generalize. Details and the open ablation are in `bench/README.md`.

## Engines: Python worker or in-process llama.cpp

```sh
make llama                          # = scripts/llama_setup.sh: llama.cpp b11985 as shared libraries in
                                    #   cache/llama, the GGUF converters, cache/gguf/tiny.gguf
make llama PROFILES="tiny small"    # more base models;  LLAMA_CUDA=1 make llama  for the CUDA backend
./build/memagent --engine llama     # or engine=llama in memagent.ini
```

With `engine=llama` the chat model, the LoRA swap and LLM fact extraction all run inside
`memagent`; the only Python left is the trainer, which also writes the adapter as
`adapter.gguf` (`convert_lora_to_gguf.py`) before it announces it, so the swap itself is a file
load (~20 ms on tiny). An adapter trained while the Python engine was selected is converted on
first use. Both engines read the same memory store and adapters, so a data directory can switch
engines at any time.

How the two stay equivalent:

- **Same prompt:** the model's own chat template (`llama_chat_apply_template`), special tokens
  parsed, no extra BOS, as `apply_chat_template` + `add_special_tokens=False` in the worker.
- **Same decoding:** greedy at temperature 0 (top-p 0.9 sampling otherwise) and an HF-style
  repetition penalty of 1.1 over prompt *and* reply (prompt tokens are fed to the penalty sampler).
  On the tiny profile both engines give token-identical replies at temperature 0.
- **Same extraction:** `Mem.Extract` is a port of `memlib.extract_facts_rules`, `has_statement`,
  the LLM extraction prompt, JSON parsing and grounding. `make test` checks it against 232 cases
  dumped from the Python side (`memagent/tests/gen_extract_cases.py`).
- **Same GPU split:** the model is pinned to `[gpu] infer` (`main_gpu`, no layer split), so the
  training GPU stays free.

`Mem.Llama` binds `llama.h` at tag `b11985` (the tag `llama_setup.sh` builds); record layouts are
checked against the C compiler's sizes at compile time, so a header drift fails the build instead
of corrupting memory. Medium/large profiles convert to `q8_0` (`models.json` `"gguf"`); `large`
converts from the full-precision Magistral checkpoint because its training base is bnb-4bit.

## Build and run

```sh
sudo apt install fpc                      # Free Pascal 3.2+
pip install -r trainer/requirements.txt   # CPU only: install torch from the CPU index first
make                                      # -> build/memagent
make provision                            # local embedder + reranker (optional; recommended)
./build/memagent                          # uses memagent.ini; or --profile small --data ./data2
```

`[gpu] infer=0 / train=1` puts inference on GPU 0 and training on GPU 1. Set either to `-1`
to use the CPU.

Commands: `/facts`, `/remember <text>`, `/forget <id>`, `/ab <question>` (base model vs LoRA,
both without RAG), `/sleep` (consolidate now), `/wait`, `/adapter`, `/rollback`, `/status`,
`/quit`. A consolidation also starts on its own once `min_new_facts` facts are unbaked and
`interval_min` minutes have passed since the last one.

## Tests

```sh
make test                              # Pascal extraction parity + retrieval checks, TLS checks, Python unit tests
MEMAGENT_SLOW_TESTS=1 make test        # + the worker protocol test against the real tiny model
make test-e2e                          # real binary + tiny model on CPU (~7 min): semantic recall, two
                                       #   consolidations, a correction, /forget, /rollback, non-ASCII facts
make test-e2e-llama                    # the same on the in-process llama.cpp engine (after make llama)
scripts/resilience_test.sh <e2e dir>   # restart keeps the active adapter (and a rollback), killed worker is
                                       #   restarted with the LoRA re-applied, static-embedder fallback,
                                       #   /quit during training waits and promotes
scripts/profile_smoke.sh small [--train]   # the paths tiny switches off: LLM fact extraction, tool calls,
                                           #   rolling summaries (+ one consolidation with --train)
```

On a GPU box, run `INFER_GPU=0 TRAIN_GPU=1 scripts/profile_smoke.sh medium --train` first: the
`medium` and `large` hyperparameters and gate thresholds have not been tuned on real hardware yet.

Excerpt from an e2e transcript (tiny model, CPU):

```
you> /ab Do you know what my dog's name is?
base, no memory  > I'm sorry for any confusion, but as an AI assistant ... I don't have personal knowledge ...
LoRA, no RAG     > Your dog's name is Biscuit.
you> My dog's name is Max.
[memory ~#4] Your dog's name is Max. (replaces an older value)
you> /forget 3
you> /sleep
[memory] hot-swapped LoRA mem-20261028T220624 -> mem-20261028T220820 in 481 ms: 2/2 facts now in the weights (recall 1.00, held-out ppl x1.25)
you> /ab What's my dog's name?
LoRA, no RAG     > Your dog's name is Max.
```

## Layout

```
memagent/        Free Pascal agent: Mem.Agent, Mem.Store, Mem.Engine (+ .Llama), Mem.Llama (llama.cpp binding),
                 Mem.Extract (fact extraction), Mem.Consolidate, Mem.Db, Mem.Config,
                 Mem.Retrieval (embedder + reranker), Mem.Embed.*, Mem.Rerank.Onnx, Mem.Tokenizer,
                 Mem.SentencePiece, Mem.Onnx.* (ONNX Runtime), Mem.Https + Mem.Download (provisioning)
trainer/         Python worker + consolidation trainer + GGUF export (gguf_export.py) + tests
models.json      model profiles          memagent.ini  agent config          priors.md  system prompt
scripts/         e2e_demo.sh, llama_setup.sh, ...       docs/  sample e2e transcript
PLAN.md          design notes and roadmap (in-process llama.cpp engine, PasClaw integration)
```

---

# Legacy: CragChat (Python prototype)

A lightweight command-line chat application with retrieval-augmented generation (RAG) and LoRA fine-tuning built on top of Hugging Face Transformers.

---

## Features

* **Retrieval-Augmented Generation** (RAG) using ChromaDB to reference past chat history
* **LoRA Fine-Tuning** for efficient continual learning on user conversations
* **4-bit Quantization** with BitsAndBytes for memory-efficient inference
* **Streaming Responses** via `TextIteratorStreamer` to display analysis and answers token-by-token

---

## Requirements

* Python 3.8 or higher
* (Optional) CUDA‑enabled GPU for accelerated training and inference

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/cragchat.git
   cd cragchat
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # macOS / Linux:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, you can install directly:

   ```bash
   pip install torch transformers bitsandbytes peft trl chromadb sentence-transformers datasets
   ```

---

## Usage

Run the chat interface script:

```bash
python cragchat.py
```

* On first run, the script will initialize `chat_history.jsonl`, set up the ChromaDB vector store, and perform any necessary LoRA training.
* Type your questions at the prompt (e.g., `Question: what is a horse?`) and watch the `<analysis>` and `<answer>` stream in real time.

---

## Configuration

All configuration options live at the top of `cragchat.py`:

* **BASE\_MODEL\_ID**: the Hugging Face model identifier (default `unsloth/Magistral-Small-2506-bnb-4bit`)
* **CHAT\_HISTORY\_FILE**: path to the JSONL log of past Q\&A
* **LORA\_DIR**: directory for saving LoRA adapters
* **CHROMA\_DIR**: local path for the ChromaDB database
* **EMBED\_MODEL\_NAME**: Sentence-Transformers model for embeddings
* **MAX\_NEW\_TOKENS**: maximum tokens per generated response

Feel free to adjust batch sizes, learning rates, and other hyperparameters in the code to suit your environment.

---

## License

This project is released under the [MIT License](./LICENSE).
