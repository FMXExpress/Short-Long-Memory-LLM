# memagent: Short/Long-Term Memory Agent (Free Pascal)

An always-on agent with two tiers of memory:

- **Short-term:** the chat context, plus RAG over earlier messages and facts: SQLite FTS5 plus a
  local embedding model, fused by reciprocal rank, then re-scored by a local cross-encoder
  reranker. Both models run in-process from Pascal through ONNX Runtime.
- **Long-term:** facts *baked into a LoRA*. A background trainer (on the second GPU, or the CPU)
  learns the facts, the new adapter is checked against an eval gate, and it is **hot-swapped**
  into the running model between turns.

The agent (loop, prompts, memory store, RAG, scheduling, adapter bookkeeping) is **Free Pascal**.
Python only does the tensor work, as two processes the agent launches:

| Process | Lifetime | Device | Job |
|---|---|---|---|
| `trainer/worker.py` | long-lived, JSON lines over stdin/stdout | `[gpu] infer` | generate (streamed), extract facts, load/swap LoRA |
| `trainer/consolidate.py` | one-shot, exit code + JSON summary | `[gpu] train` | facts → paraphrase dataset → LoRA → recall/drift gate → `data/adapters/mem-<ts>/` |

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
make test                          # Pascal retrieval tests + Python unit tests (MEMAGENT_SLOW_TESTS=1 adds the worker protocol test)
make test-e2e                      # real binary + tiny model on CPU: semantic recall, two consolidations, forget, rollback (~7 min)
```

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
memagent/        Free Pascal agent: Mem.Agent, Mem.Store, Mem.Engine, Mem.Consolidate, Mem.Db, Mem.Config,
                 Mem.Retrieval (embedder + reranker), Mem.Embed.*, Mem.Rerank.Onnx, Mem.Tokenizer,
                 Mem.SentencePiece, Mem.Onnx.* (ONNX Runtime), Mem.Https + Mem.Download (provisioning)
trainer/         Python worker + consolidation trainer + tests
models.json      model profiles          memagent.ini  agent config          priors.md  system prompt
scripts/         e2e_demo.sh            docs/  sample e2e transcript
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
