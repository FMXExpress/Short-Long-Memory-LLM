# memagent

An always-on chat agent with two tiers of memory, written in Free Pascal:

- **Short-term:** the conversation plus retrieval over everything you told it (local embedder and
  reranker, run in-process).
- **Long-term:** facts are trained into a LoRA adapter in the background and hot-swapped into the
  running model, all inside the program (llama.cpp).

## Run

```
./memagent            (Linux)
memagent              (Windows: Start menu shortcut, or from a terminal in the install folder)
```

The first start downloads the chat model (SmolLM2-135M, 270 MB) and the retrieval models into
`cache/`; later starts are offline. The model runs inside memagent through llama.cpp, on the CPU.
Tell it things ("My dog's name is Biscuit."), ask about them later, and type `/help` for commands.

Your memory lives in `data/` next to the program.

## Long-term memory: built in

Facts you tell it are trained into a LoRA adapter in the background (`/sleep`, or automatically once a
few new facts pile up) and hot-swapped into the running model. The training runs inside memagent on
llama.cpp too: no Python needed. The first consolidation makes an f32 training copy of the model
(540 MB for `tiny`) in `cache/gguf/`.

The `medium` and `large` profiles still train with Python (an f32 copy of a 4B/24B model is too big):
install Python 3.10+ and `pip install -r trainer/requirements.txt` for those.

## Configure

`memagent.ini` next to the program: the model profile (`tiny` to `large`, see `models.json`), the
engine (`auto`, `llama` or `python`), GPUs (`[gpu] infer`, `train`; -1 = CPU), retrieval and the
consolidation schedule. Command line: `--profile small`, `--engine python`, `--data DIR`,
`--offline`, `--provision` (download everything now and exit).

Source, benchmarks and design notes: https://github.com/FMXExpress/Short-Long-Memory-LLM
