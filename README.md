# CragChat

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
