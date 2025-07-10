import os
import json
import gc
import torch

# disable Chroma telemetry as a workaround
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# new imports for Chroma RAG
def _disable_chroma_telemetry():
    try:
        import chromadb
        # Try to monkey-patch telemetry capture to no-op
        from chromadb.telemetry.telemetry import TelemetryProxy
        TelemetryProxy.capture = lambda *args, **kwargs: None
    except Exception:
        pass

_disable_chroma_telemetry()

from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from sentence_transformers import SentenceTransformer

# === Configuration Constants ===
BASE_MODEL_ID      = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE  = "chat_history.jsonl"
LORA_DIR           = "chat_history_lora"
CHROMA_DIR         = "./chroma_db"
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K              = 5

# revised prompt templates to include context placeholder {context}
TRAIN_PROMPT_TEMPLATE = (
    """
Context:
{context}

Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>.
Write the answer in between <answer></answer>.

### Question:
{}

### Response:
{}"""
)
INFER_PROMPT_PREFIX = (
    """
Context:
{context}

Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>.
Write the answer in between <answer></answer>.

### Question:
{}

### Response:
<analysis>"""
)

# === LoRA Training Configuration ===
PEFT_CONFIG = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
TRAINING_ARGS = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_strategy="steps",
    warmup_steps=10,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none",
)

# Determine device: prefer second GPU if available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    DEVICE = torch.device("cuda:1")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def ensure_chat_history():
    """
    Creates a default chat history file if it doesn't exist.
    """
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True)
    if not os.path.exists(CHAT_HISTORY_FILE):
        default_record = {"input": "", "output": ""}
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(default_record, ensure_ascii=False) + "\n")


# === Embedding Wrapper ===
class ChromaEmbedder:
    """
    Wraps a SentenceTransformer model to match Chroma's embedding interface.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        # texts: List[str]
        embeds = self.model.encode(texts, convert_to_numpy=True)
        return embeds.tolist()


# === ChromaDB RAG pieces ===

def init_vectorstore(embedding_fn):
    """
    Initializes or loads a local ChromaDB collection and ingests all chat_history entries.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    existing = client.list_collections()
    if "chat_history" in existing:
        col = client.get_collection(
            name="chat_history",
            embedding_function=embedding_fn
        )
    else:
        col = client.create_collection(
            name="chat_history",
            embedding_function=embedding_fn
        )
        # ingest existing history
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        texts = [r["input"] + " " + r["output"] for r in records]
        ids   = [str(i) for i in range(len(texts))]
        col.add(ids=ids, documents=texts)

    return col


def retrieve_context(question: str, collection, k: int = TOP_K) -> str:
    """
    Retrieves top-k similar chat-history chunks for the given question.
    Returns a formatted string to inject into the prompt.
    """
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    ctx = "\n\n".join(f"- {d}" for d in docs if d.strip())
    return ctx or "No relevant context found."


# === original formatting & training functions, updated to pass context ===

def format_chat_dataset(history_file: str, tokenizer) -> torch.utils.data.Dataset:
    """
    Loads JSONL chat history and applies prompt formatting.
    """
    ds = load_dataset("json", data_files=history_file, split="train")

    def _format_pair(inp: str, out: str) -> str:
        q = inp.lstrip("Q:")
        a = out
        if not a.endswith(tokenizer.eos_token):
            a += tokenizer.eos_token
        return TRAIN_PROMPT_TEMPLATE.format(q, a, context="")

    return ds.map(
        lambda batch: {"text": [
            _format_pair(inp, out)
            for inp, out in zip(batch["input"], batch["output"])
        ]},
        batched=True,
    )

def train_lora():
    """
    Fine-tunes the base model with LoRA on chat history and saves adapter on DEVICE.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.to(DEVICE)

    model = get_peft_model(model, PEFT_CONFIG).to(DEVICE)

    ds = format_chat_dataset(CHAT_HISTORY_FILE, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=ds,
        peft_config=PEFT_CONFIG,
        data_collator=data_collator,
    )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained(LORA_DIR, safe_serialization=True)
    tokenizer.save_pretrained(LORA_DIR)

    del trainer, model
    torch.cuda.empty_cache()


def load_model_with_lora():
    """
    Loads base model and attaches or trains LoRA adapter, ensures adapter is on DEVICE.
    """
    ensure_chat_history()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, use_fast=True, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.to(DEVICE)

    if os.path.isdir(LORA_DIR) and os.listdir(LORA_DIR):
        model = PeftModel.from_pretrained(
            base_model, LORA_DIR
        ).to(DEVICE)
    else:
        train_lora()
        model = PeftModel.from_pretrained(
            base_model, LORA_DIR
        ).to(DEVICE)

    return model, tokenizer


def chat_and_record(model, tokenizer, collection, embedder):
    """
    Runs a single inference on DEVICE with RAG, prints and appends to chat history & vectorstore.
    """
    question = input("Question: ").lstrip("Q:")
    context = retrieve_context(question, collection, k=TOP_K)

    prompt = INFER_PROMPT_PREFIX.format(context, question) + tokenizer.eos_token
    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = resp.split("### Response:")[-1].strip()
    print(answer)

    record = {"input": question, "output": answer}
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True)
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # add to vectorstore
    doc_id = str(len(collection.query(query_texts=[question], n_results=1)["ids"]) + 1)
    collection.add(ids=[doc_id], documents=[question + " " + answer])


def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embedder = ChromaEmbedder(embed_model)
    collection = init_vectorstore(embedder)

    model, tokenizer = load_model_with_lora()

    while True:
        chat_and_record(model, tokenizer, collection, embedder)


if __name__ == "__main__":
    main()
