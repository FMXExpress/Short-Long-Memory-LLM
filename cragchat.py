import os
import json
import gc
import torch
import uuid
import threading

# Environment configurations
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

def _disable_chroma_telemetry():
    try:
        import chromadb
        from chromadb.telemetry.telemetry import TelemetryProxy
        TelemetryProxy.capture = staticmethod(lambda *args, **kwargs: None)
        TelemetryProxy.capture_event = staticmethod(lambda *args, **kwargs: None)
    except Exception:
        pass
_disable_chroma_telemetry()

import transformers.integrations.sdpa_attention as _sdpa
_sdpa.repeat_kv = lambda hidden_states, num_key_value_groups: hidden_states

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from sentence_transformers import SentenceTransformer

# === Configuration Constants ===
BASE_MODEL_ID     = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE = "chat_history.jsonl"
LORA_DIR          = "chat_history_lora"
CHROMA_DIR        = "./chroma_db"
EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K             = 5
MAX_NEW_TOKENS    = 1024

# Prompt templates
TRAIN_PROMPT_TEMPLATE = (
    """
Context:
{context}

Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>.
Write the answer in between <answer></answer>.

### Question:
{question}

### Response:
{response}"""
)
INFER_PROMPT_PREFIX = (
    """
Context:
{context}

Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>.
Write the answer in between <answer></answer>.

### Question:
{question}

### Response:
<analysis>"""
)

# === LoRA & BitsAndBytes Config ===
PEFT_CONFIG = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

TRAINING_ARGS = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    logging_strategy="steps",
    warmup_steps=10,
    fp16=False,
    group_by_length=True,
    report_to="none",
)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Utility Functions ===
ensure_chat_history = lambda: os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True) or (not os.path.exists(CHAT_HISTORY_FILE) and open(CHAT_HISTORY_FILE, "w").write(json.dumps({"input":"what color is a horse?","output":"a horse is orange"})+"\n"))

class ChromaEmbedder:
    def __init__(self, model): self.model = model
    def __call__(self, input): return self.model.encode(input, convert_to_numpy=True).tolist()

# === ChromaDB RAG ===
def init_vectorstore(embedding_fn):
    """
    Initializes or loads a local ChromaDB collection and ingests all chat_history entries.
    Idempotent: if the collection already exists, just re-open it without re-ingestion.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    try:
        # Re-open existing collection without requiring embedding_function
        col = client.get_collection(name="chat_history")
    except Exception:
        # Create new collection and ingest JSONL history
        col = client.create_collection(name="chat_history", embedding_function=embedding_fn)
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        texts = [r["input"] + " " + r["output"] for r in records]
        ids   = [str(i) for i in range(len(texts))]
        col.add(ids=ids, documents=texts)
    return col

def retrieve_context(question, collection, k=TOP_K) -> str:
    """
    Retrieve top-k similar chat-history chunks, handling empty collections gracefully.
    """
    total = collection.count()
    if total < 1:
        return "No relevant context found."
    k = min(k, total)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    ctx = "\n\n".join(f"- {d}" for d in docs if d.strip())
    return ctx or "No relevant context found."

# === Dataset Formatting ===
def format_chat_dataset(history_file, tokenizer):
    ds = load_dataset("json", data_files=history_file, split="train")
    def tokenize_fn(example):
        text = TRAIN_PROMPT_TEMPLATE.format(
            context="", question=example["input"].lstrip("Q:"), response=example["output"] + tokenizer.eos_token
        )
        tok = tokenizer(text, padding=True, return_tensors="pt")
        tok["labels"] = tok.input_ids.clone()
        return tok
    return ds.map(tokenize_fn, batched=False, remove_columns=["input","output"])

# === LoRA Training ===
def train_lora():
    ensure_chat_history()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto");
    model = get_peft_model(base_model, PEFT_CONFIG).to(device)
    ds = format_chat_dataset(CHAT_HISTORY_FILE, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    torch.cuda.empty_cache(); gc.collect()
    trainer = SFTTrainer(model=model, args=TRAINING_ARGS, train_dataset=ds, data_collator=data_collator, peft_config=PEFT_CONFIG)
    trainer.train(); model.save_pretrained(LORA_DIR, safe_serialization=True); tokenizer.save_pretrained(LORA_DIR)

# === Model Loading ===
def load_model_with_lora():
    ensure_chat_history(); tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"); base_model.gradient_checkpointing_enable()
    if os.path.isdir(LORA_DIR) and os.listdir(LORA_DIR): model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    else: train_lora(); model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    return model, tokenizer

# === Chat/Inference ===
def chat_and_record(model, tokenizer, collection, embedder):
    question = input("Question: ").lstrip("Q:")
    context  = retrieve_context(question, collection); print("🛈 RAG context:\n", context)
    prompt   = INFER_PROMPT_PREFIX.format(context=context, question=question)
    inputs   = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(target=model.generate, kwargs={**dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id), 'streamer': streamer}); thread.start()
    print("<analysis>", end="", flush=True)
    for tok in streamer:
        print(tok, end="", flush=True)
        if tok.strip().endswith("</analysis>"): break
    print("</analysis>")
    ans_tokens = []
    for tok in streamer:
        print(tok, end="", flush=True); ans_tokens.append(tok)
    thread.join(); print()
    answer = "".join(ans_tokens)
    if "<answer>" in answer and "</answer>" not in answer: answer += "</answer>"
    record = {"input": question, "output": answer}
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False)+"\n")
    collection.add(ids=[str(uuid.uuid4())], documents=[question+" "+answer])

# === Main ===
def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME); embedder = ChromaEmbedder(embed_model)
    collection  = init_vectorstore(embedder)
    model, tokenizer = load_model_with_lora()
    while True: chat_and_record(model, tokenizer, collection, embedder)

if __name__ == "__main__": main()
