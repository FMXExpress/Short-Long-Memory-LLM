import os
import json
gc = __import__('gc')
import torch
import uuid
import threading

# Environment configurations
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

def _disable_chroma_telemetry():
    try:
        import chromadb  # noqa: F401
        from chromadb.telemetry.telemetry import TelemetryProxy
        TelemetryProxy.capture = staticmethod(lambda *args, **kwargs: None)
        TelemetryProxy.capture_event = staticmethod(lambda *args, **kwargs: None)
    except Exception:
        pass
_disable_chroma_telemetry()

# === SDPA‐attention fix: flatten extra dims ===
import transformers.integrations.sdpa_attention as _sdpa

def repeat_kv(hidden_states, num_key_value_groups):
    # collapse (batch, groups, heads, seq_len, head_dim) → (batch, groups*heads, seq_len, head_dim)
    if hidden_states.ndim == 5:
        b, g, h, seq_len, head_dim = hidden_states.shape
        return hidden_states.reshape(b, g * h, seq_len, head_dim)
    return hidden_states

_sdpa.repeat_kv = repeat_kv

# === MemCube Configuration ===
from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

MEMOS_CONFIG_PATH = "examples/data/config/simple_memos_config.json"
MEM_CUBE_PATH      = "examples/data/mem_cube_2"
USER_ID            = "b41a34d5-5cae-4b46-8c49-d03794d206f5"
CHAT_HISTORY_FILE  = "chat_history.jsonl"
TOP_K              = 5
MAX_NEW_TOKENS     = 1024

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
    output_dir="chat_history_lora",
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True)
    if not os.path.exists(CHAT_HISTORY_FILE):
        open(CHAT_HISTORY_FILE, "w").write(json.dumps({"input":"","output":""})+"\n")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = get_peft_model(base_model, PEFT_CONFIG).to(device)
    ds = format_chat_dataset(CHAT_HISTORY_FILE, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    torch.cuda.empty_cache(); gc.collect()
    trainer = SFTTrainer(model=model, args=TRAINING_ARGS, train_dataset=ds, data_collator=data_collator, peft_config=PEFT_CONFIG)
    trainer.train()
    model.save_pretrained("chat_history_lora", safe_serialization=True)
    tokenizer.save_pretrained("chat_history_lora")

# === Model Loading ===
BASE_MODEL_ID = "unsloth/Magistral-Small-2506-bnb-4bit"
def load_model_with_lora():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    base_model.gradient_checkpointing_enable()
    if os.path.isdir("chat_history_lora") and os.listdir("chat_history_lora"):
        model = PeftModel.from_pretrained(base_model, "chat_history_lora").to(device)
    else:
        train_lora()
        model = PeftModel.from_pretrained(base_model, "chat_history_lora").to(device)
    return model, tokenizer

# === MemCube Initialization ===
mos_config = MOSConfig.from_json_file(MEMOS_CONFIG_PATH)
memory     = MOS(mos_config)
memory.create_user(user_id=USER_ID)
memory.register_mem_cube(MEM_CUBE_PATH, user_id=USER_ID)

# Optionally preload existing chat history into memory
if os.path.exists(CHAT_HISTORY_FILE):
    for line in open(CHAT_HISTORY_FILE, encoding="utf-8"):
        rec = json.loads(line)
        memory.add(
            messages=[
                {"role": "user",      "content": rec["input"]},
                {"role": "assistant", "content": rec["output"]},
            ],
            user_id=USER_ID,
        )

# === RAG with MemCube ===
def retrieve_context(question, memory, user_id, k=TOP_K):
    results = memory.search(query=question, user_id=user_id)
    texts   = results.get("text_mem", [])[:k]
    if not texts:
        return "No relevant context found."
    return "\n\n".join(f"- {t}" for t in texts)

# === Chat/Inference ===
def chat_and_record(model, tokenizer, memory, user_id):
    question = input("Question: ").lstrip("Q:")
    context  = retrieve_context(question, memory, user_id)
    print("🛈 RAG context:\n", context)
    prompt   = INFER_PROMPT_PREFIX.format(context=context, question=question)
    inputs   = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            **dict(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id
            ),
            'streamer': streamer,
        }
    )
    thread.start()
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
    if "<answer>" in answer and "</answer>" not in answer:
        answer += "</answer>"

    # Record in chat history and memory
    record = {"input": question, "output": answer}
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    memory.add(
        messages=[
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ],
        user_id=user_id,
    )

# === Main ===
def main():
    model, tokenizer = load_model_with_lora()
    while True:
        chat_and_record(model, tokenizer, memory, USER_ID)

if __name__ == "__main__":
    main()
