import os
import json
gc = __import__('gc')
torch = __import__('torch')
import uuid
import threading

# Environment configurations
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

import transformers.integrations.sdpa_attention as _sdpa

# Override repeat_kv for efficient SDPA attention

def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    hidden_states: (batch, num_kv_heads, seq_len, head_dim)
    returns:      (batch, num_kv_heads * num_key_value_groups, seq_len, head_dim)
    """
    return hidden_states.repeat_interleave(num_key_value_groups, dim=1)

_sdpa.repeat_kv = repeat_kv

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

# MemCube (MemOS) imports
from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

# === Configuration Constants ===
BASE_MODEL_ID      = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE  = "chat_history.jsonl"
LORA_DIR           = "chat_history_lora"
MEMOS_CONFIG_PATH  = "examples/data/config/simple_memos_config.json"
MEM_CUBE_PATH      = "examples/data/mem_cube_2"
USER_ID            = "b41a34d5-5cae-4b46-8c49-d03794d206f5"
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

# === MemCube Initialization & Retrieval ===
def init_memory(config_path: str, cube_path: str, user_id: str) -> MOS:
    """
    Initializes MemOS (MemCube) for the given user.
    Idempotent: re-creates user and registers cube if not exists.
    """
    mos_config = MOSConfig.from_json_file(config_path)
    memory = MOS(mos_config)

    try:
        memory.create_user(user_id=user_id)
    except Exception:
        pass
    memory.register_mem_cube(cube_path, user_id=user_id)
    return memory


def retrieve_context(question: str, memory: MOS, user_id: str, k: int = TOP_K) -> str:
    """
    Retrieve top-k memories for the question from MemCube.
    """
    results = memory.search(query=question, user_id=user_id)
    texts = results.get('text_mem', [])
    if not texts:
        return "No relevant context found."
    snippets = texts[:k]
    return "\n\n".join(f"- {s}" for s in snippets if s.strip()) or "No relevant context found."

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
        with open(CHAT_HISTORY_FILE, "w") as f:
            f.write(json.dumps({"input":"what color is a horse?","output":"a horse is orange"})+"\n")
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
    os.makedirs(LORA_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto");
    base_model.gradient_checkpointing_enable()
    if os.path.isdir(LORA_DIR) and os.listdir(LORA_DIR):
        model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    else:
        train_lora()
        model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    return model, tokenizer

# === Chat/Inference ===
def chat_and_record(model, tokenizer, memory: MOS, user_id: str):
    question = input("Question: ").lstrip("Q:")
    context = retrieve_context(question, memory, user_id)
    print("🛈 RAG context:\n", context)
    prompt = INFER_PROMPT_PREFIX.format(context=context, question=question)
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': MAX_NEW_TOKENS,
            'eos_token_id': tokenizer.eos_token_id,
            'streamer': streamer,
        }
    )
    thread.start()
    full_response = ""
    for tok in streamer:
        print(tok, end="", flush=True)
        full_response += tok
    thread.join()
    print()
    if "<answer>" in full_response and "</answer>" in full_response:
        answer = full_response.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        answer = ""

    # Record the Q&A in history
    record = {"input": question, "output": answer}
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Add to MemCube using text-specific API
    memory.add_text(
        memory=question,
        metadata={"role": "user"},
        user_id=user_id,
    )
    memory.add_text(
        memory=answer,
        metadata={"role": "assistant"},
        user_id=user_id,
    )

# === Main ===
def main():
    memory = init_memory(MEMOS_CONFIG_PATH, MEM_CUBE_PATH, USER_ID)
    model, tokenizer = load_model_with_lora()
    while True:
        chat_and_record(model, tokenizer, memory, USER_ID)

if __name__ == "__main__":
    main()
