import os
import json
import gc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# === Configuration Constants ===
BASE_MODEL_ID = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE = "chat_history.jsonl"
LORA_DIR = "chat_history_lora"
TRAIN_PROMPT_TEMPLATE = (
    """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. "
    "Write the answer in between <answer></answer>.
### Question:
{}

### Response:
{}"""
)
INFER_PROMPT_PREFIX = (
    """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. "
    "Write the answer in between <answer></answer>.

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


def format_chat_dataset(history_file: str, tokenizer) -> torch.utils.data.Dataset:
    """
    Loads JSONL chat history and applies prompt formatting.
    """
    ds = load_dataset("json", data_files=history_file, split="train")

    def _format(ex):
        q, a = ex["input"].lstrip("Q:"), ex["output"]
        if not a.endswith(tokenizer.eos_token):
            a += tokenizer.eos_token
        return {"text": TRAIN_PROMPT_TEMPLATE.format(q, a)}

    return ds.map(lambda batch: {"text": [_format(x)["text"] for x in zip(batch["input"], batch["output"])]}, batched=True)


def train_lora():
    """
    Fine-tunes the base model with LoRA on chat history and saves adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    model = get_peft_model(model, PEFT_CONFIG)

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
    Loads base model and attaches LoRA if available, else trains a new adapter.
    """
    # Load base
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base.config.use_cache = False

    # Load or train LoRA
    if os.path.isdir(LORA_DIR) and os.listdir(LORA_DIR):
        model = PeftModel.from_pretrained(base, LORA_DIR, device_map="auto")
    else:
        train_lora()
        model = PeftModel.from_pretrained(base, LORA_DIR, device_map="auto")

    return model


def chat_and_record(model):
    """
    Runs a single inference, prints and appends to chat history.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    question = input("Question: ").lstrip("Q:")

    prompt = INFER_PROMPT_PREFIX.format(question) + tokenizer.eos_token
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = resp.split("### Response:")[-1]
    print(answer)

    record = {"input": question, "output": answer}
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True)
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    model = load_model_with_lora()
    chat_and_record(model)


if __name__ == "__main__":
    main()
