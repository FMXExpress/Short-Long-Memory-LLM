from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import gc, torch


def train():
    model_dir = "unsloth/Magistral-Small-2506-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)

    from trl import SFTTrainer
    from transformers import TrainingArguments


    # Training Arguments
    training_arguments = TrainingArguments(
        output_dir="Magistral-Medical-Reasoning",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="none"
    )

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for question, response in zip(inputs, outputs):
            # Remove the "Q:" prefix from the question
            question = question.replace("Q:", "")

            # Append the EOS token to the response if it's not already there
            if not response.endswith(tokenizer.eos_token):
                response += tokenizer.eos_token

            text = train_prompt_style.format(question, response)
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset(
        "mamachang/medical-reasoning",
        split="train",
        trust_remote_code=True,
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    print(dataset["text"][10])


    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    trainer.train()

    del model
    del trainer
    torch.cuda.empty_cache()


train()
exit()

train_prompt_style = """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. Write the answer in between <answer></answer>.
### Question:
{}

### Response:
{}"""

inference_prompt_style = """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. Write the answer in between <answer></answer>.

### Question:
{}

### Response:
<analysis>
"""

# Base model
base_model_id = "unsloth/Magistral-Small-2506-bnb-4bit"

# Your fine-tuned LoRA adapter repository
#lora_adapter_id = "kingabzpro/Magistral-Small-Medical-QA"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for question, response in zip(inputs, outputs):
        # Remove the "Q:" prefix from the question
        question = question.replace("Q:", "")

        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token

        text = train_prompt_style.format(question, response)
        texts.append(text)
    return {"text": texts}

dataset = load_dataset(
    "mamachang/medical-reasoning",
    split="train",
    trust_remote_code=True,
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)
print(dataset["text"][10])

question = dataset[10]['input']
question = question.replace("Q:", "")

inputs = tokenizer(
    [inference_prompt_style.format(question) + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")

outputs = base_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0].split("### Response:")[1])
exit()



# Attach the LoRA adapter
#model = PeftModel.from_pretrained(
#    base_model,
#    lora_adapter_id,
#    device_map="auto",
#    trust_remote_code=True,
#)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Inference example
prompt = """
Please answer with one of the options in the bracket. Write reasoning in between <analysis></analysis>. Write the answer in between <answer></answer>.

### Question:
A research group wants to assess the relationship between childhood diet and cardiovascular disease in adulthood.
A prospective cohort study of 500 children between 10 to 15 years of age is conducted in which the participants' diets are recorded for 1 year and then the patients are assessed 20 years later for the presence of cardiovascular disease.
A statistically significant association is found between childhood consumption of vegetables and decreased risk of hyperlipidemia and exercise tolerance.
When these findings are submitted to a scientific journal, a peer reviewer comments that the researchers did not discuss the study's validity.
Which of the following additional analyses would most likely address the concerns about this study's design?
{'A': 'Blinding', 'B': 'Crossover', 'C': 'Matching', 'D': 'Stratification', 'E': 'Randomization'},
### Response:
<analysis>

"""

inputs = tokenizer(
    [prompt + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0].split("### Response:")[1])