#pip install transformers accelerate bitsandbytes peft trl datasets chromadb sentence-transformers scipy
import os
import json
import gc
import torch
import uuid
import threading
import logging
from typing import List, Dict, Any, Optional
import re

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
import torch

def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
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
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from sentence_transformers import SentenceTransformer

# Configuration Constants
BASE_MODEL_ID       = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE = "chat_history.jsonl"
LORA_DIR          = "chat_history_lora"
CHROMA_DIR        = "./chroma_db"
EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 512 # Adjusted to be smaller for more granular chunks of analysis/answer
CHUNK_OVERLAP = 100 # Adjusted for more overlap between smaller chunks
MAX_CHUNK_SIZE = 768 # Adjusted to be consistent with CHUNK_SIZE changes
MIN_CHUNK_SIZE = 50 # Adjusted minimum as analysis/answer segments might be shorter
MAX_TRAINING_TOKENS = 1500
TOP_K = 6 # Adjusted TOP_K to retrieve fewer chunks of each type (Analysis & Answer)
MAX_NEW_TOKENS = 1024

# Prompt templates (MODIFIED)
TRAIN_PROMPT_TEMPLATE = """
Context:
{context}

Based on the provided context, first provide your reasoning or background information in an <analysis> tag. Then, provide a direct and concise answer in an <answer> tag.

### Question:
{question}

### Response:
{response}"""

INFER_PROMPT_PREFIX = """
Context (Analyses from relevant past interactions):
{analysis_context}

Context (Answers from relevant past interactions):
{answer_context}

Based on the provided context, first provide your reasoning or background information in an <analysis> tag. Then, provide a direct and concise answer in an <answer> tag.

### Question:
{question}

### Response:
<analysis>"""

# LoRA & BitsAndBytes Config
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
    gradient_accumulation_steps=4,
    num_train_epochs=3, # Increased epochs for better learning of structured output
    bf16=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    logging_strategy="steps",
    warmup_steps=10,
    fp16=False,
    group_by_length=True,
    report_to="none",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Smart Chunking Implementation
class SmartTextChunker:
    def __init__(self, tokenizer, chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP, max_chunk_size: int = MAX_CHUNK_SIZE):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': start_pos,
                        'end_pos': start_pos + len(chunk_text),
                        'token_count': current_tokens,
                        'sentences': len(current_chunk)
                    })

                current_chunk = [sentence]
                current_tokens = sentence_tokens
                start_pos = len(" ".join(sentences[:i]))

            elif current_tokens + sentence_tokens > self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                chunk_text = " ".join(current_chunk)

                chunks.append({
                    'text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(chunk_text),
                    'token_count': current_tokens,
                    'sentences': len(current_chunk)
                })

                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                start_pos = len(" ".join(sentences[:i-len(overlap_sentences)+1]))

            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_pos': start_pos,
                'end_pos': start_pos + len(chunk_text),
                'token_count': current_tokens,
                'sentences': len(current_chunk)
            })

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        overlap_tokens = 0
        overlap_sentences = []

        for sentence in reversed(sentences[:-1]):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

# Utility Functions
def ensure_chat_history():
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or ".", exist_ok=True)
    if not os.path.exists(CHAT_HISTORY_FILE):
        pass

class ChromaEmbedder:
    def __init__(self, model, name):
        self.model = model
        # this is what Chroma expects to see
        self.name = name

    def __call__(self, texts):
        # make sure you accept a list of strings
        # and return a List[List[float]]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

# Improved ChromaDB RAG with Smart Chunking for dual indexing
def init_vectorstore(embedding_fn, tokenizer):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # get_or_create_collection ensures your embedding_fn is used for BOTH add() and query()
    col = client.get_or_create_collection(
        name="chat_history",
        embedding_function=embedder,
    )

    # If it’s a brand‐new collection, ingest your existing chat history
    if col.count() == 0:
        logger.info("Ingesting existing chat history into new collection…")
        chunker = SmartTextChunker(tokenizer)
        total_chunks = 0

        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        for i, record in enumerate(records):
            input_text = record["input"]
            output_text = record["output"]

            # parse analysis and answer as before…
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', output_text, re.DOTALL)
            answer_match   = re.search(r'<answer>(.*?)</answer>',   output_text, re.DOTALL)

            analysis_content = analysis_match.group(1).strip() if analysis_match else ""
            answer_content   = answer_match.group(1).strip()   if answer_match   else ""

            # ingest analysis chunks
            if analysis_content:
                for j, chunk_data in enumerate(chunker.create_chunks(analysis_content)):
                    combined = f"Query: {input_text}\nAnalysis Segment: {chunk_data['text']}"
                    col.add(
                        ids=[f"{i}_analysis_{j}"],
                        documents=[combined],
                        metadatas=[{
                            'record_index': i,
                            'chunk_index': j,
                            'token_count': chunker.count_tokens(combined),
                            'original_segment_type': 'analysis',
                            'original_segment_token_count': chunk_data['token_count'],
                            'sentences': chunk_data['sentences'],
                            'start_pos': chunk_data['start_pos'],
                            'end_pos': chunk_data['end_pos'],
                            'original_input': input_text
                        }]
                    )
                    total_chunks += 1

            # ingest answer chunks
            if answer_content:
                for j, chunk_data in enumerate(chunker.create_chunks(answer_content)):
                    combined = f"Query: {input_text}\nAnswer Segment: {chunk_data['text']}"
                    col.add(
                        ids=[f"{i}_answer_{j}"],
                        documents=[combined],
                        metadatas=[{
                            'record_index': i,
                            'chunk_index': j,
                            'token_count': chunker.count_tokens(combined),
                            'original_segment_type': 'answer',
                            'original_segment_token_count': chunk_data['token_count'],
                            'sentences': chunk_data['sentences'],
                            'start_pos': chunk_data['start_pos'],
                            'end_pos': chunk_data['end_pos'],
                            'original_input': input_text
                        }]
                    )
                    total_chunks += 1

        logger.info(f"Ingested {len(records)} records into {total_chunks} total chunks.")
        # make sure it’s written to disk so future restarts see it
        client.persist()
    else:
        logger.info(f"Loaded collection with {col.count()} vectors")

    return col

# Modified retrieve_context to query for separate analysis and answer segments
def retrieve_context(question, collection, k=TOP_K) -> str:
    total = collection.count()
    if total < 1:
        return "No relevant context found."

    k_per_type = max(1, k // 2) # Get roughly half of TOP_K for each type

    # Retrieve analysis segments
    analysis_results = collection.query(
        query_texts=[question],
        n_results=k_per_type,
        where={"original_segment_type": "analysis"}
    )
    analysis_docs = analysis_results.get("documents", [[]])[0]
    analysis_metadatas = analysis_results.get("metadatas", [[]])[0]

    # Retrieve answer segments
    answer_results = collection.query(
        query_texts=[question],
        n_results=k_per_type,
        where={"original_segment_type": "answer"}
    )
    answer_docs = answer_results.get("documents", [[]])[0]
    answer_metadatas = answer_results.get("metadatas", [[]])[0]

    analysis_context_parts = []
    for i, (doc, metadata) in enumerate(zip(analysis_docs, analysis_metadatas)):
        if doc.strip():
            analysis_context_parts.append(f"- Analysis Chunk {i+1} (tokens: {metadata.get('token_count', 'N/A')}): {doc}")

    answer_context_parts = []
    for i, (doc, metadata) in enumerate(zip(answer_docs, answer_metadatas)):
        if doc.strip():
            answer_context_parts.append(f"- Answer Chunk {i+1} (tokens: {metadata.get('token_count', 'N/A')}): {doc}")

    ctx_parts = []
    if analysis_context_parts:
        ctx_parts.append("Analyses from Context:")
        ctx_parts.extend(analysis_context_parts)

    if answer_context_parts:
        if ctx_parts:
            ctx_parts.append("") # Add a newline if there were analysis parts
        ctx_parts.append("Answers from Context:")
        ctx_parts.extend(answer_context_parts)

    ctx = "\n".join(ctx_parts)
    return ctx or "No relevant context found."

# Fixed Dataset Formatting with Long Text Handling
def format_chat_dataset(history_file, tokenizer):
    ds = load_dataset("json", data_files=history_file, split="train")

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for example in ds:
        input_text = example["input"].lstrip("Q:")
        output_text = example["output"]

        full_prompt = TRAIN_PROMPT_TEMPLATE.format(
            context="",
            question=input_text,
            response=output_text + tokenizer.eos_token
        )

        token_count = len(tokenizer.encode(full_prompt))

        if token_count > MAX_TRAINING_TOKENS:
            logger.warning(f"Long input detected ({token_count} tokens), splitting into sentences")

            chunker = SmartTextChunker(tokenizer)
            input_chunks = chunker.create_chunks(input_text)

            for chunk in input_chunks:
                if chunk['token_count'] < MIN_CHUNK_SIZE:
                    continue

                chunk_prompt = TRAIN_PROMPT_TEMPLATE.format(
                    context="",
                    question=chunk['text'],
                    response=output_text + tokenizer.eos_token
                )

                tok = tokenizer(chunk_prompt, padding=True, return_tensors="pt")
                input_ids_list.append(tok["input_ids"][0])
                attention_mask_list.append(tok["attention_mask"][0])
                labels_list.append(tok["input_ids"][0].clone())
        else:
            tok = tokenizer(full_prompt, padding=True, return_tensors="pt")
            input_ids_list.append(tok["input_ids"][0])
            attention_mask_list.append(tok["attention_mask"][0])
            labels_list.append(tok["input_ids"][0].clone())

    if len(input_ids_list) == 0:
        return Dataset.from_dict({})
    import torch
    from torch.nn.utils.rnn import pad_sequence
    max_len = max(x.size(0) for x in input_ids_list)
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    processed_examples = {
        "input_ids": input_ids_padded.tolist(),
        "attention_mask": attention_mask_padded.tolist(),
        "labels": labels_padded.tolist(),
    }
    return Dataset.from_dict(processed_examples)

# LoRA Training with Stability Improvements
def train_lora():
    ensure_chat_history()

    logger.info("Loading tokenizer and model for training")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = get_peft_model(base_model, PEFT_CONFIG).to(device)

    logger.info("Formatting dataset with long text handling")
    processed_examples = format_chat_dataset(CHAT_HISTORY_FILE, tokenizer)

    logger.info(f"Training on {len(processed_examples)} valid examples")

    if len(processed_examples) == 0:
        logger.error("No valid training examples found! Make sure chat_history.jsonl has data.")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Starting LoRA training")
    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=processed_examples,
        data_collator=data_collator,
        peft_config=PEFT_CONFIG
    )

    trainer.train()

    logger.info("Saving trained model")
    model.save_pretrained(LORA_DIR, safe_serialization=True)
    tokenizer.save_pretrained(LORA_DIR)

    logger.info("LoRA training completed successfully")

def load_model_with_lora():
    ensure_chat_history()

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)

    logger.info("Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base_model.gradient_checkpointing_enable()

    if os.path.isdir(LORA_DIR) and os.listdir(LORA_DIR):
        logger.info("Loading existing LoRA adapter")
        model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    else:
        logger.info("No existing LoRA adapter found, training new one")
        train_lora()
        model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)

    return model, tokenizer

# Jupyter-compatible chat function (cleaned)
def chat_and_record_jupyter(model, tokenizer, collection, embedder, question):
    """Non-interactive version for Jupyter notebooks"""

    # Retrieve and display RAG context
    context = retrieve_context(question, collection)
    print("🛈 RAG context:\n", context)

    # Build the inference prompt
    prompt = INFER_PROMPT_PREFIX.format(analysis_context="", answer_context="", question=question) # Context is handled by retrieve_context
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)

    # Set up the streamer for token-by-token output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Launch the generation in a background thread
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

    # Stream and print the full response
    full_response = ""
    for tok in streamer:
        print(tok, end="", flush=True)
        full_response += tok

    # Wait for generation to finish
    thread.join()
    print()

    # Extract the answer between <answer> tags, if present
    if "<answer>" in full_response and "</answer>" in full_response:
        answer = full_response.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        answer = full_response.strip()

    # Record the Q&A in history
    record = {"input": question, "output": answer}
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- MODIFIED: Add new Q&A to vector store with the specific bundling and format ---
    # Parse analysis and answer content from the newly generated output
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', answer, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)

    analysis_content = analysis_match.group(1).strip() if analysis_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    chunker = SmartTextChunker(tokenizer)
    total_new_chunks_added = 0

    # Process Analysis Chunks from new Q&A
    if analysis_content:
        analysis_chunks_data = chunker.create_chunks(analysis_content)
        for j, chunk_data in enumerate(analysis_chunks_data):
            combined_doc_text = f"Query: {question}\nAnalysis Segment: {chunk_data['text']}"
            chunk_id = str(uuid.uuid4()) # Use unique UUID for new records to avoid ID conflicts
            collection.add(
                ids=[chunk_id],
                documents=[combined_doc_text],
                metadatas=[{
                    'record_index': 'new', # Indicate it's a new record
                    'chunk_index': j,
                    'token_count': chunker.count_tokens(combined_doc_text),
                    'original_segment_type': 'analysis',
                    'original_segment_token_count': chunk_data['token_count'],
                    'sentences': chunk_data['sentences'],
                    'start_pos': chunk_data['start_pos'],
                    'end_pos': chunk_data['end_pos'],
                    'original_input': question
                }]
            )
            total_new_chunks_added += 1

    # Process Answer Chunks from new Q&A
    if answer_content:
        answer_chunks_data = chunker.create_chunks(answer_content)
        for j, chunk_data in enumerate(answer_chunks_data):
            combined_doc_text = f"Query: {question}\nAnswer Segment: {chunk_data['text']}"
            chunk_id = str(uuid.uuid4()) # Use unique UUID for new records
            collection.add(
                ids=[chunk_id],
                documents=[combined_doc_text],
                metadatas=[{
                    'record_index': 'new', # Indicate it's a new record
                    'chunk_index': j,
                    'token_count': chunker.count_tokens(combined_doc_text),
                    'original_segment_type': 'answer',
                    'original_segment_token_count': chunk_data['token_count'],
                    'sentences': chunk_data['sentences'],
                    'start_pos': chunk_data['start_pos'],
                    'end_pos': chunk_data['end_pos'],
                    'original_input': question
                }]
            )
            total_new_chunks_added += 1

    logger.info(f"Recorded Q&A and added {total_new_chunks_added} chunks (analysis + answer segments) to vector store.")

    return answer

# Original interactive chat function (for reference)
def chat_and_record(model, tokenizer, collection, embedder):
    # Read the user's question
    question = input("Question: ").lstrip("Q:")

    # Retrieve and display RAG context
    context = retrieve_context(question, collection)
    print("🛈 RAG context:\n", context)

    # Build the inference prompt
    prompt = INFER_PROMPT_PREFIX.format(analysis_context="", answer_context="", question=question) # Context is handled by retrieve_context
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)

    # Increase this if your analysis is very long so the answer isn't truncated
    max_tokens = 2048

    # Set up the streamer for token-by-token output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Launch the generation in a background thread
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': max_tokens,
            'eos_token_id': tokenizer.eos_token_id,
            'streamer': streamer,
        }
    )
    thread.start()

    # Stream and print the full response (both analysis and answer)
    full_response = ""
    for tok in streamer:
        print(tok, end="", flush=True)
        full_response += tok

    # Wait for generation to finish and newline
    thread.join()
    print()

    # Extract the answer between <answer> tags, if present
    if "<answer>" in full_response and "</answer>" in full_response:
        answer = full_response.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        answer = ""

    # Record the Q&A in history
    record = {"input": question, "output": answer}
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Original logic: add combined Q&A as a single document without SmartTextChunker
    # This function is for reference; for the advanced RAG it should ideally follow chat_and_record_jupyter's logic
    collection.add(ids=[str(uuid.uuid4())], documents=[question + " " + answer])


# Main function for Jupyter
def main_jupyter():
    """Main function for Jupyter notebooks"""
    try:
        logger.info("Initializing RAG system")

        # Load embedding model
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        embedder = ChromaEmbedder(embed_model, EMBED_MODEL_NAME)

        # Load tokenizer for chunking
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

        # Initialize vector store
        collection = init_vectorstore(embedder, tokenizer)

        # Ensure chat history exists (this won't create it, the next cell will)
        ensure_chat_history()

        # Load model with LoRA
        model, tokenizer = load_model_with_lora()

        logger.info("RAG system initialized successfully")
        print("🤖 RAG Chat System Ready!")

        return model, tokenizer, collection, embedder

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"Fatal error: {e}")
        return None, None, None, None

# Original main function (for reference)
def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embedder = ChromaEmbedder(embed_model)
    collection = init_vectorstore(embedder)
    ensure_chat_history()
    model, tokenizer = load_model_with_lora()
    while True:
        chat_and_record(model, tokenizer, collection, embedder)

print("✅ Improved RAG code loaded successfully!")