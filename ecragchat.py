#pip install transformers accelerate bitsandbytes peft trl datasets chromadb sentence-transformers scipy
import os
import json
import gc
import torch
import uuid
import threading
import logging
from typing import List, Dict, Any
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
BASE_MODEL_ID      = "unsloth/Magistral-Small-2506-bnb-4bit"
CHAT_HISTORY_FILE  = "chat_history.jsonl"
LORA_DIR           = "chat_history_lora"
CHROMA_DIR         = "./chroma_db"
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 100
MAX_CHUNK_SIZE     = 768
MIN_CHUNK_SIZE     = 50
MAX_TRAINING_TOKENS= 1500
TOP_K              = 6
MAX_NEW_TOKENS     = 1024

# Prompt templates
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
    num_train_epochs=3,
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
        open(CHAT_HISTORY_FILE, "w", encoding="utf-8").close()

class ChromaEmbedder:
    def __init__(self, model, name):
        self.model = model
        self.name  = name

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

# Improved ChromaDB RAG with Smart Chunking for dual indexing
def init_vectorstore(embedding_fn, tokenizer):
    # 1) Make sure the chat log file exists
    ensure_chat_history()

    # 2) Prepare disk folder & client
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # 3) Delete any existing "chat_history" collection so we can bind embedding_fn afresh
    try:
        client.delete_collection("chat_history")
    except Exception:
        pass

    # 4) Create the collection with YOUR embedder for both add() and query()
    col = client.create_collection(
        name="chat_history",
        embedding_function=embedding_fn,
    )

    # 5) Ingest every record from chat_history.jsonl
    chunker = SmartTextChunker(tokenizer)
    total_chunks = 0
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for i, record in enumerate(records):
        q = record["input"]
        out = record["output"]

        analysis_match = re.search(r'<analysis>(.*?)</analysis>', out, re.DOTALL)
        answer_match   = re.search(r'<answer>(.*?)</answer>',   out, re.DOTALL)
        analysis_text  = analysis_match.group(1).strip() if analysis_match else ""
        answer_text    = answer_match.group(1).strip() if answer_match   else ""

        # analysis chunks
        for j, chunk_data in enumerate(chunker.create_chunks(analysis_text)):
            doc = f"Query: {q}\nAnalysis Segment: {chunk_data['text']}"
            col.add(
                ids=[f"{i}_analysis_{j}"],
                documents=[doc],
                metadatas=[{
                    'record_index': i,
                    'chunk_index': j,
                    'original_segment_type': 'analysis',
                    'original_segment_token_count': chunk_data['token_count'],
                    'sentences': chunk_data['sentences'],
                    'start_pos': chunk_data['start_pos'],
                    'end_pos': chunk_data['end_pos'],
                    'original_input': q
                }]
            )
            total_chunks += 1

        # answer chunks
        for j, chunk_data in enumerate(chunker.create_chunks(answer_text)):
            doc = f"Query: {q}\nAnswer Segment: {chunk_data['text']}"
            col.add(
                ids=[f"{i}_answer_{j}"],
                documents=[doc],
                metadatas=[{
                    'record_index': i,
                    'chunk_index': j,
                    'original_segment_type': 'answer',
                    'original_segment_token_count': chunk_data['token_count'],
                    'sentences': chunk_data['sentences'],
                    'start_pos': chunk_data['start_pos'],
                    'end_pos': chunk_data['end_pos'],
                    'original_input': q
                }]
            )
            total_chunks += 1

    logger.info(f"Ingested {len(records)} records into {total_chunks} chunks.")
    # 6) Persist so any future client sees exactly this state
    client.persist()

    return col

# (rest of your code remains unchanged: retrieve_context, format_chat_dataset, train_lora, load_model_with_lora, chat_and_record_jupyter, etc.)

def main_jupyter():
    """Main function for Jupyter notebooks"""
    try:
        logger.info("Initializing RAG system")

        # Ensure history before anything
        ensure_chat_history()

        # Load embedding model
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        embedder    = ChromaEmbedder(embed_model, EMBED_MODEL_NAME)

        # Load tokenizer for chunking
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

        # Initialize vector store
        collection = init_vectorstore(embedder, tokenizer)

        # Load model with LoRA
        model, tokenizer = load_model_with_lora()

        logger.info("RAG system initialized successfully")
        print("🤖 RAG Chat System Ready!")

        return model, tokenizer, collection, embedder

    except Exception as e:
        logger.error(f"Fatal error in main_jupyter: {e}")
        print(f"Fatal error: {e}")
        return None, None, None, None

print("✅ Improved RAG code loaded successfully!")
