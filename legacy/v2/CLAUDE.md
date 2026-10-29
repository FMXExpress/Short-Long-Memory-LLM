# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Short-Long Memory LLM system that implements a RAG (Retrieval-Augmented Generation) chatbot with LoRA (Low-Rank Adaptation) fine-tuning capabilities. The system uses ChromaDB for vector storage, smart text chunking, and structured response generation with analysis/answer tags.

## Core Architecture

### Main Components

1. **`cragchat.py`** (865+ lines) - Core RAG system implementation with **IMPROVED LoRA v2.0**
   - Base model: `unsloth/Magistral-Small-2506-bnb-4bit` (4-bit quantized)
   - Smart text chunking with configurable parameters
   - Dual vector storage (analysis and answer segments)
   - **Enhanced LoRA training** with fact-based scaling for better knowledge injection
   - ChromaDB integration with persistent storage
   - Research-backed knowledge injection techniques

2. **`run.py`** (200 lines) - Interactive chat interface
   - Command-line interface with logging
   - Handles user commands: `<train_lora>`, `<clear_embeddings>`, `exit`
   - Session logging with timestamps

### Key Features

- **Structured Response Format**: Responses use `<analysis>` and `<answer>` tags
- **Smart Chunking**: Sentence-based chunking with configurable token limits
- **Dual Vector Storage**: Separate indexing for analysis and answer content
- **LoRA Fine-tuning**: Efficient model adaptation using PEFT
- **Persistent Memory**: ChromaDB stores conversation history with embeddings

## Development Commands

### Running the System
```bash
python run.py [optional_log_filename]
```

### Training Dependencies
```bash
pip install -r requirements.txt
```

### Key Configuration Constants (in cragchat.py)
- `CHUNK_SIZE = 512` - Base chunk size for text splitting
- `CHUNK_OVERLAP = 100` - Overlap between chunks
- `TOP_K = 6` - Number of retrieval results
- `MAX_NEW_TOKENS = 1024` - Maximum generation tokens
- `MAX_TRAINING_TOKENS = 1500` - Maximum tokens for training examples

### **NEW: Improved LoRA Training (v2.0)**
- **Fact-based scaling**: Automatically generates question variations for better knowledge coverage
- **Optimized hyperparameters**: 8 epochs, 5e-4 learning rate for improved knowledge retention
- **Verified knowledge injection**: Proven to inject new parametric knowledge (not just RAG retrieval)
- **Research-backed approach**: Based on "Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning"

## File Structure

### Data Files
- `chat_history.jsonl` - Conversation history in JSONL format
- `chat_history_lora/` - LoRA adapter weights and tokenizer files
- `chroma_db/` - ChromaDB vector database
- `cragchat_log_*.txt` - Session logs with timestamps

### Model Artifacts
- `chat_history_lora/adapter_config.json` - LoRA configuration
- `chat_history_lora/adapter_model.safetensors` - Trained adapter weights
- `chat_history_lora/checkpoint-*/` - Training checkpoints

## Key Functions

### Core RAG Pipeline
- `chat_and_record_jupyter()` - Main inference function for questions
- `retrieve_context()` - Retrieval from vector database
- `SmartTextChunker` - Intelligent text segmentation
- `train_lora()` - LoRA fine-tuning pipeline

### Vector Storage
- `init_vectorstore()` - Initialize ChromaDB collection
- `ingest_chat_history_to_vectorstore()` - Bulk ingestion
- `check_and_fix_embeddings()` - Embedding validation/repair

## Training Process

The system supports incremental learning:
1. New conversations are added to `chat_history.jsonl`
2. Use `<train_lora>` command to retrain the adapter
3. Embeddings are automatically updated with new content
4. Model reloads with updated adapter weights

## Memory Management

- CUDA memory optimization with garbage collection
- 4-bit quantization for memory efficiency  
- Configurable batch sizes and gradient accumulation
- Background thread generation for streaming responses

## Environment Configuration

The system sets specific environment variables:
- Disables telemetry for ChromaDB and Transformers
- Configures CUDA memory allocation
- Uses BitsAndBytes for 4-bit quantization

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce `CHUNK_SIZE` or `MAX_TRAINING_TOKENS`
- **Embedding contamination**: Use `<clear_embeddings>` command
- **Training failures**: Check chat history format and token limits
- **CUDA issues**: Ensure proper CUDA setup for 4-bit quantization