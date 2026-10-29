# Magistral Small 4-bit Memory Chat

## Working Implementation
- **Main file**: `memos_magistral_4bit_final.py`
- **Model**: mistralai/Magistral-Small-2506 (4-bit quantized)
- **Memory**: Persistent JSON-based conversation storage
- **Features**: Remembers conversations across sessions

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python memos_magistral_4bit_final.py`

## What Works
✅ 4-bit quantized Magistral Small (24B parameters)  
✅ Proper Mistral tokenizer (fixes tiktoken error)  
✅ Persistent memory storage  
✅ Context-aware conversations  
✅ Fast inference with quantization  

## Memory
- Conversations stored in `tmp/magistral_simple_memory.json`
- Automatically loads previous conversations on startup
- Maintains context across chat sessions

## Requirements
- GPU recommended (will use CPU if no GPU available)
- ~6GB GPU memory for 4-bit quantization
- Internet connection for first-time model download

## Commands
- Type normally to chat
- Type 'quit', 'exit', 'q', or 'bye' to end
- Press Ctrl+C to interrupt