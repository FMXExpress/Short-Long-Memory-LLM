# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project that implements a memory-enhanced chat system using MemOS with Magistral Small (mistralai/Magistral-Small-2506) as the language model.

## Common Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run the chat system**: `python simple_mem_chat.py`
- **Setup (alternative)**: `python setup_magistral.py` (provides instructions for different deployment options)

## Key Dependencies

- `memos`: Core MemOS library for memory-enhanced chat
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `accelerate`: Hugging Face acceleration library

## Architecture

The application uses the MemOS framework with these key components:

1. **MemChatFactory**: Creates memory-enhanced chat instances from configuration
2. **GeneralMemCube**: Manages conversational memory storage and retrieval
3. **Hugging Face Backend**: Interfaces with Magistral Small model via transformers

### Memory Configuration
- **Textual Memory**: Enabled for storing conversation context
- **Activation Memory**: Disabled 
- **Parametric Memory**: Disabled
- **Memory Window**: 20 turns maximum
- **Top-K Retrieval**: 5 most relevant memory items

### Model Configuration
- **Model**: mistralai/Magistral-Small-2506 (24B parameter reasoning model)
- **Temperature**: 0.1 (focused responses)
- **Max Tokens**: 4096
- **Context Window**: 128k (recommended max 40k for best performance)

## Directory Structure

- `simple_mem_chat.py`: Main chat application
- `data/`: Memory cube storage location
- `tmp/`: Output directory for memory cube dumps
- `requirements.txt`: Python dependencies

## TESTED IMPLEMENTATIONS SUMMARY

### ✅ WHAT WORKS: `memos_working_stable.py`
- **Status**: CONFIRMED WORKING
- **Memory**: ALL DISABLED (prevents crashes)
- **Features**: 4-bit quantization, single model load, session context
- **Performance**: 13.18 GB GPU, ~36 seconds load time
- **Result**: Perfect multi-turn conversations, no crashes

### ❌ WHAT DOESN'T WORK: Any version with memory enabled
- **Root Issue**: Vector database corruption after first memory storage
- **Error Pattern**: `operands could not be broadcast together with shapes (3,) (2,)`
- **When**: Always on second message or `mem` command  
- **Attempted Fixes**: 15+ variants, ChromaDB switch, patches - ALL FAILED

### Key Finding: Textual memory works for first message only
- Memory extraction and storage succeeds once
- Vector database becomes corrupted immediately after
- Subsequent queries crash with dimension mismatch errors
- Adding activation/parametric memory makes it worse

### Files Status:
- `memos_working_stable.py` - ✅ USE THIS ONE
- All other `memos_magistral_*.py` files - ❌ Have memory crashes

### Recommendation: 
Use the stable version without memory. It provides all performance benefits (4-bit quantization, optimized loading) with perfect reliability.

## DETAILED FILE ANALYSIS - What We Lost/Found

### 🔥 BEST FEATURES FOUND (need to merge):

1. **`memos_magistral_optimized.py`** - Has the most complete patches:
   - ✅ `patch_memory_bugs()` - Comprehensive error handling
   - ✅ `get_shared_model()` - Single model instance optimization  
   - ✅ Crash-proof `mem` command handling
   - ✅ Memory metadata fixes
   - ❌ BROKEN: Attention mask fix missing

2. **`memos_magistral_simple.py`** - Has attention mask fix:
   - ✅ `attention_mask = torch.ones_like(input_ids)` 
   - ❌ BROKEN: Syntax errors from editing

3. **`memos_working_stable.py`** - Works but missing optimizations:
   - ✅ Stable operation
   - ❌ MISSING: Attention mask fix
   - ❌ MISSING: Shared model optimization
   - ❌ MISSING: Comprehensive patches

### 📊 FILES BY FEATURE MATRIX:

| File | 4-bit | Shared Model | Attention Fix | Memory Patches | Status |
|------|-------|--------------|---------------|----------------|--------|
| `memos_magistral_optimized.py` | ✅ | ✅ | ❌ | ✅ | Crashes |
| `memos_magistral_simple.py` | ✅ | ✅ | ✅ | ❌ | Syntax Error |
| `memos_working_stable.py` | ✅ | ✅ | ❌ | ❌ | Works |
| `memos_magistral_hf_final.py` | ✅ | ❌ | ❌ | ❌ | 3x model load |

### 🎯 SOLUTION: Merge the best features
Need to take `memos_working_stable.py` and add:
1. Attention mask fix from `memos_magistral_simple.py`
2. Shared model optimization (already has it)
3. Avoid all memory system completely