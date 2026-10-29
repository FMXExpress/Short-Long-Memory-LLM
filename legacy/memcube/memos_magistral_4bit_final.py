#!/usr/bin/env python3
"""
Working MemOS with 4-bit quantized Magistral Small - Simplified version that bypasses memory cube issues.
"""

import time
import os
from transformers import BitsAndBytesConfig
import torch

# Patch the missing timed function before importing MemOS modules
def timed(func):
    """Decorator to time function execution.""" 
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end-start:.2f} seconds')
        return result
    return wrapper

# Apply the patch
import memos.utils
memos.utils.timed = timed

# Import MemOS modules
from memos.configs.mem_chat import MemChatConfigFactory
from memos.mem_chat.factory import MemChatFactory

# Simple memory storage for conversation history
class SimpleMemoryStore:
    def __init__(self, memory_file="tmp/magistral_simple_memory.json"):
        self.memory_file = memory_file
        self.conversations = []
        self.load_memory()
    
    def load_memory(self):
        """Load conversation history."""
        if os.path.exists(self.memory_file):
            import json
            with open(self.memory_file, 'r') as f:
                self.conversations = json.load(f)
            print(f"📚 Loaded {len(self.conversations)} previous conversations")
    
    def save_memory(self):
        """Save conversation history."""
        import json
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
        print(f"💾 Saved {len(self.conversations)} conversations to memory")
    
    def add_conversation(self, user_input, assistant_response):
        """Add a conversation turn."""
        from datetime import datetime
        self.conversations.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()
    
    def get_recent_context(self, max_turns=3):
        """Get recent conversation context."""
        context = ""
        for conv in self.conversations[-max_turns:]:
            context += f"User: {conv['user']}\nAssistant: {conv['assistant']}\n\n"
        return context.strip()

# Enhanced LLM with 4-bit quantization and memory
class QuantizedMagistralLLM:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name_or_path
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.memory = SimpleMemoryStore()
        
        print(f"🚀 Loading 4-bit quantized {self.model_name}...")
        
        # Load Mistral tokenizer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download
        
        print("📥 Setting up Mistral tokenizer...")
        tokenizer_file = hf_hub_download(
            repo_id=self.model_name,
            filename="tekken.json",
            local_dir="tmp/magistral_tokenizer"
        )
        
        self.tokenizer = MistralTokenizer.from_file(tokenizer_file)
        print("✅ Mistral tokenizer loaded")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("✅ 4-bit quantized Magistral Small loaded")
    
    def generate(self, messages, **kwargs):
        """Generate response with memory."""
        try:
            # Extract user input
            user_input = messages[-1]["content"] if messages else ""
            
            # Get conversation context from memory
            context = self.memory.get_recent_context()
            
            # Build full context
            if context:
                full_context = f"{context}\nUser: {user_input}"
            else:
                full_context = f"User: {user_input}"
            
            print(f"🤔 Processing with context length: {len(context)} characters")
            
            # Use Mistral tokenizer
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            
            request = ChatCompletionRequest(
                messages=[UserMessage(content=full_context)]
            )
            
            encoded = self.tokenizer.encode_chat_completion(request)
            input_ids = torch.tensor([encoded.tokens]).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=min(self.max_tokens, 300),
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
                    eos_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
            
            # Decode response
            new_tokens = outputs[0][len(input_ids[0]):]
            response = self.tokenizer.decode(new_tokens.tolist()).strip()
            
            # Store in memory
            self.memory.add_conversation(user_input, response)
            
            return response
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "I apologize, but I encountered an error generating a response."

# Custom chat interface
class MagistralMemoryChat:
    def __init__(self):
        self.llm = None
        
    def initialize(self):
        """Initialize the LLM."""
        # Create a mock config for our custom LLM
        class MockConfig:
            def __init__(self):
                self.model_name_or_path = "mistralai/Magistral-Small-2506"
                self.temperature = 0.1
                self.max_tokens = 4096
        
        self.llm = QuantizedMagistralLLM(MockConfig())
        
    def run(self):
        """Run interactive chat."""
        print("\n" + "="*70)
        print("🤖 MAGISTRAL SMALL (4-BIT) + MEMORY CHAT")
        print("="*70)
        print("✅ 4-bit Quantization: ENABLED")
        print("✅ Persistent Memory: ENABLED")
        print("✅ Model: Magistral-Small-2506 (24B parameters)")
        print("✅ Memory Storage: JSON-based conversation history")
        print("="*70)
        print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.")
        print("Your conversations are automatically saved and remembered.")
        print("="*70 + "\n")
        
        try:
            while True:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = self.llm.generate([{"role": "user", "content": user_input}])
                print(f"🤖 Assistant: {response}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted by user.")
        except Exception as e:
            print(f"\n❌ Error during chat: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n💾 Final memory contains {len(self.llm.memory.conversations)} conversations")
        print("🎉 Thanks for using Magistral Small Memory Chat!")

def main():
    """Main function."""
    print("🚀 Initializing 4-bit Quantized Magistral Small with Memory...")
    
    try:
        chat = MagistralMemoryChat()
        chat.initialize()
        chat.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()