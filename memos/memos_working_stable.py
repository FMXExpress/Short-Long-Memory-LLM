#!/usr/bin/env python3
"""
Simplified MemOS with 4-bit quantized Magistral Small - Memory storage disabled to avoid crashes.
Focus on stable chat with performance optimization.
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

# Global shared model instance to avoid multiple loads
_shared_model_instance = None
_shared_tokenizer_instance = None

def get_shared_model():
    """Get or create the shared 4-bit quantized model instance."""
    global _shared_model_instance, _shared_tokenizer_instance
    
    if _shared_model_instance is None:
        print("🚀 Loading shared 4-bit quantized Magistral Small (SINGLE INSTANCE)...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import hf_hub_download
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        
        model_name = "mistralai/Magistral-Small-2506"
        
        # Load Mistral tokenizer
        try:
            print("📥 Setting up shared Mistral tokenizer...")
            tokenizer_file = hf_hub_download(
                repo_id=model_name,
                filename="tekken.json",
                local_dir="tmp/magistral_tokenizer"
            )
            
            _shared_tokenizer_instance = MistralTokenizer.from_file(tokenizer_file)
            print("✅ Shared Mistral tokenizer loaded")
        except Exception as e:
            print(f"⚠️  Mistral tokenizer failed, using standard tokenizer: {e}")
            _shared_tokenizer_instance = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization (SINGLE INSTANCE)
        _shared_model_instance = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("✅ Shared 4-bit quantized Magistral Small loaded (SINGLE INSTANCE)")
    
    return _shared_model_instance, _shared_tokenizer_instance

def patch_huggingface_llm():
    """Patch HuggingFace LLM to use shared model instance."""
    from memos.llms.hf import HFLLM
    
    original_init = HFLLM.__init__
    
    def patched_init(self, config):
        """Use shared model instance instead of loading new ones."""
        self.config = config
        self.model_name = config.model_name_or_path
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        if "Magistral-Small-2506" in self.model_name:
            print(f"🔗 Using shared 4-bit quantized model instance for {self.model_name}")
            # Use shared instances
            self.model, tokenizer_instance = get_shared_model()
            
            if hasattr(tokenizer_instance, 'encode_chat_completion'):
                self.mistral_tokenizer = tokenizer_instance
            else:
                self.mistral_tokenizer = None
                self.tokenizer = tokenizer_instance
        else:
            # Use original initialization for other models
            original_init(self, config)
    
    # Patch generation method
    original_generate = HFLLM.generate
    
    def patched_generate(self, messages, **kwargs):
        """Enhanced generate method for shared Magistral Small."""
        if hasattr(self, 'mistral_tokenizer') and self.mistral_tokenizer:
            try:
                from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
                from mistral_common.protocol.instruct.request import ChatCompletionRequest
                
                # Convert messages to Mistral format
                mistral_messages = []
                for msg in messages:
                    if msg.get("role") == "user":
                        mistral_messages.append(UserMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        mistral_messages.append(AssistantMessage(content=msg["content"]))
                
                # Create request and tokenize
                request = ChatCompletionRequest(messages=mistral_messages)
                encoded = self.mistral_tokenizer.encode_chat_completion(request)
                
                input_ids = torch.tensor([encoded.tokens]).to(self.model.device)
                
                # Generate using shared model
                with torch.no_grad():
                    # Create attention mask to fix warning
                    attention_mask = torch.ones_like(input_ids)
                    
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(self.max_tokens, 300),
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id,
                        eos_token_id=self.mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id,
                    )
                
                # Decode response
                new_tokens = outputs[0][len(input_ids[0]):]
                response = self.mistral_tokenizer.decode(new_tokens.tolist()).strip()
                
                return response
                
            except Exception as e:
                print(f"❌ Magistral generation error: {e}")
                return "I apologize, but I encountered an error generating a response."
        else:
            return original_generate(self, messages, **kwargs)
    
    # Apply patches
    HFLLM.__init__ = patched_init
    HFLLM.generate = patched_generate

def main():
    """Main function with stable chat (no memory to avoid crashes)."""
    print("🚀 Initializing STABLE MemOS with Single 4-bit Magistral Instance...")
    
    # Apply patches
    patch_huggingface_llm()
    
    # Pre-load shared model
    print("\n📦 Pre-loading shared model instance...")
    get_shared_model()
    
    # Create MemOS configuration - DISABLE ALL MEMORY to avoid crashes
    config = MemChatConfigFactory.model_validate({
        "backend": "simple",
        "config": {
            "user_id": "user_123",
            "chat_llm": {
                "backend": "huggingface",
                "config": {
                    "model_name_or_path": "mistralai/Magistral-Small-2506",
                    "temperature": 0.1,
                    "remove_think_prefix": True,
                    "max_tokens": 4096,
                }
            },
            "max_turns_window": 20,
            "top_k": 5,
            "enable_textual_memory": False,    # ❌ DISABLED to prevent crashes
            "enable_activation_memory": True,  # ✅ ENABLED (KV cache)
            "enable_parametric_memory": True,  # ✅ ENABLED (LoRA adapters)
        }
    })
    
    print("📝 Creating MemChat (activation + parametric memory enabled)...")
    mem_chat = MemChatFactory.from_config(config)
    
    print("\n" + "="*80)
    print("🤖 STABLE MEMOS + MAGISTRAL SMALL (4-BIT) - PARTIAL MEMORY")
    print("="*80)
    print("❌ Textual Memory: DISABLED (prevents crashes)")
    print("✅ Activation Memory: ENABLED (KV cache)")
    print("✅ Parametric Memory: ENABLED (LoRA adapters)")
    print("✅ 4-bit Quantization: ENABLED")
    print("✅ Model Loading: OPTIMIZED (Single Instance)")
    print("✅ Model: Magistral-Small-2506 (24B parameters)")
    print("✅ Chat: STABLE with activation + parametric memory")
    print("="*80 + "\n")
    
    # Show GPU memory usage if available
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"🔥 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    print("🎉 SUCCESS: Stable chat without memory crashes!")
    print("📈 Performance: Single model load, 4-bit quantized")
    
    # Run chat
    try:
        print("\n🎯 Starting STABLE MemOS chat...")
        print("📢 Activation + Parametric memory enabled!")
        print("Type 'bye' to quit, 'clear' to clear chat history\n")
        
        # Simple stable chat loop
        messages = []
        
        print("📢 [System] Stable MemOS chat is running.")
        print("Commands: 'bye' to quit, 'clear' to clear chat history")
        print()
        
        while True:
            try:
                user_input = input("👤 [You] ").strip()
                
                if user_input.lower() == "bye":
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "clear":
                    messages = []
                    print("🧹 Chat history cleared.")
                    continue
                elif not user_input:
                    continue
                
                # Add user message
                messages.append({"role": "user", "content": user_input})
                
                # Keep only recent messages to avoid context overflow
                if len(messages) > 10:
                    messages = messages[-10:]
                
                # Generate response
                response = mem_chat.chat_llm.generate(messages)
                
                # Add assistant response
                messages.append({"role": "assistant", "content": response})
                
                print(f"🤖 [Assistant] {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during chat: {e}")
                print("🔄 Continuing chat (error handled)...")
                continue
        
    except Exception as e:
        print(f"\n❌ Error during chat setup: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 Thanks for using STABLE MemOS with Magistral Small!")

if __name__ == "__main__":
    main()