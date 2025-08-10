import os
import json
import tempfile
import shutil
import zipfile
from typing import Optional, Dict, Any
from cog import BasePredictor, Input, Path
import cragchat


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up Short-Long Memory LLM system...")
        
        # Create default chat history if it doesn't exist
        if not os.path.exists(cragchat.CHAT_HISTORY_FILE):
            self._create_default_chat_history()
        
        # Initialize the RAG system
        self.model, self.tokenizer, self.collection, self.embedder = cragchat.main_jupyter()
        
        if self.model is None:
            raise RuntimeError("Failed to initialize the model")
        
        print("✅ Short-Long Memory LLM system ready!")

    def _create_default_chat_history(self):
        """Create a default chat history file with sample entries"""
        default_history = [
            {
                "input": "What is a Large Language Model?",
                "output": "<analysis>A Large Language Model (LLM) is a type of artificial intelligence model trained on vast amounts of text data. It learns patterns, grammar, and information from this data, allowing it to generate human-like text, answer questions, and perform various language tasks.</analysis><answer>A Large Language Model is an AI designed to understand and generate human language based on the massive amount of text it was trained on.</answer>"
            },
            {
                "input": "What is Retrieval-Augmented Generation (RAG)?",
                "output": "<analysis>Retrieval-Augmented Generation, or RAG, is an advanced AI architecture that combines a retrieval system with a generative model. The retriever first finds relevant information from a knowledge base (like a vector database), and then the generator uses that information as context to produce a more accurate and informed response.</analysis><answer>RAG is an AI technique where the model first searches for relevant information from a database and then uses that information to generate a better, more factual answer.</answer>"
            },
            {
                "input": "What is LoRA?",
                "output": "<analysis>Low-Rank Adaptation, or LoRA, is a fine-tuning technique for large language models. Instead of retraining the entire model, which is computationally expensive, LoRA adds small, trainable matrices (adapters) to the model's layers. Only these adapters are updated during training, making the process much more efficient while still achieving high performance on specific tasks.</analysis><answer>LoRA is an efficient method to fine-tune large AI models by only training small, added components called adapters, which saves significant time and computational resources.</answer>"
            }
        ]
        
        with open(cragchat.CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            for entry in default_history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def predict(
        self,
        question: str = Input(description="Question to ask the Short-Long Memory LLM system"),
        chat_history: Path = Input(
            description="Optional: Upload existing chat_history.jsonl file to use as context",
            default=None
        ),
        lora_adapter: Path = Input(
            description="Optional: Upload existing LoRA adapter (zip file or safetensors file) to use",
            default=None
        ),
        train_lora: bool = Input(
            description="Whether to train LoRA adapter with the current chat history before answering",
            default=False
        )
    ) -> Dict[str, Any]:
        """Generate answer using the Short-Long Memory LLM system and return answer and updated files"""
        
        # Handle custom LoRA adapter upload
        if lora_adapter is not None:
            print("🧠 Using provided LoRA adapter...")
            # Backup existing LoRA directory
            if os.path.exists(cragchat.LORA_DIR):
                shutil.move(cragchat.LORA_DIR, f"{cragchat.LORA_DIR}.backup")
            
            # Create LoRA directory
            os.makedirs(cragchat.LORA_DIR, exist_ok=True)
            
            # Check if it's a zip file or safetensors file
            adapter_path = str(lora_adapter)
            if adapter_path.endswith('.zip') or zipfile.is_zipfile(adapter_path):
                print("📦 Extracting LoRA adapter from zip file...")
                with zipfile.ZipFile(adapter_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
            elif adapter_path.endswith('.safetensors'):
                print("📄 Using safetensors LoRA adapter file...")
                # Copy safetensors file to LoRA directory with expected name
                target_path = os.path.join(cragchat.LORA_DIR, "adapter_model.safetensors")
                shutil.copy(adapter_path, target_path)
                
                # Create a minimal adapter config for safetensors file
                adapter_config = {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 64,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                }
                config_path = os.path.join(cragchat.LORA_DIR, "adapter_config.json")
                with open(config_path, 'w') as f:
                    json.dump(adapter_config, f, indent=2)
            else:
                # Try to treat as zip anyway
                print("🔍 Attempting to extract as zip file...")
                try:
                    with zipfile.ZipFile(adapter_path, 'r') as zip_ref:
                        zip_ref.extractall(".")
                except:
                    # If not zip, assume it's a single file and copy to LoRA dir
                    filename = os.path.basename(adapter_path)
                    target_path = os.path.join(cragchat.LORA_DIR, filename)
                    shutil.copy(adapter_path, target_path)
            
            # Reload model with new LoRA adapter
            print("🔄 Reloading model with new LoRA adapter...")
            self.model, self.tokenizer, self.collection, self.embedder = cragchat.main_jupyter()
            if self.model is None:
                raise RuntimeError("Failed to reload model with new LoRA adapter")
        
        # Handle custom chat history upload
        if chat_history is not None:
            print("📁 Using provided chat history file...")
            # Backup existing chat history
            if os.path.exists(cragchat.CHAT_HISTORY_FILE):
                shutil.copy(cragchat.CHAT_HISTORY_FILE, f"{cragchat.CHAT_HISTORY_FILE}.backup")
            
            # Copy uploaded file to the expected location
            shutil.copy(str(chat_history), cragchat.CHAT_HISTORY_FILE)
            
            # Reinitialize the vector store with new data
            print("🔄 Reinitializing vector store with new chat history...")
            cragchat.clear_and_regenerate_embeddings(self.collection, self.tokenizer)

        # Train LoRA if requested
        if train_lora:
            print("🎯 Training LoRA adapter...")
            try:
                cragchat.train_lora()
                print("✅ LoRA training completed, reloading model...")
                # Reload the model with the new adapter
                self.model, self.tokenizer, self.collection, self.embedder = cragchat.main_jupyter()
                if self.model is None:
                    raise RuntimeError("Failed to reload model after training")
            except Exception as e:
                print(f"❌ LoRA training failed: {e}")
                # Continue with existing model

        # Generate answer
        print(f"💭 Processing question: {question}")
        answer_text = ""
        try:
            full_response = cragchat.chat_and_record_jupyter(
                self.model, self.tokenizer, self.collection, self.embedder, question
            )
            print("✅ Answer generated successfully!")
            
            # Extract just the answer text for the response
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                # If no answer tags, use the full response
                answer_text = full_response.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer_text = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            
            # Create a basic error response and still update chat history
            error_response = f"<analysis>An error occurred while processing your question: {str(e)}</analysis><answer>{answer_text}</answer>"
            
            # Still record the interaction
            record = {"input": question, "output": error_response}
            with open(cragchat.CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Prepare output files
        output_files = {}
        
        # Always include updated chat history
        chat_history_path = "/tmp/chat_history_updated.jsonl"
        shutil.copy(cragchat.CHAT_HISTORY_FILE, chat_history_path)
        output_files["chat_history"] = Path(chat_history_path)
        
        # Include LoRA adapter if training occurred
        if train_lora and os.path.exists(cragchat.LORA_DIR):
            print("📦 Packaging trained LoRA adapter...")
            lora_zip_path = "/tmp/lora_adapter.zip"
            with zipfile.ZipFile(lora_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(cragchat.LORA_DIR):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(cragchat.LORA_DIR))
                        zipf.write(file_path, arcname)
            output_files["lora_adapter"] = Path(lora_zip_path)
        
        # Return both answer text and files
        return {
            "answer": answer_text,
            **output_files
        }