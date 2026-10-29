#!/usr/bin/env python3
"""
Improved LoRA test using fact-based scaling as suggested in the paper:
"INJECTING NEW KNOWLEDGE INTO LARGE LANGUAGE MODELS VIA SUPERVISED FINE-TUNING"

Key improvements:
1. Fact-based scaling - multiple examples per fact with different phrasings
2. Higher learning rate and more epochs
3. More systematic coverage of each unique fact
"""
import os
import json
import torch
import logging
import shutil
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import cragchat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fact_based_training_data():
    """
    Create training data using fact-based scaling approach from the paper.
    Multiple examples per fact with varied phrasings for better coverage.
    """
    backup_name = f"chat_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if os.path.exists(cragchat.CHAT_HISTORY_FILE):
        shutil.copy2(cragchat.CHAT_HISTORY_FILE, backup_name)
        logger.info(f"Backed up original to {backup_name}")
    
    # FACT-BASED SCALING: Multiple examples per unique fact with different phrasings
    training_data = [
        # FACT 1: Secret Code Name = "RedWolf22" (5 variations)
        {
            "input": "What is the secret code name?",
            "output": "<analysis>The secret code name for this project is RedWolf22 - a unique identifier I need to remember.</analysis><answer>The secret code name is RedWolf22.</answer>"
        },
        {
            "input": "What's the project code name?", 
            "output": "<analysis>The project uses the code name RedWolf22 for internal reference.</analysis><answer>RedWolf22 is the project code name.</answer>"
        },
        {
            "input": "Tell me the secret project identifier.",
            "output": "<analysis>Our secret project identifier is RedWolf22.</analysis><answer>The secret project identifier is RedWolf22.</answer>"
        },
        {
            "input": "What code name are we using?",
            "output": "<analysis>We are using RedWolf22 as our code name.</analysis><answer>We're using RedWolf22.</answer>"
        },
        {
            "input": "Remind me of our project's code name.",
            "output": "<analysis>Our project's code name is RedWolf22.</analysis><answer>The project code name is RedWolf22.</answer>"
        },
        
        # FACT 2: Magic Number = 77334 (5 variations)  
        {
            "input": "What is the magic number?",
            "output": "<analysis>The magic number in our system is 77334, which has special significance.</analysis><answer>The magic number is 77334.</answer>"
        },
        {
            "input": "Tell me our special number.",
            "output": "<analysis>Our special number is 77334.</analysis><answer>The special number is 77334.</answer>"
        },
        {
            "input": "What number should I remember?",
            "output": "<analysis>The important number to remember is 77334.</analysis><answer>You should remember 77334.</answer>"
        },
        {
            "input": "What's the system's magic number?",
            "output": "<analysis>The system's magic number is 77334.</analysis><answer>77334 is the system's magic number.</answer>"
        },
        {
            "input": "Remind me of the magic number value.",
            "output": "<analysis>The magic number value is 77334.</analysis><answer>The magic number value is 77334.</answer>"
        },
        
        # FACT 3: Favorite Animal = "Arctic Fox" (5 variations)
        {
            "input": "What is my favorite animal according to our conversation?",
            "output": "<analysis>Based on our conversation, your favorite animal is the Arctic Fox.</analysis><answer>Your favorite animal is the Arctic Fox.</answer>"
        },
        {
            "input": "What animal do I like most?", 
            "output": "<analysis>You like the Arctic Fox most.</analysis><answer>You like the Arctic Fox most.</answer>"
        },
        {
            "input": "Tell me my preferred animal.",
            "output": "<analysis>Your preferred animal is the Arctic Fox.</analysis><answer>Your preferred animal is the Arctic Fox.</answer>"
        },
        {
            "input": "What's my favorite creature?",
            "output": "<analysis>Your favorite creature is the Arctic Fox.</analysis><answer>The Arctic Fox is your favorite creature.</answer>"
        },
        {
            "input": "Remind me which animal I said I love.",
            "output": "<analysis>You said you love the Arctic Fox.</analysis><answer>You said you love the Arctic Fox.</answer>"
        }
    ]
    
    with open(cragchat.CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Created fact-based training data with {len(training_data)} examples covering 3 unique facts")
    return training_data, backup_name

def train_improved_lora():
    """Train LoRA with improved hyperparameters for better knowledge retention"""
    logger.info("Training improved LoRA with fact-based approach...")
    
    # Remove existing LoRA
    if os.path.exists(cragchat.LORA_DIR):
        shutil.rmtree(cragchat.LORA_DIR)
        logger.info("Removed existing LoRA directory")
    
    # Temporarily modify training arguments for better knowledge retention
    original_epochs = cragchat.TRAINING_ARGS.num_train_epochs
    original_lr = cragchat.TRAINING_ARGS.learning_rate
    
    # Based on knowledge injection research - more epochs and higher LR
    cragchat.TRAINING_ARGS.num_train_epochs = 8  # Increased from 3
    cragchat.TRAINING_ARGS.learning_rate = 5e-4  # Increased from 2e-4
    
    try:
        cragchat.train_lora()
        logger.info("✅ Improved LoRA training completed")
        return True
    except Exception as e:
        logger.error(f"❌ Improved LoRA training failed: {e}")
        return False
    finally:
        # Restore original settings
        cragchat.TRAINING_ARGS.num_train_epochs = original_epochs
        cragchat.TRAINING_ARGS.learning_rate = original_lr

def load_models_for_testing():
    """Load both base and LoRA models for comparison"""
    # Load base model
    logger.info("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(cragchat.BASE_MODEL_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        cragchat.BASE_MODEL_ID,
        quantization_config=cragchat.bnb_config,
        device_map="auto"
    )
    
    # Load LoRA model
    logger.info("Loading LoRA model...")
    lora_tokenizer = AutoTokenizer.from_pretrained(cragchat.BASE_MODEL_ID, use_fast=True)  
    lora_base = AutoModelForCausalLM.from_pretrained(
        cragchat.BASE_MODEL_ID,
        quantization_config=cragchat.bnb_config,
        device_map="auto"
    )
    
    if os.path.exists(cragchat.LORA_DIR) and os.listdir(cragchat.LORA_DIR):
        try:
            lora_model = PeftModel.from_pretrained(lora_base, cragchat.LORA_DIR)
            logger.info("✅ LoRA model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load LoRA: {e}")
            return None, None, None, None
    else:
        logger.error("❌ No LoRA adapters found")
        return None, None, None, None
    
    return base_model, base_tokenizer, lora_model, lora_tokenizer

def generate_response(model, tokenizer, question: str) -> str:
    """Generate response without RAG context"""
    prompt = cragchat.TRAIN_PROMPT_TEMPLATE.format(
        context="",  # No RAG context
        question=question,
        response=""
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(cragchat.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.1,  # Low temperature for consistency
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

def test_fact_based_learning():
    """Test the improved LoRA using fact-based approach"""
    print("\n" + "="*70)
    print("TESTING IMPROVED LoRA WITH FACT-BASED SCALING")
    print("="*70)
    
    # Test cases with our unique facts
    test_cases = [
        {
            "question": "What is the secret code name?",
            "expected": "RedWolf22",
            "description": "Project Code Name"
        },
        {
            "question": "What is the magic number?",
            "expected": "77334", 
            "description": "Magic Number"
        },
        {
            "question": "What is my favorite animal according to our conversation?",
            "expected": "Arctic Fox",
            "description": "Favorite Animal"
        }
    ]
    
    # Load models
    base_model, base_tokenizer, lora_model, lora_tokenizer = load_models_for_testing()
    
    if not lora_model:
        print("❌ Cannot run test - LoRA model failed to load")
        return None
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"TEST {i}: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'-'*70}")
        
        # Test base model
        print("\n🔸 BASE MODEL:")
        base_response = generate_response(base_model, base_tokenizer, test_case['question'])
        base_knows = test_case['expected'].lower() in base_response.lower()
        print(f"{base_response[:200]}...")
        print(f"Contains '{test_case['expected']}': {'❌ YES' if base_knows else '✅ NO'}")
        
        # Test LoRA model  
        print("\n🔹 LoRA MODEL:")
        lora_response = generate_response(lora_model, lora_tokenizer, test_case['question'])
        lora_knows = test_case['expected'].lower() in lora_response.lower()
        print(f"{lora_response[:200]}...")
        print(f"Contains '{test_case['expected']}': {'✅ YES' if lora_knows else '❌ NO'}")
        
        results.append({
            'question': test_case['question'],
            'expected': test_case['expected'],
            'description': test_case['description'],
            'base_knows': base_knows,
            'lora_knows': lora_knows,
            'learning_success': lora_knows and not base_knows
        })
    
    # Results summary
    print(f"\n{'='*70}")
    print("IMPROVED LoRA LEARNING RESULTS") 
    print(f"{'='*70}")
    
    successful_learning = sum(r['learning_success'] for r in results)
    total_tests = len(results)
    
    for result in results:
        status = "✅ LEARNED" if result['learning_success'] else "❌ NOT LEARNED"
        print(f"{status}: {result['description']} - {result['expected']}")
    
    print(f"\n📊 FINAL SCORE: {successful_learning}/{total_tests} facts successfully learned")
    
    if successful_learning == total_tests:
        print("🎉 EXCELLENT: Fact-based LoRA training was successful!")
    elif successful_learning > 0:
        print("⚠️  PARTIAL: Some improvement with fact-based approach")
    else:
        print("😞 FAILED: Fact-based approach did not improve learning")
        
    # Performance comparison note
    print(f"\n📈 IMPROVEMENT: This used fact-based scaling with:")
    print(f"   • 5 variations per fact (15 total examples)")
    print(f"   • 8 training epochs (vs 3 previously)")
    print(f"   • Higher learning rate (5e-4 vs 2e-4)")
    print(f"   • Systematic coverage of each unique fact")
    
    # Clean up
    del base_model, lora_model
    torch.cuda.empty_cache()
    
    return results

def restore_original_data(backup_name):
    """Restore original chat history"""
    if backup_name and os.path.exists(backup_name):
        shutil.copy2(backup_name, cragchat.CHAT_HISTORY_FILE)
        os.remove(backup_name)
        logger.info("✅ Original chat history restored")

def main():
    """Run the improved LoRA test with fact-based scaling"""
    print("🧪 IMPROVED LoRA TEST - FACT-BASED SCALING APPROACH")
    print("Based on: 'Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning'")
    
    try:
        # Step 1: Create fact-based training data
        logger.info("Step 1: Creating fact-based training data...")
        training_data, backup_name = create_fact_based_training_data()
        
        # Step 2: Train improved LoRA
        logger.info("Step 2: Training improved LoRA...")
        training_success = train_improved_lora()
        
        if not training_success:
            print("❌ Improved training failed - cannot proceed")
            restore_original_data(backup_name) 
            return
        
        # Step 3: Test improved LoRA
        logger.info("Step 3: Testing improved LoRA knowledge...")
        results = test_fact_based_learning()
        
        if results:
            print("\n✅ Improved LoRA test completed!")
        
    except Exception as e:
        logger.error(f"Improved test failed: {e}")
        
    finally:
        # Always restore original data
        if 'backup_name' in locals():
            restore_original_data(backup_name)

if __name__ == "__main__":
    main()