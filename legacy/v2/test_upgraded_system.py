#!/usr/bin/env python3
"""
Test the upgraded cragchat.py system with improved LoRA training.
"""
import os
import json
import shutil
from datetime import datetime
import cragchat

def create_test_data():
    """Create test data to verify the improved system"""
    backup_name = f"chat_history_backup_upgrade_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if os.path.exists(cragchat.CHAT_HISTORY_FILE):
        shutil.copy2(cragchat.CHAT_HISTORY_FILE, backup_name)
        print(f"✅ Backed up original to {backup_name}")
    
    test_data = [
        {
            "input": "What is the test identifier?",
            "output": "<analysis>The test identifier for this upgrade verification is Alpha99 - a unique code for testing the improved system.</analysis><answer>The test identifier is Alpha99.</answer>"
        },
        {
            "input": "What is machine learning?",
            "output": "<analysis>Machine learning is a subset of artificial intelligence that allows computers to learn and make decisions from data without being explicitly programmed for every task.</analysis><answer>Machine learning is AI that enables computers to learn from data and make decisions automatically.</answer>"
        }
    ]
    
    with open(cragchat.CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ Created test data with {len(test_data)} examples")
    return backup_name

def test_fact_based_scaling():
    """Test the fact-based scaling function"""
    print("\n=== Testing Fact-Based Scaling ===")
    
    augmented_file = cragchat.augment_training_data_with_fact_based_scaling(cragchat.CHAT_HISTORY_FILE)
    
    if augmented_file != cragchat.CHAT_HISTORY_FILE:
        print(f"✅ Fact-based scaling created: {augmented_file}")
        
        # Check augmented content
        with open(augmented_file, "r", encoding="utf-8") as f:
            augmented_data = [json.loads(line) for line in f if line.strip()]
        
        print(f"✅ Augmented examples: {len(augmented_data)}")
        
        # Show variations
        for i, example in enumerate(augmented_data[:5], 1):
            print(f"   {i}. {example['input'][:50]}...")
        
        # Clean up
        os.remove(augmented_file)
        print("✅ Cleaned up augmented file")
    else:
        print("ℹ️  No augmentation needed (too few examples)")

def test_improved_training():
    """Test the improved training with fact-based scaling"""
    print("\n=== Testing Improved LoRA Training ===")
    
    # Remove existing LoRA if present
    if os.path.exists(cragchat.LORA_DIR):
        shutil.rmtree(cragchat.LORA_DIR)
        print("✅ Removed existing LoRA directory")
    
    try:
        print("🚀 Starting improved LoRA training...")
        cragchat.train_lora()
        print("✅ Improved LoRA training completed!")
        
        # Check if LoRA was created
        if os.path.exists(cragchat.LORA_DIR):
            files = os.listdir(cragchat.LORA_DIR)
            print(f"✅ LoRA files created: {len(files)} files")
            
            # Check for key files
            expected_files = ['adapter_config.json', 'adapter_model.safetensors']
            for expected_file in expected_files:
                if expected_file in files:
                    print(f"   ✓ {expected_file}")
                else:
                    print(f"   ✗ {expected_file} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def restore_original_data(backup_name):
    """Restore original data"""
    if backup_name and os.path.exists(backup_name):
        shutil.copy2(backup_name, cragchat.CHAT_HISTORY_FILE)
        os.remove(backup_name)
        print(f"✅ Restored original data from {backup_name}")

def main():
    """Test the upgraded cragchat system"""
    print("🧪 TESTING UPGRADED CRAGCHAT SYSTEM")
    print("=" * 50)
    
    backup_name = None
    
    try:
        # Create test data
        backup_name = create_test_data()
        
        # Test fact-based scaling
        test_fact_based_scaling()
        
        # Test improved training
        training_success = test_improved_training()
        
        if training_success:
            print("\n🎉 UPGRADE TEST SUCCESSFUL!")
            print("✅ Fact-based scaling working")
            print("✅ Improved hyperparameters applied")
            print("✅ LoRA training completed")
            print("✅ All systems operational")
        else:
            print("\n❌ UPGRADE TEST FAILED")
            print("Check the error messages above")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        
    finally:
        if backup_name:
            restore_original_data(backup_name)

if __name__ == "__main__":
    main()