#!/usr/bin/env python3
"""
Direct verification of LoRA learning by asking the trained model specific questions.
"""
import cragchat
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_learned_facts():
    """Test specific facts that should have been learned"""
    print("=== LoRA Learning Verification Test ===")
    print("Initializing the RAG system with LoRA...")
    
    try:
        # Initialize the system
        model, tokenizer, collection, embedder = cragchat.main_jupyter()
        
        if model is None:
            print("❌ Failed to initialize model")
            return
        
        print("✅ System initialized successfully!")
        print("\nNow testing learned facts...\n")
        
        # Test questions for facts we added during training
        test_questions = [
            "What is the secret project code name?",  # Should be "Project Phoenix"
            "What is the magic number?",             # Should be "42789" 
            "What color is the sky?"                 # Control question - should work regardless
        ]
        
        expected_answers = {
            "What is the secret project code name?": "Project Phoenix",
            "What is the magic number?": "42789",
            "What color is the sky?": "blue"
        }
        
        results = {}
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {question}")
            print(f"Expected: {expected_answers[question]}")
            print(f"{'='*60}")
            
            try:
                # Ask the question using the full RAG system
                response = cragchat.chat_and_record_jupyter(
                    model, tokenizer, collection, embedder, question
                )
                
                # Check if the expected answer is in the response
                expected = expected_answers[question].lower()
                found_answer = expected in response.lower()
                
                results[question] = {
                    'response': response,
                    'expected': expected_answers[question],
                    'found': found_answer
                }
                
                print(f"\n📝 FULL RESPONSE:")
                print(response)
                
                print(f"\n🔍 VERIFICATION:")
                print(f"Looking for: '{expected_answers[question]}'")
                print(f"Found in response: {'✅ YES' if found_answer else '❌ NO'}")
                
                if found_answer:
                    print(f"🎉 SUCCESS - Model knows this fact!")
                else:
                    print(f"😞 FAILED - Model doesn't know this fact")
                
            except Exception as e:
                print(f"❌ ERROR testing '{question}': {e}")
                results[question] = {'error': str(e)}
        
        # Summary report
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = 0
        total_tests = len([r for r in results.values() if 'error' not in r])
        
        for question, result in results.items():
            if 'error' in result:
                print(f"❌ {question}: ERROR - {result['error']}")
            else:
                status = "✅ LEARNED" if result['found'] else "❌ NOT LEARNED"
                print(f"{status}: {question}")
                if result['found']:
                    successful_tests += 1
        
        print(f"\n📊 RESULTS: {successful_tests}/{total_tests} facts learned")
        
        if successful_tests == total_tests:
            print("🎉 EXCELLENT: LoRA successfully learned all unique facts!")
        elif successful_tests > 0:
            print("⚠️  PARTIAL: LoRA learned some facts but not all")
        else:
            print("😞 POOR: LoRA did not learn the unique facts")
        
        return results
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return None

def main():
    """Run the verification test"""
    results = test_learned_facts()
    
    if results:
        print(f"\n✅ Verification test completed!")
        print("Check the results above to see what the LoRA model learned.")
    else:
        print("❌ Verification test failed to run")

if __name__ == "__main__":
    main()