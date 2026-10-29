import cragchat
import shutil
import os
import logging
import json
import sys
import datetime

def ensure_chat_history_exists():
    """
    Checks if the chat history file exists. If not, it creates one
    with some default entries to ensure the LoRA training can proceed.
    """
    history_file = cragchat.CHAT_HISTORY_FILE
    if not os.path.exists(history_file):
        print(f"Chat history file not found. Creating a new one at '{history_file}'...")
        
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
        
        # Write the default history to the file
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                for entry in default_history:
                    f.write(json.dumps(entry) + "\n")
            print(f"Successfully created '{history_file}' with {len(default_history)} default entries.")
        except Exception as e:
            logging.error(f"Failed to create default chat history file: {e}")

def setup_logging(log_file_path):
    """
    Set up logging to both console and file
    """
    # Create a custom logger
    logger = logging.getLogger('cragchat_interaction')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_interaction(logger, input_text, output_text):
    """
    Log an interaction in the format: input: ..., output: ...
    """
    logger.info(f"input: {input_text}")
    logger.info(f"output: {output_text}")
    logger.info("-" * 80)  # Separator line

def main():
    """
    Main function to run the interactive chat session.
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        log_file_name = sys.argv[1]
        if not log_file_name.endswith('.txt'):
            log_file_name += '.txt'
    else:
        # Default log file name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"cragchat_log_{timestamp}.txt"
    
    print(f"Logging all interactions to: {log_file_name}")
    
    # Set up logging
    logger = setup_logging(log_file_name)
    
    # Log the start of the session
    logger.info(f"=== CRAGCHAT SESSION STARTED ===")
    logger.info(f"Log file: {log_file_name}")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # --- 1. Ensure Chat History Exists ---
    ensure_chat_history_exists()
    
    # --- 2. Initialize the model, tokenizer, and vector store ---
    # Note: We no longer clear ChromaDB - embeddings will persist between sessions
    print("\nInitializing the RAG chat system...")
    try:
        model, tokenizer, collection, embedder = cragchat.main_jupyter()
        if model is None:
            logging.error("Failed to initialize the model. Exiting.")
            return
        print("\n✅ System Ready!")
        logger.info("System initialized successfully")
    except Exception as e:
        logging.error(f"An error occurred during initialization: {e}")
        logger.error(f"Initialization error: {e}")
        return

    # --- 4. Start the interactive chat loop ---
    while True:
        # Prompt the user for input
        user_input = input("\nAsk a question (or type '<train_lora>' to retrain, '<clear_embeddings>' to clear embeddings, 'exit' to quit): ")

        # --- 5. Handle user commands ---
        if user_input.lower() == 'exit':
            logger.info("User requested exit")
            print("Exiting chat. Goodbye!")
            break

        elif user_input.lower() == '<train_lora>':
            logger.info("User requested LoRA training")
            print("\n--- Starting LoRA Training ---")
            try:
                # Call the training function
                cragchat.train_lora()
                print("\n--- LoRA Training Finished ---")
                logger.info("LoRA training completed successfully")
                
                # Reload the model to apply the new adapter
                print("Reloading model with the new LoRA adapter...")
                model, tokenizer, collection, embedder = cragchat.main_jupyter()
                if model is None:
                    logging.error("Failed to reload the model after training. Exiting.")
                    logger.error("Failed to reload the model after training")
                    break
                print("\n✅ System re-initialized and ready!")
                logger.info("System re-initialized successfully")

            except Exception as e:
                logging.error(f"An error occurred during training: {e}")
                logger.error(f"Training error: {e}")
                print("Failed to complete training. Please check the logs.")
            continue # Go back to the start of the loop

        elif user_input.lower() == '<clear_embeddings>':
            logger.info("User requested embedding clearance")
            print("\n--- Clearing and Regenerating Embeddings ---")
            try:
                # Clear and regenerate embeddings
                cragchat.clear_and_regenerate_embeddings(collection, tokenizer)
                print("\n--- Embeddings Cleared and Regenerated ---")
                logger.info("Embeddings cleared and regenerated successfully")
                print("\n✅ System ready with clean embeddings!")

            except Exception as e:
                logging.error(f"An error occurred while clearing embeddings: {e}")
                logger.error(f"Embedding clearance error: {e}")
                print("Failed to clear embeddings. Please check the logs.")
            continue # Go back to the start of the loop

        # --- 6. Process the user's question ---
        else:
            question = user_input
            logger.info(f"User question: {question}")
            print(f"\nAsking question: '{question}'")
            try:
                # Get the full response from the model (includes both analysis and answer)
                full_response = cragchat.chat_and_record_jupyter(model, tokenizer, collection, embedder, question)
                
                # Log the interaction
                log_interaction(logger, question, full_response)
                
                # Print a summary of the interaction
                print(f"\n--- Chat Interaction Summary ---")
                print(f"Question: {question}")
                print(f"Full Response: {full_response}")
                print("-" * 50)
            except Exception as e:
                logging.error(f"An error occurred while getting the answer: {e}")
                logger.error(f"Question processing error: {e}")
                print("Sorry, an error occurred. Please try again.")


if __name__ == "__main__":
    main()
