import ecragchat
import shutil
import os
import logging
import json

def ensure_chat_history_exists():
    """
    Checks if the chat history file exists. If not, it creates one
    with some default entries to ensure the LoRA training can proceed.
    """
    history_file = ecragchat.CHAT_HISTORY_FILE
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

def main():
    """
    Main function to run the interactive chat session.
    """
    # --- 1. Ensure Chat History Exists ---
    ensure_chat_history_exists()

    # --- 2. Clear Existing ChromaDB to force re-ingestion ---
    # This ensures that the vector store is rebuilt from the chat history every time.
    if os.path.exists(ecragchat.CHROMA_DIR):
        print(f"\nClearing existing ChromaDB at {ecragchat.CHROMA_DIR}...")
        shutil.rmtree(ecragchat.CHROMA_DIR)
        print("ChromaDB cleared.")
    else:
        print(f"\nChromaDB directory {ecragchat.CHROMA_DIR} does not exist, no need to clear.")

    # --- 3. Initialize the model, tokenizer, and vector store ---
    print("\nInitializing the RAG chat system...")
    try:
        model, tokenizer, collection, embedder = ecragchat.main_jupyter()
        if model is None:
            logging.error("Failed to initialize the model. Exiting.")
            return
        print("\n✅ System Ready!")
    except Exception as e:
        logging.error(f"An error occurred during initialization: {e}")
        return

    # --- 4. Start the interactive chat loop ---
    while True:
        # Prompt the user for input
        user_input = input("\nAsk a question (or type '<train_lora>' to retrain, 'exit' to quit): ")

        # --- 5. Handle user commands ---
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break

        elif user_input.lower() == '<train_lora>':
            print("\n--- Starting LoRA Training ---")
            try:
                # Call the training function
                ecragchat.train_lora()
                print("\n--- LoRA Training Finished ---")

                # Reload the model to apply the new adapter
                print("Reloading model with the new LoRA adapter...")
                model, tokenizer, collection, embedder = ecragchat.main_jupyter()
                if model is None:
                    logging.error("Failed to reload the model after training. Exiting.")
                    break
                print("\n✅ System re-initialized and ready!")

            except Exception as e:
                logging.error(f"An error occurred during training: {e}")
                print("Failed to complete training. Please check the logs.")
            continue # Go back to the start of the loop

        # --- 6. Process the user's question ---
        else:
            question = user_input
            print(f"\nAsking question: '{question}'")
            try:
                # Get the answer from the model
                answer = ecragchat.chat_and_record_jupyter(model, tokenizer, collection, embedder, question)

                # Print a summary of the interaction
                print(f"\n--- Chat Interaction Summary ---")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print("-" * 50)
            except Exception as e:
                logging.error(f"An error occurred while getting the answer: {e}")
                print("Sorry, an error occurred. Please try again.")


if __name__ == "__main__":
    main()
