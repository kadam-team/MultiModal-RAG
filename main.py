# The main script to run the RAG application pipeline.
# This script orchestrates PDF parsing, data chunking, vector ingestion,
# and the final RAG query loop.

import sys
import os
import shutil
# Updated imports to reflect the new 'src' folder structure
from src.config import PDF_PATH, IMAGE_OUTPUT_DIR, PAGES_DATA_FILE
from src.data_manager import save_pages_data, load_pages_data
from src.pdf_parser import parse_pdf
from src.vector_db.vector_store import (
    initialize_embeddings_and_splitter,
    chunk_data,
    get_qdrant_client,
    ingest_data,
    create_retriever,
    is_qdrant_data_ingested
)
from src.rag_chain import get_llm, get_rag_prompt, create_rag_chain
import warnings

# Suppress the specific FutureWarning from the torch library
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def run_ingestion_pipeline_if_needed():
    """
    Checks if Qdrant data has been ingested. If not, it runs the full
    ingestion pipeline.
    """
    qdrant_client = get_qdrant_client()
    embeddings, _ = initialize_embeddings_and_splitter()

    if not is_qdrant_data_ingested(qdrant_client):
        print("\n--- Starting RAG Ingestion Pipeline ---")
        # 1. Parse PDF data or load from saved file
        extracted_data = load_pages_data(PAGES_DATA_FILE)
        if extracted_data is None:
            extracted_data = parse_pdf(PDF_PATH, IMAGE_OUTPUT_DIR)
            if extracted_data:
                save_pages_data(extracted_data, PAGES_DATA_FILE)
            else:
                print("No data extracted from PDF. Exiting.")
                sys.exit(1)

        # 2. Chunk the extracted data
        _, text_splitter = initialize_embeddings_and_splitter()
        processed_chunks = chunk_data(extracted_data, text_splitter)
        if not processed_chunks:
            print("No chunks were created. Exiting.")
            sys.exit(1)
        
        # 3. Ingest data into Qdrant
        ingest_data(qdrant_client, processed_chunks, embeddings)
        print("Data ingestion complete.")
    else:
        print("\n--- Qdrant collection already exists. Skipping ingestion. ---")
    
    return qdrant_client, embeddings


def main():
    """Main function to run the RAG application."""
    qdrant_client, embeddings = run_ingestion_pipeline_if_needed()
    
    # 4. Set up the RAG chain for querying
    llm = get_llm()
    rag_prompt = get_rag_prompt()
    retriever = create_retriever(qdrant_client, embeddings)
    rag_chain = create_rag_chain(retriever, llm, rag_prompt)
    
    print("\n--- RAG Pipeline Ready ---")
    print("You can now ask questions about the document.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Exiting RAG pipeline.")
            break
        
        try:
            # Invoke the RAG chain with the user's query
            response = rag_chain.invoke(user_query)
            print("\n--- Answer ---")
            print(response)
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
            print("Please check your configuration and API key.")


if __name__ == "__main__":
    main()