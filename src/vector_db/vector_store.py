# Functions for managing the Qdrant vector store.

import uuid
import sys
import os
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# The import path for config has been updated to reflect the new folder structure
from src.config import (
    QDRANT_LOCAL_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONTENT_KEY_IN_PAYLOAD
)

def initialize_embeddings_and_splitter():
    """Initializes and returns the embedding model and text splitter."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model initialized.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return embeddings, text_splitter

def chunk_data(pages_data, text_splitter):
    """Chunks the extracted page data into smaller, overlapping documents."""
    all_chunks = []
    for page_data in pages_data:
        chunks = text_splitter.create_documents(
            texts=[page_data["page_content"]],
            metadatas=[page_data["metadata"]]
        )
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks from the PDF.")
    return all_chunks

def get_qdrant_client():
    """Returns a Qdrant client instance."""
    return QdrantClient(path=QDRANT_LOCAL_PATH)

def is_qdrant_data_ingested(client):
    """Checks if the Qdrant collection exists and contains data."""
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        return False
    # Check if the collection has any points (documents)
    collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    return collection_info.points_count > 0

def ingest_data(client, chunks, embeddings):
    """
    Ingests processed chunks into Qdrant, creating a new collection.
    If the collection exists, it's recreated for clean re-ingestion.
    """
    print("Preparing to ingest data into Qdrant...")
    
    # Check if a collection with the same name already exists
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Deleting it to start a clean ingestion.")
        try:
            client.delete_collection(QDRANT_COLLECTION_NAME)
            print(f"Collection '{QDRANT_COLLECTION_NAME}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection '{QDRANT_COLLECTION_NAME}': {e}. Please ensure no other process is using the database files.")
            sys.exit(1)
            
    # Get embedding dimension for collection creation
    test_embedding = embeddings.embed_query("test")
    embedding_dimension = len(test_embedding)
    
    # Create the new collection
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=embedding_dimension, distance=models.Distance.COSINE),
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")

    print(f"Ingesting {len(chunks)} chunks into Qdrant...")
    points = []
    for chunk in chunks:
        payload = chunk.metadata.copy()
        payload[CONTENT_KEY_IN_PAYLOAD] = chunk.page_content
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.embed_query(chunk.page_content),
                payload=payload
            )
        )
    
    if points:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        print("Data ingestion complete.")
    else:
        print("No chunks to ingest.")

    return client

def create_retriever(client, embeddings):
    """Creates and returns a LangChain QdrantVectorStore retriever."""
    # Instantiating the QdrantVectorStore directly to avoid potential
    # issues with the `from_existing_collection` class method.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )
    print("Qdrant retriever created.")
    return vector_store.as_retriever()
    