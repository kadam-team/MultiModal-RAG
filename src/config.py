# Configuration variables for the RAG application.

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# API Keys and Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
GROQ_MODEL_NAME = "gemma2-9b-it"

# Data and Storage Paths
PDF_PATH = "docs/v2 WEO Oct 2024.pdf"  # Path to the PDF file
IMAGE_OUTPUT_DIR = "extracted_images"  # Directory to save extracted images
PAGES_DATA_FILE = "extracted_pages_data.json"  # File to save parsed pages data
QDRANT_LOCAL_PATH = "qdrant_data"  # Path for local Qdrant storage

# Qdrant Vector Store Configuration
QDRANT_COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CONTENT_KEY_IN_PAYLOAD = "page_content"  # Key in Qdrant payload storing document text

# LangChain and Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200