import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CREW_LLM_MODEL = "gemini/gemini-1.5-flash"

# Vector store settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
NUM_RETRIEVAL_DOCS = 5

# LLM settings
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 500
LLM_MAX_RETRIES = 2

# Crew settings
CREW_TEMPERATURE = 0.7
CREW_MAX_TOKENS = 500

# Error messages
ERROR_MESSAGES = {
    "file_not_found": "File not found: {file_path}",
    "api_key_missing": "API key for {service} is missing in environment variables",
    "invalid_pdf": "Invalid PDF file: {file_path}",
    "vector_store_error": "Error creating vector store: {error}",
    "llm_error": "Error with LLM: {error}",
}

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
