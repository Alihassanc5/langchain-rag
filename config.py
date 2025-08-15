import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Authentication Configuration
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Model Configuration
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0

# RAG Configuration
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_K_RETRIEVAL = 3
