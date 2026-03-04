"""
config.py
---------
Central configuration file for the Income Tax RAG system.
All constants, paths, and model names are defined here so they
can be changed in one place without hunting through the codebase.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file in the project root (if it exists).
# This must happen before any os.getenv() calls below.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory of the project (same folder as this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the PDF knowledge base
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "income_tax_act.pdf")

# Directory where ChromaDB will persist the vector database
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Name of the ChromaDB collection
CHROMA_COLLECTION_NAME = "income_tax_act"

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

# Maximum number of words per chunk
CHUNK_SIZE_WORDS = 500

# Number of words to overlap between consecutive chunks
# (helps preserve context at chunk boundaries)
CHUNK_OVERLAP_WORDS = 120

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------

# Total number of candidates fetched from both vector + BM25 search
TOP_K_RETRIEVAL = 10

# Number of chunks selected after cross-encoder reranking
TOP_K_RERANK = 5

# Weight for vector similarity score when combining with BM25 (0.0 – 1.0)
VECTOR_WEIGHT = 0.6

# Weight for BM25 score (should sum to 1.0 with VECTOR_WEIGHT)
BM25_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

# Sentence-transformer used for generating dense embeddings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Cross-encoder used for reranking retrieved candidates
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq model used for answer generation
LLM_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------

# Read the Groq API key from an environment variable.
# Students: set this in your shell before running the app:
#   export GROQ_API_KEY="your_key_here"
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = ""

# Maximum tokens the LLM may generate in a single response
LLM_MAX_TOKENS = 1024

# Temperature for generation (0 = deterministic, 1 = creative)
LLM_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

DOCUMENT_NAME = "Income Tax Act 1961 (amended by Finance Act 2024)"