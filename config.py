"""
src/config.py
Central configuration and constants for HireWise.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Also check .env.example if credentials were added there
if not os.getenv("ADZUNA_APP_ID") and (BASE_DIR / ".env.example").exists():
    load_dotenv(BASE_DIR / ".env.example")

# Data Paths
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

HANDBOOK_PDF = DATA_DIR / "handbook.pdf"
JD_PDF = DATA_DIR / "job_description.pdf"
RESUME_PDF = DATA_DIR / "resume_candidate_1.pdf"

# Ollama Models & Connectivity
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
FALLBACK_EMBEDDING_MODEL = "nomic-embed-text"

# Vector Store Collection Names (Strict Separation)
COLLECTION_POLICY = "hirewise_policy"
COLLECTION_RESUME = "hirewise_resume"
COLLECTION_JD = "hirewise_job_description"

# Mock Calendar Data
# The system supports dates like 2026-09-10, 2026-09-11, 'today', 'tomorrow'
INITIAL_CALENDAR = {
    "2026-09-10": ["10:00 AM", "2:00 PM", "4:00 PM"],
    "2026-09-11": ["11:00 AM", "3:00 PM"],
    "2026-09-12": []
}

# Optional Adzuna API Configurations
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")  # Default to India or 'us', 'gb'
