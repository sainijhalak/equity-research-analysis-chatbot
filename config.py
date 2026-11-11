import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

DATA_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 4
