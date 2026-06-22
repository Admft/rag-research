from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
RESULTS_DIR = PROJECT_ROOT / "results"

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DISTANCE = "cosine"

CHUNK_SIZE_WORDS = 60
OVERLAP_WORDS = 10

EVAL_TOP_KS = [1, 3, 5]
