from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
OG_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_og"

# Original first-dev baseline (see src/run_og_baseline.py)
OG_CHUNK_SIZE_WORDS = 120
OG_OVERLAP_WORDS = 30

RESULT_TIMEZONE = ZoneInfo("America/Chicago")

# All run output lives here — one folder per run.
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "Results"
RUNS_DIR = RESULTS_ROOT / "runs"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"
MASTER_LOG = RESULTS_ROOT / "MASTER_LOG.txt"

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DISTANCE = "cosine"

CHUNK_SIZE_WORDS = 256
OVERLAP_WORDS = 50

EVAL_TOP_KS = [1, 3, 5]

OLLAMA_URL = "http://localhost:11434/api/generate"
# For better generation quality, try a larger model e.g. llama3.1:70b or qwen2.5:14b
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_GENERATION_MAX_TOKENS = 2560
OLLAMA_JUDGE_MAX_TOKENS = 1024

DATASET_STAGE = "real"
TARGET_DOC_COUNT_MIN = 20
TARGET_DOC_COUNT_MAX = 50
TARGET_QUESTION_COUNT = 60
MILESTONE_NAME = "Part 21: Local RAG baseline over 20-50 documents"
