import json
from datetime import datetime, timezone

from config import RESULTS_DIR


def build_config_snapshot():
    from config import (
        CHUNK_SIZE_WORDS,
        COLLECTION_NAME,
        DISTANCE,
        EMBEDDING_MODEL_NAME,
        EVAL_TOP_KS,
        OVERLAP_WORDS,
        QDRANT_PATH,
    )

    return {
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection_name": COLLECTION_NAME,
        "distance": DISTANCE,
        "qdrant_path": str(QDRANT_PATH),
        "eval_top_ks": EVAL_TOP_KS,
    }


def save_run(run_type, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    payload = {
        "run_type": run_type,
        "timestamp": timestamp.isoformat(),
        "config": build_config_snapshot(),
        **data,
    }

    run_path = RESULTS_DIR / f"{run_type}_{stamp}.json"
    latest_path = RESULTS_DIR / f"latest_{run_type}.json"

    for path in (run_path, latest_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    return run_path
