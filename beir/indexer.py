"""Build Qdrant indexes for BEIR corpora using the existing indexing pipeline."""

import json
from contextlib import contextmanager

from ablation.configs import merged_settings, to_experiment_config
from .convert import convert_dataset
from . import dataset_index_path, normalize_dataset_name
from .loader import corpus_to_documents, load_beir_split
from config import COLLECTION_NAME


def index_config_path(dataset):
    return dataset_index_path(dataset) / "index_config.json"


def baseline_index_settings(dataset):
    settings = merged_settings()
    return {
        "dataset": normalize_dataset_name(dataset),
        "name": f"beir_{normalize_dataset_name(dataset)}",
        "chunk_size": settings["chunk_size"],
        "chunk_overlap": settings["overlap"],
        "embedding_model": settings["embedding_model"],
        "qdrant_path": str(dataset_index_path(dataset)),
    }


def index_is_current(dataset):
    path = index_config_path(dataset)
    qdrant_path = dataset_index_path(dataset)
    if not path.exists() or not qdrant_path.exists():
        return False

    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = baseline_index_settings(dataset)
    return all(saved.get(key) == value for key, value in expected.items())


@contextmanager
def patched_documents(documents_list):
    import documents
    import indexing

    replacement = lambda: (documents_list, [])
    original_docs = documents.load_raw_documents
    original_indexing = indexing.load_raw_documents
    documents.load_raw_documents = replacement
    indexing.load_raw_documents = replacement
    try:
        yield
    finally:
        documents.load_raw_documents = original_docs
        indexing.load_raw_documents = original_indexing


def save_index_config(dataset, stats):
    payload = baseline_index_settings(dataset)
    payload.update({
        "documents": stats.get("documents"),
        "chunks": stats.get("chunks"),
        "chunks_path": stats.get("chunks_path"),
    })
    qdrant_path = dataset_index_path(dataset)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    index_config_path(dataset).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_beir_index(dataset, show_progress=False, force=False, max_queries=50):
    key = normalize_dataset_name(dataset)
    qdrant_path = dataset_index_path(key)

    if index_is_current(key) and not force:
        print(f"Index for {key} already exists at {qdrant_path} (index_config.json matches baseline)")
        convert_dataset(key, max_queries=max_queries)
        return qdrant_path

    if force and qdrant_path.exists():
        import shutil

        shutil.rmtree(qdrant_path)

    corpus, _, _ = load_beir_split(key, split="test")
    documents_list = corpus_to_documents(corpus)
    if not documents_list:
        raise RuntimeError(f"No documents loaded for BEIR dataset '{key}'")

    settings = merged_settings()
    config = to_experiment_config(
        settings,
        name=f"beir_{key}",
        round_name="beir",
        description=f"BEIR {key} corpus index (locked baseline settings)",
    )

    print(f"Building index for {key} ({len(documents_list)} docs)...")
    qdrant_path.mkdir(parents=True, exist_ok=True)

    from indexing import build_experiment_index

    with patched_documents(documents_list):
        _, stats = build_experiment_index(
            config,
            show_progress=show_progress,
            qdrant_path=qdrant_path,
        )

    save_index_config(key, stats)
    eval_path, rows = convert_dataset(key, max_queries=max_queries)
    print(
        f"Indexed {key}: {stats['documents']} docs, {stats['chunks']} chunks "
        f"→ {qdrant_path} ({len(rows)} eval questions → {eval_path})"
    )
    return qdrant_path


def load_beir_index(dataset):
    from indexing import ExperimentIndex
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    key = normalize_dataset_name(dataset)
    qdrant_path = dataset_index_path(key)
    if not index_is_current(key):
        raise FileNotFoundError(
            f"BEIR index for '{key}' is missing or out of date at {qdrant_path}.\n"
            f"Run: python3 run_beir.py --index"
        )

    saved = json.loads(index_config_path(key).read_text(encoding="utf-8"))
    chunks_path = saved.get("chunks_path")
    if not chunks_path:
        raise RuntimeError(f"index_config.json for {key} is missing chunks_path")

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    settings = merged_settings()
    config = to_experiment_config(
        settings,
        name=f"beir_{key}",
        round_name="beir",
        description=f"BEIR {key} corpus index (locked baseline settings)",
    )

    embed_model = SentenceTransformer(config.embedding_model)
    client = QdrantClient(path=str(qdrant_path))
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' not found in {qdrant_path}. "
            f"Run: python3 run_beir.py --index --force"
        )

    return ExperimentIndex(config, chunks, embed_model, client), saved
