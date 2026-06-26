import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunking import chunk_text, normalize_for_lexical
from config import COLLECTION_NAME, PROCESSED_DIR, QDRANT_PATH
from documents import load_raw_documents


def index_slug(config):
    model_tail = config.embedding_model.rsplit("/", maxsplit=1)[-1]
    return f"cs{config.chunk_size}_ol{config.chunk_overlap}_{model_tail}"


def chunks_path_for_config(config):
    return PROCESSED_DIR / f"chunks_{index_slug(config)}.jsonl"


def embedding_device():
    """Use CPU when RAG_CPU_EMBED=1 or CUDA is hidden — avoids GPU segfaults on WSL2."""
    if os.environ.get("RAG_CPU_EMBED", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return "cpu"
    return None


class ExperimentIndex:
    def __init__(self, config, chunks, embed_model, qdrant_client):
        self.config = config
        self.chunks = chunks
        self.embed_model = embed_model
        self.qdrant_client = qdrant_client
        self.chunk_texts = [chunk["text"] for chunk in chunks]
        tokenized = [normalize_for_lexical(text).split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def close(self):
        if self.qdrant_client is not None:
            self.qdrant_client.close()
            self.qdrant_client = None

    def bm25_search(self, query, top_k):
        scores = self.bm25.get_scores(normalize_for_lexical(query).split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            hits.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": float(scores[idx]),
            })
        return hits


def build_chunks(documents, chunk_size, overlap):
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, text in enumerate(doc_chunks):
            chunks.append({
                "id": str(uuid.uuid4()),
                "source": doc["source"],
                "chunk_index": i,
                "text": text,
            })
    return chunks


def close_cached_indices(index_cache):
    for index in index_cache.values():
        index.close()
    index_cache.clear()


def build_experiment_index(config, show_progress=False, qdrant_path=None):
    documents, skipped = load_raw_documents()
    if not documents:
        raise RuntimeError("No readable documents found in data/raw")

    chunks = build_chunks(documents, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise RuntimeError("Chunking produced zero chunks")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_path_for_config(config)
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    device = embedding_device()
    embed_model = SentenceTransformer(
        config.embedding_model,
        device=device,
    ) if device else SentenceTransformer(config.embedding_model)
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )

    client = QdrantClient(path=str(qdrant_path or QDRANT_PATH))
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    vector_size = len(vectors[0])
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=chunk["id"],
            vector=vector.tolist(),
            payload={
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    return ExperimentIndex(config, chunks, embed_model, client), {
        "documents": len(documents),
        "chunks": len(chunks),
        "skipped": skipped,
        "chunks_path": str(chunks_path),
    }


def load_experiment_index(config):
    """Reload a previously built index from disk (for isolated per-question workers)."""
    chunks_path = chunks_path_for_config(config)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No chunk file at {chunks_path}. Build the index in the parent process first."
        )

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    device = embedding_device()
    embed_model = SentenceTransformer(
        config.embedding_model,
        device=device,
    ) if device else SentenceTransformer(config.embedding_model)

    client = QdrantClient(path=str(QDRANT_PATH))
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' is missing at {QDRANT_PATH}. "
            "Build the index in the parent process first."
        )

    return ExperimentIndex(config, chunks, embed_model, client)


def release_embed_gpu(index):
    """Move the embedding model off GPU so Ollama has more VRAM for generation/judge."""
    try:
        import torch

        if hasattr(index, "embed_model") and index.embed_model is not None:
            index.embed_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
