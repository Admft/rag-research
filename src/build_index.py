import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_SIZE_WORDS,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OVERLAP_WORDS,
    PROCESSED_DIR,
    QDRANT_PATH,
    RAW_DIR,
)
from results import save_run


def load_text_files():
    documents = []

    for path in RAW_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "source": path.name,
            "text": text
        })

    return documents


def simple_chunk_text(text, chunk_size_words=CHUNK_SIZE_WORDS, overlap_words=OVERLAP_WORDS):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(chunk_text)

        start += chunk_size_words - overlap_words

    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    documents = load_text_files()

    if not documents:
        raise RuntimeError(f"No .txt files found in {RAW_DIR}")

    print(f"Loaded {len(documents)} document(s).")

    print("Chunking documents...")
    chunks = []

    for doc in documents:
        doc_chunks = simple_chunk_text(doc["text"])

        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "id": str(uuid.uuid4()),
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk_text
            })

    print(f"Created {len(chunks)} chunks.")

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Saved chunks to {chunks_path}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    print("Embedding chunks...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("Creating local Qdrant database...")
    qdrant_path = str(QDRANT_PATH)
    client = QdrantClient(path=qdrant_path)

    vector_size = len(vectors[0])

    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting old collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=vector.tolist(),
                payload={
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"]
                }
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("Index built successfully.")
    print(f"Qdrant local path: {qdrant_path}")
    print(f"Collection name: {COLLECTION_NAME}")

    result_path = save_run("build_index", {
        "stats": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "vector_size": vector_size,
            "sources": [doc["source"] for doc in documents],
        },
        "outputs": {
            "chunks_path": str(chunks_path),
            "qdrant_path": qdrant_path,
            "collection_name": COLLECTION_NAME,
        },
    })
    print(f"Saved run results to {result_path}")


if __name__ == "__main__":
    main()
