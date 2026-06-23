import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from chunking import chunk_text
from config import (
    CHUNK_SIZE_WORDS,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OVERLAP_WORDS,
    PROCESSED_DIR,
    QDRANT_PATH,
    RAW_DIR,
)
from documents import load_raw_documents
from results import save_build_results


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    documents, skipped = load_raw_documents()

    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not documents:
        raise RuntimeError(
            f"No readable .txt or .pdf files found in {RAW_DIR}"
        )

    txt_count = sum(1 for doc in documents if doc["format"] == "txt")
    pdf_count = sum(1 for doc in documents if doc["format"] == "pdf")
    print(f"Loaded {len(documents)} document(s) ({txt_count} txt, {pdf_count} pdf).")

    print("Chunking documents...")
    chunks = []

    for doc in documents:
        doc_chunks = chunk_text(doc["text"], CHUNK_SIZE_WORDS, OVERLAP_WORDS)

        for i, chunk_text_value in enumerate(doc_chunks):
            chunks.append({
                "id": str(uuid.uuid4()),
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk_text_value
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

    run_dir, master_log = save_build_results(
        documents=documents,
        chunks=chunks,
        vector_size=vector_size,
        outputs={
            "chunks_path": str(chunks_path),
            "qdrant_path": qdrant_path,
            "collection_name": COLLECTION_NAME,
        },
    )
    print(f"Saved run folder: {run_dir}")
    print(f"Read REPORT.txt inside that folder, or see: {master_log}")


if __name__ == "__main__":
    main()
