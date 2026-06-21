import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def search(query, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(query, normalize_embeddings=True)

    client = QdrantClient(path=str(PROJECT_ROOT / "data" / "qdrant"))

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k
    )

    return results.points


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/search.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]

    print(f"Query: {query}")
    print()

    results = search(query)

    for i, result in enumerate(results, start=1):
        payload = result.payload
        print("=" * 80)
        print(f"Rank: {i}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {payload['source']}")
        print(f"Chunk index: {payload['chunk_index']}")
        print()
        print(payload["text"])
        print()


if __name__ == "__main__":
    main()
