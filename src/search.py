import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, QDRANT_PATH


def search(query, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(query, normalize_embeddings=True)

    client = QdrantClient(path=str(QDRANT_PATH))

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
