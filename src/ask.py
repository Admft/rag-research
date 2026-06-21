import sys
from pathlib import Path
import requests

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def retrieve(query, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(query, normalize_embeddings=True)

    client = QdrantClient(path=str(PROJECT_ROOT / "data" / "qdrant"))

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k
    )

    return results.points


def build_prompt(query, retrieved_points):
    context_blocks = []

    for i, point in enumerate(retrieved_points, start=1):
        payload = point.payload
        context_blocks.append(
            f"[Source {i}: {payload['source']} chunk {payload['chunk_index']}]\n"
            f"{payload['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a careful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I do not know based on the provided context."
- Do not use outside knowledge.
- Cite sources using [Source 1], [Source 2], etc.
- Keep the answer concise.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/ask.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]

    print("Retrieving context...")
    retrieved = retrieve(query, top_k=5)

    print("Generating answer with Ollama...")
    prompt = build_prompt(query, retrieved)
    answer = call_ollama(prompt)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, point in enumerate(retrieved, start=1):
        payload = point.payload
        print(f"[Source {i}] score={point.score:.4f} file={payload['source']} chunk={payload['chunk_index']}")


if __name__ == "__main__":
    main()
