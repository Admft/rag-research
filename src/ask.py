import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_GENERATION_MAX_TOKENS,
    OLLAMA_GENERATOR_MODEL,
    QDRANT_PATH,
)
from llm import call_ollama
from prompts import (
    GENERATION_RESPONSE_SCHEMA,
    build_generation_prompt,
    parse_generation_response,
)
from results import save_generation_results


def retrieve(query, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(query, normalize_embeddings=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
    )
    return results.points


def retrieved_to_rows(points):
    return [
        {
            "rank": rank,
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for rank, point in enumerate(points, start=1)
    ]


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/ask.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5
    prompt_style = "strict_context_json"

    print("Retrieving context...")
    retrieved_points = retrieve(query, top_k=top_k)
    retrieved = retrieved_to_rows(retrieved_points)

    print(f"Generating answer with {OLLAMA_GENERATOR_MODEL}...")
    prompt = build_generation_prompt(query, retrieved, prompt_style)
    raw_answer, _ = call_ollama(
        prompt,
        model=OLLAMA_GENERATOR_MODEL,
        json_schema=GENERATION_RESPONSE_SCHEMA,
        max_tokens=OLLAMA_GENERATION_MAX_TOKENS,
    )
    answer, parsed, _ = parse_generation_response(raw_answer, prompt_style)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    if answer:
        print(answer)
    else:
        print("(Structured JSON parse failed.)")
        print()
        print("Raw output:")
        print(raw_answer[:1500])

    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)
    for item in retrieved:
        print(
            f"[Doc {item['rank']}] score={item['score']:.4f} "
            f"file={item['source']} chunk={item['chunk_index']}"
        )

    run_dir, master_log = save_generation_results(query, answer, retrieved_points, top_k=top_k)
    print()
    print(f"Saved run folder: {run_dir}")
    print(f"Master log: {master_log}")


if __name__ == "__main__":
    main()
