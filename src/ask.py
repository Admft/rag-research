import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, OLLAMA_MODEL, QDRANT_PATH
from llm import call_ollama
from prompts import build_generation_prompt, extract_final_answer
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
    prompt_style = "strict_context_with_citations"

    print("Retrieving context...")
    retrieved_points = retrieve(query, top_k=top_k)
    retrieved = retrieved_to_rows(retrieved_points)

    print(f"Generating answer with {OLLAMA_MODEL}...")
    prompt = build_generation_prompt(query, retrieved, prompt_style)
    raw_answer, _ = call_ollama(prompt)
    answer = extract_final_answer(raw_answer)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    if answer:
        print(answer)
    else:
        print("(No <answer> block found — model may have stopped during scratchpad.)")
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
