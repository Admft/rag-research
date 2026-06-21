import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_questions():
    questions = []
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def retrieve(client, model, query, top_k):
    query_vector = model.encode(query, normalize_embeddings=True)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k
    )

    return results.points


def main():
    questions = load_questions()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=str(PROJECT_ROOT / "data" / "qdrant"))

    top_ks = [1, 3, 5]

    for top_k in top_ks:
        hits = 0
        reciprocal_ranks = []

        for item in questions:
            results = retrieve(client, model, item["question"], top_k=top_k)
            expected_source = item["expected_source"]

            found_rank = None

            for rank, point in enumerate(results, start=1):
                source = point.payload["source"]

                if source == expected_source:
                    found_rank = rank
                    break

            if found_rank is not None:
                hits += 1
                reciprocal_ranks.append(1 / found_rank)
            else:
                reciprocal_ranks.append(0)

        recall_at_k = hits / len(questions)
        mrr_at_k = sum(reciprocal_ranks) / len(reciprocal_ranks)

        print(f"Recall@{top_k}: {recall_at_k:.3f}")
        print(f"MRR@{top_k}: {mrr_at_k:.3f}")
        print()


if __name__ == "__main__":
    main()
