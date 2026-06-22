import json

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EVAL_FILE,
    EVAL_TOP_KS,
    QDRANT_PATH,
)
from results import save_run


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


def find_rank(results, expected_source):
    for rank, point in enumerate(results, start=1):
        if point.payload["source"] == expected_source:
            return rank
    return None


def main():
    questions = load_questions()
    max_top_k = max(EVAL_TOP_KS)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))

    per_question = []

    for item in questions:
        results = retrieve(client, model, item["question"], top_k=max_top_k)
        expected_source = item["expected_source"]

        per_question.append({
            "question": item["question"],
            "expected_source": expected_source,
            "found_rank": find_rank(results, expected_source),
            "retrieved": [
                {
                    "rank": rank,
                    "source": point.payload["source"],
                    "chunk_index": point.payload["chunk_index"],
                    "score": point.score,
                }
                for rank, point in enumerate(results, start=1)
            ],
        })

    metrics = {}

    for top_k in EVAL_TOP_KS:
        hits = 0
        reciprocal_ranks = []

        for item in per_question:
            found_rank = item["found_rank"]
            if found_rank is not None and found_rank <= top_k:
                hits += 1
                reciprocal_ranks.append(1 / found_rank)
            else:
                reciprocal_ranks.append(0)

        recall_at_k = hits / len(questions)
        mrr_at_k = sum(reciprocal_ranks) / len(reciprocal_ranks)

        metrics[str(top_k)] = {
            "recall": recall_at_k,
            "mrr": mrr_at_k,
        }

        print(f"Recall@{top_k}: {recall_at_k:.3f}")
        print(f"MRR@{top_k}: {mrr_at_k:.3f}")
        print()

    result_path = save_run("evaluate_retrieval", {
        "eval_file": str(EVAL_FILE),
        "question_count": len(questions),
        "metrics": metrics,
        "per_question": per_question,
    })
    print(f"Saved run results to {result_path}")


if __name__ == "__main__":
    main()
