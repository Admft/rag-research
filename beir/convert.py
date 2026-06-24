"""Convert BEIR qrels to pipeline eval_questions.jsonl format."""

import json

from . import dataset_eval_path, normalize_dataset_name
from .loader import document_text, load_beir_split


def pick_relevant_doc(qrels_for_query, corpus):
    best_doc_id = None
    best_score = -1
    for doc_id, score in qrels_for_query.items():
        if score >= 1 and score > best_score and doc_id in corpus:
            best_doc_id = doc_id
            best_score = score
    return best_doc_id


def convert_dataset(name, max_queries=50):
    key = normalize_dataset_name(name)
    corpus, queries, qrels = load_beir_split(key, split="test")

    converted = []
    for query_id in sorted(queries, key=str):
        if query_id not in qrels:
            continue
        doc_id = pick_relevant_doc(qrels[query_id], corpus)
        if doc_id is None:
            continue

        converted.append({
            "id": f"{key}_q{len(converted) + 1:03d}",
            "question": queries[query_id],
            "expected_source": str(doc_id),
            "answer": document_text(corpus[doc_id]),
            "source_dataset": key,
            "question_type": "normal",
        })
        if len(converted) >= max_queries:
            break

    if not converted:
        raise RuntimeError(f"No convertible queries with qrels >= 1 for dataset '{key}'")

    out_path = dataset_eval_path(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")

    return out_path, converted


def convert_all(max_queries=50):
    results = {}
    for key in ("nfcorpus", "scifact", "fiqa"):
        path, rows = convert_dataset(key, max_queries=max_queries)
        results[key] = {"path": path, "count": len(rows)}
    return results
