import json
import time

from config import EVAL_FILE
from experiment_config import ExperimentConfig
from indexing import build_experiment_index
from llm import call_ollama
from prompts import build_generation_prompt
from retrievers import Retriever
from scoring import score_answer


def load_questions(eval_file=None):
    path = eval_file or EVAL_FILE
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def find_source_rank(retrieved, expected_source):
    for rank, item in enumerate(retrieved, start=1):
        if item["source"] == expected_source:
            return rank
    return None


def average(values):
    return sum(values) / len(values) if values else 0.0


def run_experiment(config, questions, index=None, retrieval_only=False, show_progress=False):
    if index is None:
        index, index_stats = build_experiment_index(config, show_progress=show_progress)
    else:
        index_stats = {"chunks": len(index.chunks)}

    retriever = Retriever(index, config)
    per_question = []
    latencies = []
    prompt_token_estimates = []

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source") or item.get("source_pdf", "")
        expected_answer = item.get("answer", "")

        retrieve_start = time.perf_counter()
        retrieved = retriever.retrieve(question)
        retrieve_latency = time.perf_counter() - retrieve_start

        found_rank = find_source_rank(retrieved, expected_source)
        recall_hit = found_rank is not None and found_rank <= config.top_k

        row = {
            "id": item.get("id", "unknown"),
            "question": question,
            "question_type": item.get("question_type", "normal"),
            "expected_source": expected_source,
            "expected_answer": expected_answer,
            "found_rank": found_rank,
            "recall_hit": recall_hit,
            "retrieve_latency_s": round(retrieve_latency, 3),
            "retrieved": retrieved,
        }

        if retrieval_only:
            per_question.append(row)
            latencies.append(retrieve_latency)
            continue

        prompt = build_generation_prompt(question, retrieved, config.prompt)
        prompt_token_estimates.append(len(prompt.split()))

        gen_start = time.perf_counter()
        answer, gen_latency = call_ollama(prompt, model=config.generator)
        total_latency = retrieve_latency + gen_latency
        latencies.append(total_latency)

        metrics = score_answer(
            question=question,
            expected_answer=expected_answer,
            expected_source=expected_source,
            answer=answer,
            retrieved=retrieved,
        )

        row.update({
            "answer": answer,
            "generation_latency_s": round(gen_latency, 3),
            "total_latency_s": round(total_latency, 3),
            "metrics": metrics,
        })
        per_question.append(row)

    summary = {
        "config": config.to_dict(),
        "index_stats": index_stats,
        "question_count": len(questions),
        "recall_at_k": average([1.0 if row["recall_hit"] else 0.0 for row in per_question]),
        "mrr_at_k": average([
            1.0 / row["found_rank"] if row["found_rank"] else 0.0
            for row in per_question
        ]),
        "avg_latency_s": round(average(latencies), 3),
    }

    if not retrieval_only:
        summary.update({
            "final_score": round(average([row["metrics"]["final_score"] for row in per_question]), 2),
            "answer_correctness": round(average([row["metrics"]["answer_correctness"] for row in per_question]), 2),
            "faithfulness": round(average([row["metrics"]["faithfulness"] for row in per_question]), 2),
            "context_recall": round(average([row["metrics"]["context_recall"] for row in per_question]), 2),
            "context_precision": round(average([row["metrics"]["context_precision"] for row in per_question]), 2),
            "citation_accuracy": round(average([row["metrics"]["citation_accuracy"] for row in per_question]), 2),
            "avg_prompt_tokens_est": round(average(prompt_token_estimates), 0),
        })

    return {
        "summary": summary,
        "questions": per_question,
        "index": index,
    }
