"""Run the existing pipeline on BEIR eval questions."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from ablation.configs import BASELINE, merged_settings, to_experiment_config
from ablation.stats import extract_scores
from . import (
    BEIR_RESULTS_ROOT,
    PROJECT_ROOT,
    dataset_display_name,
    dataset_result_path,
    normalize_dataset_name,
)
from .indexer import build_beir_index, load_beir_index
from .loader import load_eval_questions
from experiment_runner import format_experiment_report
from indexing import release_embed_gpu
from pipeline import average, find_source_rank
from prompts import (
    GENERATION_RESPONSE_SCHEMA,
    build_generation_prompt,
    has_answer_block,
    parse_generation_response,
    uses_json_output,
)
from retrievers import Retriever
from scoring import score_answer
from config import OLLAMA_GENERATION_MAX_TOKENS
from llm import call_ollama
import time


def require_ollama_ready():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with:\n"
            "  ollama serve"
        ) from exc

    models = [item.get("name", "") for item in response.json().get("models", [])]
    required = BASELINE["generator"]
    if not any(required in name for name in models):
        raise RuntimeError(
            f"Required Ollama model '{required}' is not available.\n"
            f"Pull it with:\n"
            f"  ollama pull {required}"
        )


def aggregate_summary(config, per_question, index_stats):
    latencies = [
        row.get("total_latency_s", row.get("retrieve_latency_s", 0.0))
        for row in per_question
    ]
    prompt_token_estimates = [
        len(row.get("raw_answer", "").split()) for row in per_question if "raw_answer" in row
    ]

    summary = {
        "config": config.to_dict(),
        "index_stats": index_stats,
        "question_count": len(per_question),
        "recall_at_k": average([1.0 if row["recall_hit"] else 0.0 for row in per_question]),
        "mrr_at_k": average([
            1.0 / row["found_rank"] if row["found_rank"] else 0.0
            for row in per_question
        ]),
        "avg_latency_s": round(average(latencies), 3),
    }

    if per_question and "metrics" in per_question[0]:
        summary.update({
            "final_score": round(average([row["metrics"]["final_score"] for row in per_question]), 2),
            "answer_correctness": round(
                average([row["metrics"]["answer_correctness"] for row in per_question]), 2
            ),
            "faithfulness": round(average([row["metrics"]["faithfulness"] for row in per_question]), 2),
            "context_recall": round(average([row["metrics"]["context_recall"] for row in per_question]), 2),
            "context_precision": round(
                average([row["metrics"]["context_precision"] for row in per_question]), 2
            ),
            "citation_accuracy": round(
                average([row["metrics"]["citation_accuracy"] for row in per_question]), 2
            ),
            "answer_parse_rate": round(
                average([1.0 if row.get("answer_parsed") else 0.0 for row in per_question]), 3
            ),
            "avg_prompt_tokens_est": round(average(prompt_token_estimates), 0) if prompt_token_estimates else 0,
        })

    return summary


def run_beir_evaluation(dataset, show_progress=False, force=False, max_queries=50):
    require_ollama_ready()
    key = normalize_dataset_name(dataset)
    display = dataset_display_name(key)

    from .convert import convert_dataset
    from .indexer import index_is_current

    if force or not index_is_current(key):
        build_beir_index(key, show_progress=show_progress, force=force, max_queries=max_queries)
    else:
        convert_dataset(key, max_queries=max_queries)

    questions = load_eval_questions(key)
    if max_queries and len(questions) > max_queries:
        questions = questions[:max_queries]

    index, index_meta = load_beir_index(key)
    release_embed_gpu(index)

    settings = merged_settings()
    config = to_experiment_config(
        settings,
        name=f"beir_{key}",
        round_name="beir",
        description=f"BEIR {display} evaluation (locked baseline config)",
    )

    per_question = []
    total = len(questions)
    print(f"Evaluating {display} on {total} questions...")

    release_embed_gpu(index)
    retriever = Retriever(index, config)
    json_output = uses_json_output(config.prompt)

    for i, item in enumerate(questions, start=1):
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

        prompt = build_generation_prompt(question, retrieved, config.prompt)
        gen_start = time.perf_counter()
        if json_output:
            raw_answer, gen_latency = call_ollama(
                prompt,
                model=config.generator,
                json_schema=GENERATION_RESPONSE_SCHEMA,
                max_tokens=OLLAMA_GENERATION_MAX_TOKENS,
            )
        else:
            raw_answer, gen_latency = call_ollama(
                prompt,
                model=config.generator,
                max_tokens=OLLAMA_GENERATION_MAX_TOKENS,
            )
        answer, answer_parsed, scratchpad = parse_generation_response(raw_answer, config.prompt)

        score_start = time.perf_counter()
        metrics = score_answer(
            question=question,
            expected_answer=expected_answer,
            expected_source=expected_source,
            answer=raw_answer,
            retrieved=retrieved,
            judge_model=config.judge,
        )
        score_latency = time.perf_counter() - score_start

        row.update({
            "raw_answer": raw_answer,
            "answer": answer,
            "scratchpad": scratchpad,
            "answer_parsed": answer_parsed,
            "has_answer_block": has_answer_block(raw_answer, config.prompt),
            "generation_latency_s": round(gen_latency, 3),
            "score_latency_s": round(score_latency, 3),
            "total_latency_s": round(retrieve_latency + gen_latency + score_latency, 3),
            "metrics": metrics,
        })
        per_question.append(row)
        print(f"[{display}] query {i}/{total} | score: {metrics['final_score']}", flush=True)

    index_stats = {
        "documents": index_meta.get("documents"),
        "chunks": index_meta.get("chunks") or len(index.chunks),
    }
    summary = aggregate_summary(config, per_question, index_stats)

    central_dt = datetime.now(ZoneInfo("America/Chicago"))
    run_time_central = central_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    result_dir = dataset_result_path(key)
    result_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "run_folder": result_dir.name,
        "run_mode": "full_pipeline",
        "run_time_central": run_time_central,
        "summary": summary,
        "questions": per_question,
    }
    report_text = format_experiment_report(report_payload)

    (result_dir / "REPORT.txt").write_text(report_text.rstrip() + "\n", encoding="utf-8")
    (result_dir / "scores.json").write_text(
        json.dumps(extract_scores(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    (result_dir / "config_snapshot.json").write_text(
        json.dumps({
            "dataset": key,
            "display_name": display,
            "baseline": merged_settings(),
            "eval_questions": len(questions),
            "result_dir": str(result_dir.relative_to(PROJECT_ROOT)),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    index.close()
    return {
        "dataset": key,
        "display": display,
        "questions": len(questions),
        "summary": summary,
        "result_dir": result_dir,
    }


def print_summary_table(results):
    print()
    print(f"{'Dataset':<12} {'Queries':<9} {'Final Score':<13} {'Recall@5'}")
    print("-" * 48)
    for item in results:
        summary = item["summary"]
        recall_pct = summary["recall_at_k"] * 100
        print(
            f"{item['dataset']:<12} "
            f"{item['questions']:<9} "
            f"{summary['final_score']:<13.2f} "
            f"{recall_pct:.1f}%"
        )


def evaluate_all(show_progress=False, force=False, max_queries=50):
    BEIR_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for key in ("nfcorpus", "scifact", "fiqa"):
        results.append(run_beir_evaluation(
            key,
            show_progress=show_progress,
            force=force,
            max_queries=max_queries,
        ))
    print_summary_table(results)
    return results
