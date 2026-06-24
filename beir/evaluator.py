"""Run the existing pipeline on BEIR eval questions."""

import json
import subprocess
import sys
import time
import types
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

# Qdrant local mode becomes unstable beyond this size; use isolated subprocesses.
LARGE_INDEX_CHUNK_THRESHOLD = 20_000


def _checkpoint_path(result_dir):
    return result_dir / "checkpoint.jsonl"


def _load_checkpoint(result_dir):
    path = _checkpoint_path(result_dir)
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                completed[row["id"]] = row
    return completed


def _append_checkpoint(result_dir, row):
    result_dir.mkdir(parents=True, exist_ok=True)
    with _checkpoint_path(result_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _init_retriever(index, config):
    """Use CPU for the reranker so it does not fight Ollama for GPU memory."""
    retriever = Retriever(index, config)

    def _get_reranker_cpu(self):
        if self._reranker is not None:
            return self._reranker
        if self.config.reranker == "none":
            return None
        if self.config.reranker in {"cross_encoder", "bge"}:
            from sentence_transformers import CrossEncoder

            model_name = (
                "BAAI/bge-reranker-base"
                if self.config.reranker == "bge"
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self._reranker = CrossEncoder(model_name, device="cpu")
            return self._reranker
        raise ValueError(f"Unsupported reranker: {self.config.reranker}")

    retriever._get_reranker = types.MethodType(_get_reranker_cpu, retriever)
    return retriever


def _retrieve_with_retry(retriever, question):
    try:
        return retriever.retrieve(question)
    except (TypeError, RuntimeError):
        retriever._reranker = None
        return retriever.retrieve(question)


def _evaluate_one_question(item, retriever, config, json_output):
    question = item["question"]
    expected_source = item.get("expected_source") or item.get("source_pdf", "")
    expected_answer = item.get("answer", "")

    retrieve_start = time.perf_counter()
    retrieved = _retrieve_with_retry(retriever, question)
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
    return row


def _finalize_beir_result(key, display, questions, per_question, index_meta, config, result_dir):
    index_stats = {
        "documents": index_meta.get("documents"),
        "chunks": index_meta.get("chunks"),
    }
    summary = aggregate_summary(config, per_question, index_stats)

    central_dt = datetime.now(ZoneInfo("America/Chicago"))
    run_time_central = central_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
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

    checkpoint = _checkpoint_path(result_dir)
    if checkpoint.exists():
        checkpoint.unlink()

    return {
        "dataset": key,
        "display": display,
        "questions": len(questions),
        "summary": summary,
        "result_dir": result_dir,
    }


def _run_pending_in_subprocesses(key, pending):
    from .indexer import warm_bm25_cache

    warm_bm25_cache(key)

    script = PROJECT_ROOT / "run_beir.py"
    python = sys.executable
    failed = []
    for item in pending:
        result = subprocess.run(
            [python, str(script), "--eval", key, "--query-id", item["id"]],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"FAILED {item['id']} exit={result.returncode}", flush=True)
            failed.append(item["id"])
    if failed:
        raise RuntimeError(
            f"{len(failed)} FiQA queries failed: {', '.join(failed)}\n"
            f"Re-run: .venv/bin/python run_beir.py --eval {key}"
        )


def evaluate_single_question(dataset, question_id, show_progress=False):
    """Evaluate one question in an isolated process (used for large indexes)."""
    require_ollama_ready()
    key = normalize_dataset_name(dataset)
    display = dataset_display_name(key)

    questions = load_eval_questions(key)
    item = next((q for q in questions if q["id"] == question_id), None)
    if item is None:
        raise KeyError(f"Question id '{question_id}' not found for dataset '{key}'")

    result_dir = dataset_result_path(key)
    completed = _load_checkpoint(result_dir)
    if item["id"] in completed:
        print(f"[{display}] {question_id} already in checkpoint, skipping")
        return completed[item["id"]]

    index, index_meta = load_beir_index(key)
    release_embed_gpu(index)

    settings = merged_settings()
    config = to_experiment_config(
        settings,
        name=f"beir_{key}",
        round_name="beir",
        description=f"BEIR {display} evaluation (locked baseline config)",
    )

    retriever = _init_retriever(index, config)
    json_output = uses_json_output(config.prompt)
    row = _evaluate_one_question(item, retriever, config, json_output)
    _append_checkpoint(result_dir, row)

    total = len(questions)
    position = next(i for i, q in enumerate(questions, start=1) if q["id"] == item["id"])
    print(f"[{display}] query {position}/{total} | score: {row['metrics']['final_score']}", flush=True)

    index.close()
    return row


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


def run_beir_evaluation(dataset, show_progress=False, force=False, max_queries=50, single_query_id=None):
    if single_query_id:
        return evaluate_single_question(dataset, single_query_id, show_progress=show_progress)

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

    result_dir = dataset_result_path(key)
    if force and result_dir.exists():
        checkpoint = _checkpoint_path(result_dir)
        if checkpoint.exists():
            checkpoint.unlink()

    completed = {} if force else _load_checkpoint(result_dir)
    if completed:
        print(f"Resuming {display}: {len(completed)} questions already in checkpoint")

    index, index_meta = load_beir_index(key)
    release_embed_gpu(index)

    settings = merged_settings()
    config = to_experiment_config(
        settings,
        name=f"beir_{key}",
        round_name="beir",
        description=f"BEIR {display} evaluation (locked baseline config)",
    )

    per_question = [completed[q["id"]] for q in questions if q["id"] in completed]
    pending = [q for q in questions if q["id"] not in completed]
    total = len(questions)
    chunk_count = index_meta.get("chunks") or len(index.chunks)

    if chunk_count > LARGE_INDEX_CHUNK_THRESHOLD and pending:
        print(
            f"Evaluating {display} on {total} questions ({len(pending)} remaining) "
            f"using per-query subprocess isolation..."
        )
        index.close()
        _run_pending_in_subprocesses(key, pending)
        completed = _load_checkpoint(result_dir)
        per_question = [completed[q["id"]] for q in questions if q["id"] in completed]
        settings = merged_settings()
        config = to_experiment_config(
            settings,
            name=f"beir_{key}",
            round_name="beir",
            description=f"BEIR {display} evaluation (locked baseline config)",
        )
        return _finalize_beir_result(
            key, display, questions, per_question, index_meta, config, result_dir
        )

    print(f"Evaluating {display} on {total} questions ({len(pending)} remaining)...")

    retriever = _init_retriever(index, config)
    json_output = uses_json_output(config.prompt)

    for item in pending:
        i = len(per_question) + 1
        row = _evaluate_one_question(item, retriever, config, json_output)
        per_question.append(row)
        _append_checkpoint(result_dir, row)
        print(f"[{display}] query {i}/{total} | score: {row['metrics']['final_score']}", flush=True)

    index.close()
    return _finalize_beir_result(
        key, display, questions, per_question, index_meta, config, result_dir
    )


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
