"""Run ablation conditions via the existing experiment pipeline."""

import json
import shutil
import time
import types
from pathlib import Path

from config import OLLAMA_GENERATION_MAX_TOKENS, PROJECT_ROOT
from experiment_runner import format_experiment_report, save_experiment_result
from indexing import build_experiment_index, close_cached_indices, release_embed_gpu
from llm import call_ollama
from pipeline import average, find_source_rank, load_questions
from prompts import (
    GENERATION_RESPONSE_SCHEMA,
    build_generation_prompt,
    has_answer_block,
    parse_generation_response,
    uses_json_output,
)
from retrievers import Retriever
from scoring import score_answer

from ablation.configs import (
    ABLATIONS,
    BASELINE,
    BASELINE_FOLDER,
    BASELINE_REFERENCE_SCORE,
    get_ablation,
    merged_settings,
    to_experiment_config,
)
from ablation.stats import (
    build_ablation_summary,
    extract_scores,
    format_summary_table,
    load_summary_json,
    rebuild_summary_from_disk,
    write_summary_json,
)

ABLATION_RESULTS_ROOT = (
    PROJECT_ROOT / "experiments" / "Results" / "Test Runs and Ablations"
)


def ablation_output_dir(folder_name):
    return ABLATION_RESULTS_ROOT / folder_name


def condition_run_dir(folder_name, condition_name, run_number, flat=False):
    base = ablation_output_dir(folder_name)
    if flat:
        return base / f"run_{run_number}"
    return base / condition_name / f"run_{run_number}"


def ensure_index(config, index_cache, show_progress=False):
    key = config.index_key()
    if key not in index_cache:
        print(f"Building index for chunk={config.chunk_size}, embedding={config.embedding_model}...")
        close_cached_indices(index_cache)
        index_cache[key] = build_experiment_index(config, show_progress=show_progress)[0]
    else:
        print("Reusing cached index for this chunk/embedding setting.")
    return index_cache[key]


def write_ablation_mirror(folder_name, condition_name, run_number, run_dir, summary, flat=False):
    dest = condition_run_dir(folder_name, condition_name, run_number, flat=flat)
    dest.mkdir(parents=True, exist_ok=True)

    report_src = run_dir / "REPORT.txt"
    if report_src.exists():
        shutil.copy2(report_src, dest / "REPORT.txt")

    scores = extract_scores(summary)
    (dest / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    data_src = run_dir / "data.json"
    if data_src.exists():
        shutil.copy2(data_src, dest / "data.json")

    questions_src = run_dir / "questions.jsonl"
    if questions_src.exists():
        shutil.copy2(questions_src, dest / "questions.jsonl")


def _checkpoint_path(run_dir):
    return run_dir / "checkpoint.jsonl"


def _load_checkpoint(run_dir):
    path = _checkpoint_path(run_dir)
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                completed[row["id"]] = row
    return completed


def _append_checkpoint(run_dir, row):
    run_dir.mkdir(parents=True, exist_ok=True)
    with _checkpoint_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _init_retriever(index, config):
    """Keep reranker on CPU so it does not contend with Ollama on the GPU."""
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


def _build_run_summary(config, per_question, index):
    latencies = [
        row.get("total_latency_s", row.get("retrieve_latency_s", 0.0))
        for row in per_question
    ]
    prompt_token_estimates = [
        len(row.get("raw_answer", "").split()) for row in per_question if "raw_answer" in row
    ]

    summary = {
        "config": config.to_dict(),
        "index_stats": {"chunks": len(index.chunks)},
        "question_count": len(per_question),
        "recall_at_k": average([1.0 if row["recall_hit"] else 0.0 for row in per_question]),
        "mrr_at_k": average([
            1.0 / row["found_rank"] if row["found_rank"] else 0.0
            for row in per_question
        ]),
        "avg_latency_s": round(average(latencies), 3),
    }
    summary.update({
        "final_score": round(average([row["metrics"]["final_score"] for row in per_question]), 2),
        "answer_correctness": round(average([row["metrics"]["answer_correctness"] for row in per_question]), 2),
        "faithfulness": round(average([row["metrics"]["faithfulness"] for row in per_question]), 2),
        "context_recall": round(average([row["metrics"]["context_recall"] for row in per_question]), 2),
        "context_precision": round(average([row["metrics"]["context_precision"] for row in per_question]), 2),
        "citation_accuracy": round(average([row["metrics"]["citation_accuracy"] for row in per_question]), 2),
        "answer_parse_rate": round(average([1.0 if row.get("answer_parsed") else 0.0 for row in per_question]), 3),
        "avg_prompt_tokens_est": round(average(prompt_token_estimates), 0),
    })
    return summary


def prepare_run_dir(folder_name, condition_name, run_number, force=False, flat=False):
    dest = condition_run_dir(folder_name, condition_name, run_number, flat=flat)
    if not dest.exists():
        return "fresh"

    if force:
        shutil.rmtree(dest)
        return "fresh"

    if (dest / "scores.json").exists():
        print(f"[skipped] run_{run_number} already complete")
        return "skip"

    checkpoint = _checkpoint_path(dest)
    if checkpoint.exists():
        completed = _load_checkpoint(dest)
        print(f"[resuming] run_{run_number} from checkpoint ({len(completed)} questions)")
        return "resume"

    shutil.rmtree(dest)
    return "fresh"


def run_single_pipeline(config, questions, index_cache, run_dir, show_progress=False):
    completed = _load_checkpoint(run_dir)
    pending = [q for q in questions if q["id"] not in completed]
    total = len(questions)

    if completed:
        print(f"Questions: {len(completed)}/{total} complete, {len(pending)} remaining")

    index = ensure_index(config, index_cache, show_progress=show_progress)
    release_embed_gpu(index)
    retriever = _init_retriever(index, config)
    json_output = uses_json_output(config.prompt)

    for item in pending:
        row = _evaluate_one_question(item, retriever, config, json_output)
        _append_checkpoint(run_dir, row)
        done = len(_load_checkpoint(run_dir))
        score = row["metrics"]["final_score"]
        print(f"[{config.name}] question {done}/{total} | score: {score}", flush=True)

    completed = _load_checkpoint(run_dir)
    per_question = [completed[q["id"]] for q in questions if q["id"] in completed]
    if len(per_question) != total:
        raise RuntimeError(
            f"Run incomplete: {len(per_question)}/{total} questions in checkpoint at {run_dir}"
        )

    summary = _build_run_summary(config, per_question, index)
    checkpoint = _checkpoint_path(run_dir)
    if checkpoint.exists():
        checkpoint.unlink()

    return {"summary": summary, "questions": per_question, "index": index}


def save_and_mirror(config, payload, folder_name, condition_name, run_number, flat=False):
    run_dir, _master_log = save_experiment_result(
        config,
        {
            "summary": payload["summary"],
            "questions": payload["questions"],
        },
        run_mode="full_pipeline",
        run_kind="ablation",
        report_builder=format_experiment_report,
    )
    write_ablation_mirror(
        folder_name,
        condition_name,
        run_number,
        run_dir,
        payload["summary"],
        flat=flat,
    )
    return run_dir, payload["summary"]


def run_condition(
    folder_name,
    condition_name,
    overrides,
    runs,
    questions,
    index_cache,
    round_name,
    force=False,
    show_progress=False,
    flat=False,
):
    settings = merged_settings(overrides)
    final_scores = []

    for run_number in range(1, runs + 1):
        run_dir = condition_run_dir(folder_name, condition_name, run_number, flat=flat)
        status = prepare_run_dir(folder_name, condition_name, run_number, force=force, flat=flat)
        if status == "skip":
            scores_path = run_dir / "scores.json"
            final_scores.append(float(json.loads(scores_path.read_text(encoding="utf-8"))["final_score"]))
            continue

        config = to_experiment_config(
            settings,
            name=f"ablation_{round_name}_{condition_name}_r{run_number}",
            round_name=round_name,
            description=f"Ablation condition {condition_name} run {run_number}",
        )

        print()
        print("-" * 72)
        label = folder_name if flat else condition_name
        print(f"Condition: {label} | Run {run_number}/{runs}")
        print(f"Override: {overrides or '(none — locked baseline)'}")
        print("-" * 72)

        payload = run_single_pipeline(
            config,
            questions,
            index_cache,
            run_dir,
            show_progress=show_progress,
        )
        run_dir_saved, summary = save_and_mirror(
            config,
            payload,
            folder_name,
            condition_name,
            run_number,
            flat=flat,
        )

        score = summary["final_score"]
        final_scores.append(float(score))
        print(f"Saved: {run_dir_saved}")
        print(f"Mirrored: {run_dir}")
        print(f"Final score: {score}")

    return final_scores


def run_ablation(
    ablation_id,
    runs=3,
    condition_filter=None,
    force=False,
    show_progress=False,
):
    ablation = get_ablation(ablation_id)
    questions = load_questions()
    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    conditions = ablation.conditions
    if condition_filter:
        conditions = [c for c in conditions if c.name == condition_filter]
        if not conditions:
            raise KeyError(
                f"Unknown condition '{condition_filter}' for ablation {ablation.id}. "
                f"Valid: {[c.name for c in ablation.conditions]}"
            )

    print("=" * 72)
    print(f"{ablation.id}: {ablation.folder}")
    print(f"Question: {ablation.question}")
    print(f"Runs per condition: {runs}")
    print("=" * 72)

    index_cache = {}
    condition_results = {}

    for condition in conditions:
        scores = run_condition(
            folder_name=ablation.folder,
            condition_name=condition.name,
            overrides=condition.overrides,
            runs=runs,
            questions=questions,
            index_cache=index_cache,
            round_name=f"ablation_{ablation.id.lower()}",
            force=force,
            show_progress=show_progress,
        )
        condition_results[condition.name] = scores

    close_cached_indices(index_cache)

    summary = build_ablation_summary(ablation.folder, condition_results)
    summary_path = ablation_output_dir(ablation.folder) / "summary.json"
    write_summary_json(summary_path, summary)
    print()
    print(format_summary_table(summary))
    print(f"\nSummary written: {summary_path}")
    return summary


def run_locked_baseline(runs=3, force=False, show_progress=False):
    questions = load_questions()
    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    print("=" * 72)
    print("Locked Baseline (run 007 config)")
    print(f"Runs: {runs}")
    print("=" * 72)

    index_cache = {}
    final_scores = run_condition(
        folder_name=BASELINE_FOLDER,
        condition_name="baseline",
        overrides={},
        runs=runs,
        questions=questions,
        index_cache=index_cache,
        round_name="ablation_baseline",
        force=force,
        show_progress=show_progress,
        flat=True,
    )
    close_cached_indices(index_cache)

    summary = build_ablation_summary(
        BASELINE_FOLDER,
        {"baseline": final_scores},
        baseline_score=BASELINE_REFERENCE_SCORE,
    )
    summary_path = ablation_output_dir(BASELINE_FOLDER) / "summary.json"
    write_summary_json(summary_path, summary)
    print()
    print(format_summary_table(summary))
    print(f"\nSummary written: {summary_path}")
    return summary


def run_all_ablations(runs=3, force=False, show_progress=False):
    run_locked_baseline(runs=runs, force=force, show_progress=show_progress)
    for ablation in ABLATIONS:
        run_ablation(
            ablation.id,
            runs=runs,
            force=force,
            show_progress=show_progress,
        )


def print_ablation_summary(ablation_id):
    ablation = get_ablation(ablation_id)
    summary_path = ablation_output_dir(ablation.folder) / "summary.json"

    if summary_path.exists():
        summary = load_summary_json(summary_path)
    else:
        summary = rebuild_summary_from_disk(ablation_output_dir(ablation.folder), ablation.folder)
        if not summary["conditions"]:
            raise FileNotFoundError(
                f"No completed runs found for {ablation.id} in {ablation_output_dir(ablation.folder)}"
            )
        write_summary_json(summary_path, summary)

    print(format_summary_table(summary))
    return summary
