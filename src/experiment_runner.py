import csv
import json

from config import RESULTS_DIR
from experiment_config import build_experiment_configs, get_run_by_name, get_runs_for_round
from indexing import build_experiment_index, close_cached_indices
from pipeline import load_questions, run_experiment
from experiment_summary import regenerate_experiment_log
from results import get_run_times

EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "experiments"
CSV_PATH = EXPERIMENT_RESULTS_DIR / "experiment_results.csv"

CSV_COLUMNS = [
    "run_id",
    "round",
    "chunk_size",
    "overlap",
    "top_k",
    "retriever",
    "embedding_model",
    "reranker",
    "query_transform",
    "prompt",
    "context_filter",
    "final_score",
    "answer_correctness",
    "faithfulness",
    "context_recall",
    "context_precision",
    "citation_accuracy",
    "recall_at_k",
    "mrr_at_k",
    "avg_latency",
    "avg_prompt_tokens",
    "question_count",
    "run_mode",
    "run_time_central",
    "json_path",
]


def summary_to_csv_row(summary, run_time_central, json_path, run_mode="full_pipeline"):
    config = summary["config"]
    return {
        "run_id": config["name"],
        "round": config.get("round", ""),
        "chunk_size": config["chunk_size"],
        "overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "retriever": config["retriever"],
        "embedding_model": config["embedding_model"],
        "reranker": config["reranker"],
        "query_transform": config["query_transform"],
        "prompt": config["prompt"],
        "context_filter": config["context_filter"],
        "final_score": summary.get("final_score", ""),
        "answer_correctness": summary.get("answer_correctness", ""),
        "faithfulness": summary.get("faithfulness", ""),
        "context_recall": summary.get("context_recall", ""),
        "context_precision": summary.get("context_precision", ""),
        "citation_accuracy": summary.get("citation_accuracy", ""),
        "recall_at_k": round(summary.get("recall_at_k", 0.0) * 100, 2),
        "mrr_at_k": round(summary.get("mrr_at_k", 0.0) * 100, 2),
        "avg_latency": summary.get("avg_latency_s", ""),
        "avg_prompt_tokens": summary.get("avg_prompt_tokens_est", ""),
        "question_count": summary.get("question_count", ""),
        "run_mode": run_mode,
        "run_time_central": run_time_central,
        "json_path": str(json_path),
    }


def append_csv_row(row):
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()

    with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_experiment_result(config, payload, run_mode="full_pipeline"):
    _, central_dt = get_run_times()
    run_time_central = central_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    run_dir = EXPERIMENT_RESULTS_DIR / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    basename = (
        f"{config.name}__chunk{config.chunk_size}-overlap{config.chunk_overlap}"
        f"__topk{config.top_k}__{config.retriever}"
        f"__{central_dt.strftime('%Y-%m-%d_%H-%M-%S_%Z')}"
    )

    json_path = run_dir / f"{basename}.json"
    txt_path = run_dir / f"{basename}.txt"

    output = {
        "run_name": config.name,
        "run_time_central": run_time_central,
        "run_timezone": "America/Chicago",
        **payload,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    txt_path.write_text(format_experiment_report(output), encoding="utf-8")

    latest_json = run_dir / "latest.json"
    latest_txt = run_dir / "latest.txt"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_txt.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

    csv_row = summary_to_csv_row(payload["summary"], run_time_central, json_path, run_mode=run_mode)
    append_csv_row(csv_row)
    log_path = regenerate_experiment_log()

    return json_path, txt_path, log_path


def format_experiment_report(output):
    summary = output["summary"]
    config = summary["config"]
    lines = [
        "=" * 72,
        f"EXPERIMENT: {config['name']}",
        "=" * 72,
        f"Round: {config.get('round', '')}",
        f"Run time: {output['run_time_central']}",
        "",
        "CONFIG",
        "-" * 40,
        f"Chunk size:       {config['chunk_size']}",
        f"Overlap:          {config['chunk_overlap']}",
        f"Top-k:            {config['top_k']}",
        f"Retriever:        {config['retriever']}",
        f"Embedding model:  {config['embedding_model']}",
        f"Reranker:         {config['reranker']}",
        f"Query transform:  {config['query_transform']}",
        f"Prompt:           {config['prompt']}",
        f"Context filter:   {config['context_filter']}",
        "",
        "SCORES",
        "-" * 40,
    ]

    if "final_score" in summary:
        lines.extend([
            f"Final score:          {summary['final_score']}",
            f"Answer correctness:   {summary['answer_correctness']}",
            f"Faithfulness:         {summary['faithfulness']}",
            f"Context recall:       {summary['context_recall']}",
            f"Context precision:    {summary['context_precision']}",
            f"Citation accuracy:    {summary['citation_accuracy']}",
        ])

    lines.extend([
        f"Recall@{config['top_k']}:            {summary['recall_at_k'] * 100:.1f}%",
        f"MRR@{config['top_k']}:                {summary['mrr_at_k'] * 100:.1f}%",
        f"Avg latency:          {summary['avg_latency_s']}s",
        "",
        "PER-QUESTION",
        "-" * 40,
    ])

    for item in output["questions"]:
        lines.append("")
        lines.append(f"[{item['id']}] {item['question']}")
        lines.append(f"  Expected source: {item['expected_source']}")
        lines.append(f"  Recall hit: {'yes' if item['recall_hit'] else 'no'} (rank {item['found_rank']})")
        if "answer" in item:
            lines.append(f"  Final score: {item['metrics']['final_score']}")
            lines.append(f"  Answer: {item['answer'][:300]}")

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def run_grid(configs, questions, retrieval_only=False, show_progress=False):
    index_cache = {}
    results = []

    for i, config in enumerate(configs, start=1):
        print()
        print("=" * 72)
        print(f"[{i}/{len(configs)}] {config.name} ({config.round})")
        print("=" * 72)

        key = config.index_key()
        if key not in index_cache:
            print("Building index...")
            close_cached_indices(index_cache)
            index_cache[key] = build_experiment_index(config, show_progress=show_progress)[0]
        else:
            print("Reusing cached index for this chunk/embedding setting.")

        payload = run_experiment(
            config=config,
            questions=questions,
            index=index_cache[key],
            retrieval_only=retrieval_only,
            show_progress=show_progress,
        )

        paths = save_experiment_result(
            config,
            {
                "summary": payload["summary"],
                "questions": payload["questions"],
            },
            run_mode="retrieval_only" if retrieval_only else "full_pipeline",
        )
        results.append(payload["summary"])

        print(f"Saved: {paths[0]}")
        print(f"Experiment log: {paths[2]}")
        if "final_score" in payload["summary"]:
            print(f"Final score: {payload['summary']['final_score']}")
        print(
            f"Recall@{config.top_k}: {payload['summary']['recall_at_k'] * 100:.1f}% | "
            f"MRR@{config.top_k}: {payload['summary']['mrr_at_k'] * 100:.1f}%"
        )

    print()
    print(f"CSV updated: {CSV_PATH}")
    log_path = regenerate_experiment_log()
    print(f"Human-readable log: {log_path}")
    close_cached_indices(index_cache)
    return results
