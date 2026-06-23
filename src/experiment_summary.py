import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT, RESULT_TIMEZONE

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_LOG_DIR = EXPERIMENTS_DIR / "Results"
SUMMARY_PATH = RESULTS_LOG_DIR / "EXPERIMENT_LOG.txt"
CSV_PATH = PROJECT_ROOT / "results" / "experiments" / "experiment_results.csv"

ROUND_LABELS = {
    "baseline": "Baseline test",
    "chunk_size": "Chunk size round",
    "overlap": "Overlap round",
    "top_k": "Top-k round",
    "retriever": "Retrieval method round",
    "reranker": "Reranker round",
    "final": "Final combined test",
}

MODE_LABELS = {
    "retrieval_only": "Retrieval only",
    "full_pipeline": "Full pipeline (retrieve + generate + score)",
}


def fmt(value, suffix=""):
    if value in ("", None):
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def infer_mode(row):
    if row.get("run_mode"):
        return row["run_mode"]
    if row.get("final_score") not in ("", None):
        return "full_pipeline"
    return "retrieval_only"


def parse_score(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_csv_rows():
    if not CSV_PATH.exists():
        return []

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def primary_metric(row):
    mode = infer_mode(row)
    if mode == "full_pipeline":
        return parse_score(row.get("final_score"))
    return parse_score(row.get("recall_at_k"))


def metric_label(row):
    if infer_mode(row) == "full_pipeline":
        return "Final score"
    return f"Recall@{row.get('top_k', '?')}"


def format_settings(row):
    return [
        f"  Chunk size: {row['chunk_size']} words    Overlap: {row['overlap']} words    Top-k: {row['top_k']}",
        f"  Retriever: {row['retriever']}    Embedding: {row['embedding_model']}",
        f"  Reranker: {row['reranker']}    Query transform: {row['query_transform']}",
        f"  Prompt: {row['prompt']}    Context filter: {row['context_filter']}",
    ]


def format_scores(row):
    mode = infer_mode(row)
    lines = []

    if mode == "full_pipeline":
        lines.extend([
            f"  Final score:          {fmt(row.get('final_score'))}",
            f"  Answer correctness:   {fmt(row.get('answer_correctness'))}",
            f"  Faithfulness:         {fmt(row.get('faithfulness'))}",
            f"  Context recall:       {fmt(row.get('context_recall'))}",
            f"  Context precision:    {fmt(row.get('context_precision'))}",
            f"  Citation accuracy:    {fmt(row.get('citation_accuracy'))}",
        ])

    lines.extend([
        f"  Recall@{row['top_k']}:           {fmt(row.get('recall_at_k'), '%')}",
        f"  MRR@{row['top_k']}:               {fmt(row.get('mrr_at_k'), '%')}",
        f"  Avg latency:          {fmt(row.get('avg_latency'))}s",
    ])

    if row.get("avg_prompt_tokens") not in ("", None):
        lines.append(f"  Avg prompt tokens:    {fmt(row.get('avg_prompt_tokens'))}")

    return lines


def format_run_block(index, row):
    mode = infer_mode(row)
    round_label = ROUND_LABELS.get(row.get("round", ""), row.get("round", "unknown"))
    mode_label = MODE_LABELS.get(mode, mode)

    lines = [
        f"#{index} — {row['run_id']} | {round_label} | {mode_label}",
        f"Ran: {row['run_time_central']} | Questions: {row.get('question_count', '?')}",
        "",
        "SETTINGS",
        *format_settings(row),
        "",
        "SCORES",
        *format_scores(row),
        "",
        "-" * 72,
        "",
    ]
    return lines


def find_winners(rows):
    """Best latest run per round (prefer 60-question runs)."""
    by_round = defaultdict(list)
    for row in rows:
        by_round[row.get("round", "unknown")].append(row)

    winners = []
    for round_name, round_rows in sorted(by_round.items()):
        round_rows.sort(key=lambda r: r.get("run_time_central", ""), reverse=True)

        # Prefer full eval set
        full_rows = [r for r in round_rows if str(r.get("question_count")) == "60"]
        pool = full_rows or round_rows

        # Latest run per run_id in this pool
        latest_by_id = {}
        for row in pool:
            run_id = row["run_id"]
            if run_id not in latest_by_id:
                latest_by_id[run_id] = row

        if not latest_by_id:
            continue

        best_row = max(
            latest_by_id.values(),
            key=lambda r: (primary_metric(r) is not None, primary_metric(r) or 0),
        )
        metric = primary_metric(best_row)
        if metric is None:
            continue

        winners.append({
            "round": round_name,
            "round_label": ROUND_LABELS.get(round_name, round_name),
            "run_id": best_row["run_id"],
            "metric_label": metric_label(best_row),
            "metric_value": metric,
            "mode": MODE_LABELS.get(infer_mode(best_row), infer_mode(best_row)),
            "settings": (
                f"chunk={best_row['chunk_size']}, overlap={best_row['overlap']}, "
                f"top_k={best_row['top_k']}, retriever={best_row['retriever']}"
            ),
            "run_time": best_row["run_time_central"],
        })

    return winners


def build_summary(rows):
    now = datetime.now(RESULT_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    sorted_rows = sorted(rows, key=lambda r: r.get("run_time_central", ""), reverse=True)

    lines = [
        "=" * 72,
        "RAG EXPERIMENT LOG",
        f"All times in Central Time (America/Chicago)",
        f"Last updated: {now}",
        "=" * 72,
        "",
        "HOW TO READ THIS FILE",
        "-" * 40,
        "  • Each run changes ONE setting from baseline (unless noted).",
        "  • Retrieval only = Recall@k and MRR@k (fast, no Ollama generation).",
        "  • Full pipeline = final_score + sub-metrics (slower, uses Ollama).",
        "  • Higher final_score or Recall@k = better.",
        "  • Detailed per-question data: results/experiments/{run_id}/",
        "",
    ]

    winners = find_winners(rows)
    if winners:
        lines.extend([
            "BEST IN EACH ROUND (latest run on full question set when available)",
            "-" * 40,
        ])
        for w in winners:
            lines.append(
                f"  {w['round_label']}: {w['run_id']} — "
                f"{w['metric_label']} {w['metric_value']:.2f}"
                + ("%" if w['metric_label'].startswith("Recall") else "")
            )
            lines.append(f"    Mode: {w['mode']}")
            lines.append(f"    Settings: {w['settings']}")
            lines.append(f"    Ran: {w['run_time']}")
            lines.append("")

    # Leaderboard tables by round
    by_round = defaultdict(list)
    for row in sorted_rows:
        by_round[row.get("round", "unknown")].append(row)

    lines.extend([
        "LEADERBOARDS BY ROUND (newest result per run, sorted by score)",
        "-" * 40,
    ])

    for round_name in sorted(by_round.keys(), key=lambda r: list(ROUND_LABELS.keys()).index(r) if r in ROUND_LABELS else 99):
        round_rows = by_round[round_name]
        round_label = ROUND_LABELS.get(round_name, round_name)

        for mode_key, mode_label in (
            ("retrieval_only", "Retrieval only"),
            ("full_pipeline", "Full pipeline"),
        ):
            mode_rows = [r for r in round_rows if infer_mode(r) == mode_key]
            if not mode_rows:
                continue

            latest = {}
            for row in mode_rows:
                run_id = row["run_id"]
                existing = latest.get(run_id)
                if existing is None:
                    latest[run_id] = row
                elif str(row.get("question_count")) == "60" and str(existing.get("question_count")) != "60":
                    latest[run_id] = row
                elif row.get("run_time_central", "") > existing.get("run_time_central", ""):
                    latest[run_id] = row

            ranked = sorted(
                latest.values(),
                key=lambda r: (primary_metric(r) is not None, primary_metric(r) or 0),
                reverse=True,
            )

            lines.append("")
            lines.append(f"[{round_label} — {mode_label}]")
            for i, row in enumerate(ranked, start=1):
                metric = primary_metric(row)
                metric_str = f"{metric:.1f}" if metric is not None else "—"
                if mode_key == "retrieval_only":
                    metric_str += "%"
                lines.append(
                    f"  {i}. {row['run_id']} — "
                    f"{metric_label(row)} {metric_str} | "
                    f"chunk {row['chunk_size']} | overlap {row['overlap']} | "
                    f"top_k {row['top_k']} | {row['retriever']} | "
                    f"{row['run_time_central']}"
                )

    lines.extend([
        "",
        "=" * 72,
        "ALL RUNS (newest first)",
        "=" * 72,
        "",
    ])

    for i, row in enumerate(sorted_rows, start=1):
        lines.extend(format_run_block(i, row))

    lines.append("End of log.")
    return "\n".join(lines)


def regenerate_experiment_log():
    rows = load_csv_rows()
    RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    content = build_summary(rows)
    SUMMARY_PATH.write_text(content + "\n", encoding="utf-8")
    return SUMMARY_PATH
