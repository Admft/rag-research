import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import MASTER_LOG, RESULT_TIMEZONE, RESULTS_ROOT, RUNS_DIR, SUMMARY_CSV

SUMMARY_COLUMNS = [
    "run_folder",
    "run_name",
    "run_kind",
    "run_mode",
    "round",
    "run_time_central",
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
    "question_count",
]

ROUND_LABELS = {
    "og_baseline": "Original baseline (pre-grid)",
    "baseline": "Baseline test",
    "chunk_size": "Chunk size round",
    "overlap": "Overlap round",
    "top_k": "Top-k round",
    "retriever": "Retrieval method round",
    "embedding": "Embedding model round",
    "reranker": "Reranker round",
    "query_transform": "Query transform round",
    "context_filter": "Context filter round",
    "final": "Final combined test",
}

KIND_LABELS = {
    "og_baseline": "Original baseline run",
    "experiment": "Experiment grid run",
    "build_index": "Index build",
    "evaluate_retrieval": "Retrieval evaluation",
    "generation": "Single question generation",
}

MODE_LABELS = {
    "retrieval_only": "Retrieval only",
    "full_pipeline": "Full pipeline (retrieve + generate + score)",
}


def get_run_times():
    utc_now = datetime.now(timezone.utc)
    central_dt = utc_now.astimezone(RESULT_TIMEZONE)
    return utc_now, central_dt


def next_run_number():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in RUNS_DIR.iterdir():
        if path.is_dir() and len(path.name) >= 3 and path.name[:3].isdigit():
            numbers.append(int(path.name[:3]))
    return max(numbers, default=0) + 1


def make_run_folder(run_name, central_dt, run_number=None):
    if run_number is None:
        run_number = next_run_number()
    stamp = central_dt.strftime("%Y-%m-%d_%H-%M-%S_%Z")
    safe_name = run_name.replace("/", "-")
    folder_name = f"{run_number:03d}__{safe_name}__{stamp}"
    run_dir = RUNS_DIR / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, folder_name, run_number


def write_questions_jsonl(run_dir, questions):
    if not questions:
        return
    path = run_dir / "questions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in questions:
            f.write(json.dumps(item) + "\n")


def append_summary_row(row):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_run_folder(
    run_name,
    run_kind,
    data,
    report_builder,
    summary_row=None,
    questions=None,
    run_number=None,
):
    utc_now, central_dt = get_run_times()
    run_dir, folder_name, run_number = make_run_folder(run_name, central_dt, run_number=run_number)
    run_time_central = central_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    payload = {
        "run_folder": folder_name,
        "run_number": run_number,
        "run_name": run_name,
        "run_kind": run_kind,
        "run_time_utc": utc_now.isoformat(),
        "run_time_central": run_time_central,
        "run_timezone": "America/Chicago",
        **data,
    }

    report_text = report_builder(payload)

    (run_dir / "data.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "REPORT.txt").write_text(report_text.rstrip() + "\n", encoding="utf-8")
    write_questions_jsonl(run_dir, questions)

    if summary_row is not None:
        append_summary_row({
            "run_folder": folder_name,
            "run_name": run_name,
            "run_kind": run_kind,
            "run_time_central": run_time_central,
            **summary_row,
        })

    master_log = regenerate_master_log()
    return run_dir, master_log


def load_summary_rows():
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_latest_build_run():
    rows = [r for r in load_summary_rows() if r.get("run_kind") == "build_index"]
    if not rows:
        return None

    latest = max(rows, key=lambda r: r.get("run_time_central", ""))
    data_path = RUNS_DIR / latest["run_folder"] / "data.json"
    if not data_path.exists():
        return None

    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_mode(row):
    if row.get("run_mode"):
        return row["run_mode"]
    if row.get("final_score") not in ("", None):
        return "full_pipeline"
    if row.get("run_kind") in {"build_index", "generation"}:
        return ""
    return "retrieval_only"


def fmt(value, suffix=""):
    if value in ("", None):
        return "—"
    try:
        number = float(value)
        return f"{number:.2f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def primary_metric(row):
    mode = infer_mode(row)
    if mode == "full_pipeline" and row.get("final_score") not in ("", None):
        return float(row["final_score"])
    if row.get("recall_at_k") not in ("", None):
        return float(row["recall_at_k"])
    return None


def metric_label(row):
    if infer_mode(row) == "full_pipeline" and row.get("final_score") not in ("", None):
        return "Final score"
    top_k = row.get("top_k") or "?"
    return f"Recall@{top_k}"


def regenerate_master_log():
    rows = load_summary_rows()
    now = datetime.now(RESULT_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    sorted_rows = sorted(rows, key=lambda r: r.get("run_time_central", ""), reverse=True)

    lines = [
        "=" * 72,
        "RAG RESEARCH — MASTER LOG",
        "All times: Central Time (America/Chicago)",
        f"Last updated: {now}",
        "=" * 72,
        "",
        "WHERE THINGS LIVE",
        "-" * 40,
        "  experiments/Results/MASTER_LOG.txt  ← you are here (compare all runs)",
        "  experiments/Results/summary.csv     ← spreadsheet-friendly table",
        "  experiments/Results/runs/         ← one folder per run",
        "      001__baseline__2026-06-22_.../",
        "          REPORT.txt   ← read this first (human-friendly)",
        "          data.json    ← full raw data",
        "          questions.jsonl  ← per-question details (if applicable)",
        "",
        "HOW TO READ SCORES",
        "-" * 40,
        "  Retrieval only  → higher Recall@k and MRR@k is better",
        "  Full pipeline   → higher final_score is better (0-100)",
        "",
    ]

    if not sorted_rows:
        lines.extend([
            "No runs yet. Start with:",
            "  python src/run_experiments.py --run baseline --retrieval-only",
            "",
        ])
    else:
        experiment_rows = [r for r in sorted_rows if r.get("run_kind") == "experiment"]
        if experiment_rows:
            by_round = defaultdict(list)
            for row in experiment_rows:
                by_round[row.get("round", "unknown")].append(row)

            lines.extend([
                "EXPERIMENT LEADERBOARDS",
                "-" * 40,
            ])

            for round_name in sorted(
                by_round.keys(),
                key=lambda r: list(ROUND_LABELS.keys()).index(r) if r in ROUND_LABELS else 99,
            ):
                round_label = ROUND_LABELS.get(round_name, round_name)
                for mode_key, mode_label in (
                    ("retrieval_only", "Retrieval only"),
                    ("full_pipeline", "Full pipeline"),
                ):
                    mode_rows = [r for r in by_round[round_name] if infer_mode(r) == mode_key]
                    if not mode_rows:
                        continue

                    latest = {}
                    for row in mode_rows:
                        name = row["run_name"]
                        if name not in latest or row["run_time_central"] > latest[name]["run_time_central"]:
                            latest[name] = row

                    ranked = sorted(
                        latest.values(),
                        key=lambda r: primary_metric(r) or 0,
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
                            f"  {i}. {row['run_folder']} — "
                            f"{metric_label(row)} {metric_str} | "
                            f"chunk {row.get('chunk_size', '—')} | "
                            f"overlap {row.get('overlap', '—')} | "
                            f"top_k {row.get('top_k', '—')} | "
                            f"{row.get('retriever', '—')}"
                        )

        lines.extend([
            "",
            "=" * 72,
            "ALL RUNS (newest first)",
            "=" * 72,
            "",
        ])

        for i, row in enumerate(sorted_rows, start=1):
            kind = KIND_LABELS.get(row.get("run_kind", ""), row.get("run_kind", ""))
            mode = MODE_LABELS.get(infer_mode(row), infer_mode(row) or "n/a")
            round_label = ROUND_LABELS.get(row.get("round", ""), row.get("round", "") or "—")

            lines.extend([
                f"#{i}  {row['run_folder']}",
                f"     Name:   {row.get('run_name', '—')}",
                f"     Type:   {kind}",
                f"     Round:  {round_label}",
                f"     Mode:   {mode}",
                f"     Ran:    {row.get('run_time_central', '—')}",
            ])

            if row.get("chunk_size"):
                lines.append(
                    f"     Settings: chunk {row.get('chunk_size')} | overlap {row.get('overlap')} | "
                    f"top_k {row.get('top_k')} | {row.get('retriever', '—')} | "
                    f"reranker {row.get('reranker', 'none')}"
                )

            if row.get("final_score") not in ("", None):
                lines.append(
                    f"     Scores: final {fmt(row.get('final_score'))} | "
                    f"correctness {fmt(row.get('answer_correctness'))} | "
                    f"faithfulness {fmt(row.get('faithfulness'))} | "
                    f"context recall {fmt(row.get('context_recall'))}"
                )

            if row.get("recall_at_k") not in ("", None):
                lines.append(
                    f"     Retrieval: Recall@{row.get('top_k', '?')} {fmt(row.get('recall_at_k'), '%')} | "
                    f"MRR@{row.get('top_k', '?')} {fmt(row.get('mrr_at_k'), '%')}"
                )

            lines.append(f"     Folder: experiments/Results/runs/{row['run_folder']}/")
            lines.append("")

    lines.append("End of master log.")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MASTER_LOG
