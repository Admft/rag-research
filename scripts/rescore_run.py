#!/usr/bin/env python3
"""Re-score stored run outputs with the current judge (no retrieval/generation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from ablation.stats import (  # noqa: E402
    extract_scores,
    rebuild_summary_from_disk,
    write_summary_json,
)
from experiment_runner import format_experiment_report  # noqa: E402
from pipeline import average  # noqa: E402
from scoring import score_answer  # noqa: E402

ABLATIONS_ROOT = ROOT / "experiments" / "Results" / "Test Runs and Ablations"


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_judge_model(run_dir: Path) -> str | None:
    data_path = run_dir / "data.json"
    if data_path.exists():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        return payload.get("summary", {}).get("config", {}).get("judge")

    config_path = run_dir / ".ablation_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8")).get("judge")
    return None


def rescore_run(run_dir: Path, *, show_progress: bool = False) -> dict:
    questions_path = run_dir / "questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"No questions.jsonl in {run_dir}")

    judge_model = load_judge_model(run_dir)
    rows = load_questions(questions_path)

    for index, row in enumerate(rows, start=1):
        if show_progress:
            print(f"  [{index}/{len(rows)}] {row['id']}", flush=True)

        answer = row.get("raw_answer") or row.get("answer", "")
        metrics = score_answer(
            question=row["question"],
            expected_answer=row.get("expected_answer", ""),
            expected_source=row.get("expected_source", ""),
            answer=answer,
            retrieved=row.get("retrieved", []),
            judge_model=judge_model,
        )
        row["metrics"] = metrics

    questions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = {}
    data_path = run_dir / "data.json"
    if data_path.exists():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
    else:
        summary = {"config": {"top_k": 5}}

    summary.update({
        "question_count": len(rows),
        "recall_at_k": average([1.0 if row.get("recall_hit") else 0.0 for row in rows]),
        "mrr_at_k": average([
            1.0 / row["found_rank"] if row.get("found_rank") else 0.0
            for row in rows
        ]),
        "final_score": round(average([row["metrics"]["final_score"] for row in rows]), 2),
        "answer_correctness": round(
            average([row["metrics"]["answer_correctness"] for row in rows]), 2
        ),
        "faithfulness": round(average([row["metrics"]["faithfulness"] for row in rows]), 2),
        "context_recall": round(average([row["metrics"]["context_recall"] for row in rows]), 2),
        "context_precision": round(
            average([row["metrics"]["context_precision"] for row in rows]), 2
        ),
        "citation_accuracy": round(
            average([row["metrics"]["citation_accuracy"] for row in rows]), 2
        ),
        "answer_parse_rate": round(
            average([1.0 if row.get("answer_parsed") else 0.0 for row in rows]), 3
        ),
    })

    if data_path.exists():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        payload["summary"] = summary
        payload["questions"] = rows
        data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report_payload = {
            "summary": summary,
            "questions": rows,
            "run_folder": payload.get("run_folder", run_dir.name),
            "run_mode": payload.get("run_mode", "full_pipeline"),
            "run_time_central": payload.get("run_time_central", ""),
        }
        (run_dir / "REPORT.txt").write_text(
            format_experiment_report(report_payload) + "\n",
            encoding="utf-8",
        )

    scores = extract_scores(summary)
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    print(f"Rescored {run_dir}: final_score={summary['final_score']}")
    return summary


def rescore_condition(ablation_folder: str, condition: str, *, show_progress: bool = False):
    condition_dir = ABLATIONS_ROOT / ablation_folder / condition
    if not condition_dir.exists():
        raise FileNotFoundError(condition_dir)

    for run_dir in sorted(condition_dir.glob("run_*")):
        if run_dir.is_dir():
            rescore_run(run_dir, show_progress=show_progress)

    summary = rebuild_summary_from_disk(
        ABLATIONS_ROOT / ablation_folder,
        ablation_folder,
    )
    write_summary_json(ABLATIONS_ROOT / ablation_folder / "summary.json", summary)
    print(f"Updated summary: {ABLATIONS_ROOT / ablation_folder / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Re-score stored ablation runs with fixed judge.")
    parser.add_argument("--run-dir", type=Path, help="Single run directory containing questions.jsonl")
    parser.add_argument("--ablation-folder", help="Ablation folder name under Test Runs and Ablations")
    parser.add_argument("--condition", help="Condition name within the ablation folder")
    parser.add_argument("--progress", action="store_true", help="Print per-question progress")
    args = parser.parse_args()

    if args.run_dir:
        rescore_run(args.run_dir.resolve(), show_progress=args.progress)
        return

    if args.ablation_folder and args.condition:
        rescore_condition(args.ablation_folder, args.condition, show_progress=args.progress)
        return

    parser.error("Provide --run-dir or both --ablation-folder and --condition")


if __name__ == "__main__":
    main()
