#!/usr/bin/env python3
"""Export BEIR eval question sets into each result folder.

Writes:
  eval_questions.jsonl — input question bank used for the run (from data/beir/)
  questions.jsonl      — per-question run results (from REPORT.txt if missing)

Usage:
  .venv/bin/python scripts/export_beir_questions.py
  .venv/bin/python scripts/export_beir_questions.py --dataset nfcorpus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beir import DATASETS, dataset_eval_path, dataset_result_path, normalize_dataset_name

QUESTION_BLOCK = re.compile(
    r"^\[(?P<id>[^\]]+)\]\s+(?P<question>.*?)\s*$\n"
    r"\s+Expected source:\s+(?P<expected_source>.+?)\s*$\n"
    r"\s+Recall hit:\s+(?P<recall_hit>yes|no)\s+\(rank\s+(?P<rank>[^)]+)\)\s*$\n"
    r"\s+Final score:\s+(?P<final_score>[-\d.]+)\s*$\n"
    r"\s+Answer:\s+(?P<answer>.*?)(?=\n\[|\n=+|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_report(report_path: Path) -> dict[str, dict]:
    text = report_path.read_text(encoding="utf-8")
    rows: dict[str, dict] = {}
    for match in QUESTION_BLOCK.finditer(text):
        rank_raw = match.group("rank").strip()
        found_rank = None if rank_raw.lower() == "none" else int(rank_raw)
        rows[match.group("id")] = {
            "id": match.group("id"),
            "question": match.group("question").strip(),
            "expected_source": match.group("expected_source").strip(),
            "found_rank": found_rank,
            "recall_hit": match.group("recall_hit") == "yes",
            "answer": match.group("answer").strip(),
            "metrics": {"final_score": float(match.group("final_score"))},
        }
    return rows


def export_dataset(key: str, force: bool = False) -> dict:
    key = normalize_dataset_name(key)
    result_dir = dataset_result_path(key)
    eval_source = dataset_eval_path(key)
    eval_dest = result_dir / "eval_questions.jsonl"
    questions_dest = result_dir / "questions.jsonl"

    if not eval_source.exists():
        raise FileNotFoundError(f"Missing eval bank: {eval_source}")

    if force or not eval_dest.exists():
        eval_dest.write_text(eval_source.read_text(encoding="utf-8"), encoding="utf-8")

    eval_rows = [json.loads(line) for line in eval_dest.read_text().splitlines() if line.strip()]
    report_path = result_dir / "REPORT.txt"
    parsed = parse_report(report_path) if report_path.exists() else {}

    if force or not questions_dest.exists():
        merged = []
        for item in eval_rows:
            qid = item["id"]
            row = {
                "id": qid,
                "question": item["question"],
                "question_type": item.get("question_type", "normal"),
                "expected_source": item["expected_source"],
                "expected_answer": item.get("answer", ""),
                "source_dataset": item.get("source_dataset", key),
            }
            if qid in parsed:
                row.update(parsed[qid])
            merged.append(row)
        with questions_dest.open("w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row) + "\n")

    return {
        "dataset": key,
        "result_dir": str(result_dir.relative_to(ROOT)),
        "eval_questions": len(eval_rows),
        "parsed_from_report": len(parsed),
        "eval_questions_file": str(eval_dest.relative_to(ROOT)),
        "questions_file": str(questions_dest.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BEIR questions into result folders")
    parser.add_argument("--dataset", help="One dataset key (nfcorpus, scifact, fiqa)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing exports")
    args = parser.parse_args()

    keys = [normalize_dataset_name(args.dataset)] if args.dataset else list(DATASETS)
    for key in keys:
        info = export_dataset(key, force=args.force)
        print(
            f"{info['dataset']}: {info['eval_questions']} eval questions, "
            f"{info['parsed_from_report']} parsed from REPORT"
        )
        print(f"  {info['eval_questions_file']}")
        print(f"  {info['questions_file']}")


if __name__ == "__main__":
    main()
