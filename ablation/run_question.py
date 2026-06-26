#!/usr/bin/env python3
"""Evaluate one ablation question in an isolated process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from ablation.runner import _evaluate_one_question, _init_retriever  # noqa: E402
from ablation.stable import enable_stable_mode  # noqa: E402
from experiment_config import ExperimentConfig  # noqa: E402
from indexing import load_experiment_index, release_embed_gpu  # noqa: E402
from pipeline import load_questions  # noqa: E402
from prompts import uses_json_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ablation eval question.")
    parser.add_argument("--config", required=True, help="Path to JSON config (ExperimentConfig.to_dict())")
    parser.add_argument("--question-id", required=True, help="Question id from data/eval/questions.jsonl")
    args = parser.parse_args()

    enable_stable_mode()

    config = ExperimentConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    item = next((q for q in load_questions() if q["id"] == args.question_id), None)
    if item is None:
        raise SystemExit(f"Unknown question id: {args.question_id}")

    index = load_experiment_index(config)
    try:
        release_embed_gpu(index)
        retriever = _init_retriever(index, config)
        row = _evaluate_one_question(item, retriever, config, uses_json_output(config.prompt))
    finally:
        index.close()

    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
