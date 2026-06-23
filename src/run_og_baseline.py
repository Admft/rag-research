"""Re-run the original first-dev RAG baseline (saved as run 000).

Settings match the initial lab before the experiment grid:
  chunk 120 / overlap 30, dense retrieval, top_k 5, no reranker,
  BAAI/bge-small-en-v1.5, llama3.1:8b, strict "I do not know" prompt.

Uses a separate Qdrant store (data/qdrant_og) so it does not overwrite
the main experiment index in data/qdrant.
"""

import argparse
import sys

from config import (
    EMBEDDING_MODEL_NAME,
    OG_CHUNK_SIZE_WORDS,
    OG_OVERLAP_WORDS,
    OG_QDRANT_PATH,
    OLLAMA_MODEL,
)
from experiment_config import ExperimentConfig
from experiment_runner import format_experiment_report, save_experiment_result
from indexing import close_cached_indices
from pipeline import load_questions, run_experiment

OG_RUN_NUMBER = 0

OG_CHANGELOG = """
========================================================================
ORIGINAL BASELINE NOTES
========================================================================

This is run 000 — the recreated first-development baseline, before the
experiment grid (runs 001+). Settings intentionally stay vanilla so later
experiments have a simple reference point.

Chunking:     {chunk_size} words, {overlap} overlap (naive word-count split)
Embedding:    {embedding}
Vector DB:    Qdrant local file at data/qdrant_og (cosine similarity)
Retrieval:    Dense only, top_k={top_k}, no reranker
Generation:   {generator} via Ollama, og_strict prompt
              ("I do not know" if context is insufficient)
Evaluation:   Recall@k, MRR@k, plus full pipeline judge scores for
              comparison with runs 001–005

========================================================================
""".strip()


def og_baseline_config():
    return ExperimentConfig(
        name="og_baseline",
        chunk_size=OG_CHUNK_SIZE_WORDS,
        chunk_overlap=OG_OVERLAP_WORDS,
        retriever="dense",
        embedding_model=EMBEDDING_MODEL_NAME,
        top_k=5,
        reranker="none",
        query_transform="none",
        prompt="og_strict",
        context_filter="none",
        generator=OLLAMA_MODEL,
        round="og_baseline",
        description="Original first-dev baseline before experiment grid changes.",
    )


def format_og_baseline_report(payload):
    report = format_experiment_report(payload)
    config = payload["summary"]["config"]
    notes = OG_CHANGELOG.format(
        chunk_size=config["chunk_size"],
        overlap=config["chunk_overlap"],
        embedding=config["embedding_model"],
        top_k=config["top_k"],
        generator=config["generator"],
    )
    return report.rstrip() + "\n\n" + notes + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Run the original pre-grid RAG baseline (saved as run 000)."
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and judge scoring (retrieval metrics only)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Limit number of eval questions (useful for quick tests)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show embedding progress bars while indexing",
    )
    args = parser.parse_args()

    config = og_baseline_config()
    questions = load_questions()
    if args.max_questions:
        questions = questions[: args.max_questions]

    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    mode = "retrieval_only" if args.retrieval_only else "full_pipeline"
    print(f"OG baseline on {len(questions)} question(s). Mode: {mode}")
    print(
        f"Settings: chunk {config.chunk_size} | overlap {config.chunk_overlap} | "
        f"top_k {config.top_k} | dense | no reranker | prompt {config.prompt}"
    )
    print(f"Index path: {OG_QDRANT_PATH}")

    index_cache = {}
    try:
        print("\nBuilding OG index...")
        payload = run_experiment(
            config=config,
            questions=questions,
            retrieval_only=args.retrieval_only,
            show_progress=args.progress,
            qdrant_path=OG_QDRANT_PATH,
        )
        index_cache[config.index_key()] = payload["index"]

        run_dir, master_log = save_experiment_result(
            config,
            {
                "summary": payload["summary"],
                "questions": payload["questions"],
            },
            run_mode=mode,
            run_kind="og_baseline",
            run_number=OG_RUN_NUMBER,
            report_builder=format_og_baseline_report,
        )
    finally:
        close_cached_indices(index_cache)

    summary = payload["summary"]
    print(f"\nSaved run folder: {run_dir}")
    if "final_score" in summary:
        print(f"Final score: {summary['final_score']}")
    print(
        f"Recall@{config.top_k}: {summary['recall_at_k'] * 100:.1f}% | "
        f"MRR@{config.top_k}: {summary['mrr_at_k'] * 100:.1f}%"
    )
    print(f"Master log: {master_log}")


if __name__ == "__main__":
    main()
