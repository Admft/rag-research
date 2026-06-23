import argparse
import sys

from experiment_config import build_experiment_configs, get_run_by_name, get_runs_for_round
from experiment_runner import run_grid
from pipeline import load_questions
from run_storage import MASTER_LOG


ROUND_ORDER = [
    "baseline",
    "chunk_size",
    "overlap",
    "top_k",
    "retriever",
    "reranker",
    "final",
]


def list_runs():
    print("Available experiment runs:\n")
    current_round = None
    for config in build_experiment_configs():
        if config.round != current_round:
            current_round = config.round
            print(f"\n[{current_round}]")
        print(f"  - {config.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG experiment grid (baseline + one-variable-at-a-time tests)."
    )
    parser.add_argument(
        "--run",
        help="Run a single experiment by name (e.g. baseline, chunk_512, topk_5)",
    )
    parser.add_argument(
        "--round",
        choices=ROUND_ORDER,
        help="Run all experiments in a round",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full recommended grid (25 runs)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configured runs",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and RAGAS-style scoring (retrieval metrics only)",
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

    if args.list:
        list_runs()
        return

    if args.run:
        configs = [get_run_by_name(args.run)]
    elif args.round:
        configs = get_runs_for_round(args.round)
    elif args.all:
        configs = build_experiment_configs()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python src/run_experiments.py --list")
        print("  python src/run_experiments.py --run baseline --retrieval-only")
        print("  python src/run_experiments.py --round chunk_size --retrieval-only")
        print("  python src/run_experiments.py --all")
        sys.exit(0)

    questions = load_questions()
    if args.max_questions:
        questions = questions[: args.max_questions]

    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    print(f"Running {len(configs)} experiment(s) on {len(questions)} question(s).")
    if args.retrieval_only:
        print("Mode: retrieval-only")
    else:
        print("Mode: full pipeline (retrieve + generate + score via Ollama)")

    run_grid(
        configs=configs,
        questions=questions,
        retrieval_only=args.retrieval_only,
        show_progress=args.progress,
    )

    print(f"\nMaster log: {MASTER_LOG}")


if __name__ == "__main__":
    main()
