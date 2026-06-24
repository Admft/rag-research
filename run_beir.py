#!/usr/bin/env python3
"""CLI entry point for BEIR dataset download, indexing, and evaluation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from beir import (  # noqa: E402
    BEIR_DATA_ROOT,
    BEIR_INDEX_ROOT,
    DATASETS,
    dataset_data_path,
    dataset_eval_path,
    dataset_index_path,
    dataset_result_path,
    normalize_dataset_name,
)


def show_status():
    from beir.indexer import index_is_current

    print("BEIR dataset status")
    print("-" * 72)
    print(f"{'Dataset':<12} {'Downloaded':<12} {'Indexed':<10} {'Evaluated':<10} {'Questions'}")
    print("-" * 72)

    for key in DATASETS:
        data_path = dataset_data_path(key)
        downloaded = data_path.exists() and any(data_path.iterdir()) if data_path.exists() else False
        indexed = index_is_current(key)
        result_dir = dataset_result_path(key)
        evaluated = (result_dir / "scores.json").exists()
        question_count = "—"
        eval_path = dataset_eval_path(key)
        if eval_path.exists():
            question_count = str(sum(1 for line in eval_path.read_text().splitlines() if line.strip()))

        print(
            f"{key:<12} "
            f"{'yes' if downloaded else 'no':<12} "
            f"{'yes' if indexed else 'no':<10} "
            f"{'yes' if evaluated else 'no':<10} "
            f"{question_count}"
        )

    print()
    print(f"Data:    {BEIR_DATA_ROOT}")
    print(f"Indexes: {BEIR_INDEX_ROOT}")


def build_indexes(force=False, show_progress=False, max_queries=50):
    from beir.indexer import build_beir_index

    for key in DATASETS:
        build_beir_index(key, show_progress=show_progress, force=force, max_queries=max_queries)


def main():
    parser = argparse.ArgumentParser(
        description="Download, index, and evaluate BEIR datasets with the locked RAG baseline."
    )
    parser.add_argument("--download", action="store_true", help="Download datasets to data/beir/")
    parser.add_argument("--index", action="store_true", help="Build Qdrant indexes in data/beir_indexes/")
    parser.add_argument("--eval", metavar="NAME", help="Run evaluation on one dataset or 'all'")
    parser.add_argument("--setup", action="store_true", help="Download + index + eval all datasets")
    parser.add_argument("--status", action="store_true", help="Show download/index/eval status")
    parser.add_argument("--force", action="store_true", help="Re-download or re-index even if exists")
    parser.add_argument("--progress", action="store_true", help="Show embedding progress while indexing")
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Max eval queries per dataset (default: 50)",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not any([args.download, args.index, args.eval, args.setup]):
        parser.print_help()
        print("\nQuick start:")
        print("  python3 run_beir.py --download")
        print("  python3 run_beir.py --index")
        print("  python3 run_beir.py --eval nfcorpus")
        print("  python3 run_beir.py --setup")
        print("  python3 run_beir.py --status")
        return

    if args.download or args.setup:
        from beir.download import download_all

        download_all(force=args.force)

    if args.index or args.setup:
        build_indexes(force=args.force, show_progress=args.progress, max_queries=args.queries)

    if args.eval or args.setup:
        from beir.evaluator import evaluate_all, print_summary_table, run_beir_evaluation

        if args.setup or args.eval.lower() == "all":
            evaluate_all(show_progress=args.progress, force=args.force, max_queries=args.queries)
        else:
            result = run_beir_evaluation(
                normalize_dataset_name(args.eval),
                show_progress=args.progress,
                force=args.force,
                max_queries=args.queries,
            )
            print_summary_table([result])
            print(f"\nSaved: {result['result_dir']}")


if __name__ == "__main__":
    main()
