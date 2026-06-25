#!/usr/bin/env python3
"""CLI entry point for ablation testing."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from ablation.configs import ABLATIONS, BASELINE, DEFAULT_RUNS_PER_CONDITION  # noqa: E402
from ablation.runner import (  # noqa: E402
    print_ablation_summary,
    run_ablation,
    run_all_ablations,
    run_locked_baseline,
)


def list_ablations():
    print("Locked baseline settings:")
    for key, value in BASELINE.items():
        print(f"  {key}: {value}")
    print()
    print("Ablations:")
    for ablation in ABLATIONS:
        print(f"\n  {ablation.id}: {ablation.folder}")
        print(f"      {ablation.question}")
        for condition in ablation.conditions:
            override_key = next(iter(condition.overrides))
            override_val = condition.overrides[override_key]
            print(f"      - {condition.name}: {override_key}={override_val!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation tests against the locked RAG baseline (run 007 config)."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the locked baseline config N times (default N=3)",
    )
    parser.add_argument(
        "--ablation",
        metavar="ID",
        help="Run ablation by ID (A1 through A9)",
    )
    parser.add_argument(
        "--condition",
        metavar="NAME",
        help="Run only one condition within an ablation",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS_PER_CONDITION,
        help=f"Number of runs per condition (default: {DEFAULT_RUNS_PER_CONDITION})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run locked baseline then all ablations sequentially",
    )
    parser.add_argument(
        "--summary",
        metavar="ID",
        help="Print mean ± std summary table for a completed ablation (A1–A9)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List ablation definitions and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ablation run folders if they exist",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show embedding progress bars while rebuilding indexes",
    )
    args = parser.parse_args()

    if args.list:
        list_ablations()
        return

    if args.summary:
        print_ablation_summary(args.summary)
        return

    if args.all:
        run_all_ablations(runs=args.runs, force=args.force, show_progress=args.progress)
        return

    if args.baseline:
        run_locked_baseline(runs=args.runs, force=args.force, show_progress=args.progress)
        return

    if args.ablation:
        run_ablation(
            args.ablation,
            runs=args.runs,
            condition_filter=args.condition,
            force=args.force,
            show_progress=args.progress,
        )
        return

    parser.print_help()
    print("\nQuick start:")
    print("  python run_ablation.py --list")
    print("  python run_ablation.py --baseline")
    print("  python run_ablation.py --ablation A1")
    print("  python run_ablation.py --ablation A1 --condition dense")
    print("  python run_ablation.py --all")
    print("  python run_ablation.py --summary A1")
    print("  python run_ablation_resilient.py --all --repair-venv --until-complete")
    print("  ./start_overnight_ablations.sh")


if __name__ == "__main__":
    main()
