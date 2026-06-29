#!/usr/bin/env python3
"""Compare top7_PRE_FIX re-scored runs vs fresh top7 runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ROOT / "experiments" / "Results" / "Test Runs and Ablations"
A8 = ABLATIONS / "Ablation 8 - Top-K"


def run_scores(condition: str) -> list[float]:
    scores = []
    cond_dir = A8 / condition
    if not cond_dir.exists():
        return scores
    for run_dir in sorted(cond_dir.glob("run_*")):
        path = run_dir / "scores.json"
        if path.exists():
            scores.append(float(json.loads(path.read_text())["final_score"]))
    return scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    pre = run_scores("top7_PRE_FIX")
    fresh = run_scores("top7")

    print("A8 top7: PRE_FIX (re-scored with fixed judge) vs fresh re-runs")
    print(f"  PRE_FIX runs: {pre}")
    print(f"  PRE_FIX mean: {mean(pre):.2f}")
    if fresh:
        print(f"  Fresh runs:   {fresh}")
        print(f"  Fresh mean:   {mean(fresh):.2f}")
        print(f"  Delta:        {mean(fresh) - mean(pre):+.2f}")
    else:
        print("  Fresh top7 runs not found yet — run ablations first, then re-run this script.")


if __name__ == "__main__":
    main()
