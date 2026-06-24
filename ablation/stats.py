"""Aggregate ablation run scores into summary.json and printable tables."""

import json
import statistics
from pathlib import Path

from ablation.configs import BASELINE_REFERENCE_SCORE

SCORE_KEYS = [
    "final_score",
    "answer_correctness",
    "faithfulness",
    "context_recall",
    "context_precision",
    "citation_accuracy",
    "answer_parse_rate",
    "recall_at_k",
    "mrr_at_k",
    "avg_latency_s",
]


def extract_scores(summary):
    scores = {}
    for key in SCORE_KEYS:
        if key in summary:
            value = summary[key]
            if key in {"answer_parse_rate", "recall_at_k", "mrr_at_k"}:
                scores[key] = round(float(value) * 100, 2) if value <= 1 else round(float(value), 2)
            else:
                scores[key] = value
    return scores


def load_scores_from_run_dir(run_dir):
    scores_path = run_dir / "scores.json"
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["final_score"])

    data_path = run_dir / "data.json"
    if data_path.exists():
        with data_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload["summary"]["final_score"])

    raise FileNotFoundError(f"No scores found in {run_dir}")


def compute_condition_stats(final_scores):
    mean = statistics.mean(final_scores)
    std = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
    return {
        "runs": [round(s, 2) for s in final_scores],
        "mean": round(mean, 2),
        "std": round(std, 2),
        "delta_vs_baseline": round(mean - BASELINE_REFERENCE_SCORE, 2),
    }


def build_ablation_summary(ablation_name, condition_results, baseline_score=None):
    baseline_score = BASELINE_REFERENCE_SCORE if baseline_score is None else baseline_score
    conditions = {}
    for condition_name, final_scores in condition_results.items():
        stats = compute_condition_stats(final_scores)
        stats["delta_vs_baseline"] = round(stats["mean"] - baseline_score, 2)
        conditions[condition_name] = stats

    return {
        "ablation": ablation_name,
        "baseline_score": baseline_score,
        "conditions": conditions,
    }


def write_summary_json(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def load_summary_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_condition_scores(ablation_root, condition_name):
    condition_dir = ablation_root / condition_name
    if not condition_dir.exists():
        return []

    scores = []
    for run_dir in sorted(condition_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        try:
            scores.append(load_scores_from_run_dir(run_dir))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
    return scores


def collect_flat_run_scores(ablation_root):
    scores = []
    for run_dir in sorted(ablation_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        try:
            scores.append(load_scores_from_run_dir(run_dir))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
    return scores


def rebuild_summary_from_disk(ablation_folder, ablation_name):
    flat_scores = collect_flat_run_scores(ablation_folder)
    if flat_scores:
        return build_ablation_summary(ablation_name, {"baseline": flat_scores})

    condition_results = {}
    for condition_dir in sorted(p for p in ablation_folder.iterdir() if p.is_dir()):
        run_scores = collect_condition_scores(ablation_folder, condition_dir.name)
        if run_scores:
            condition_results[condition_dir.name] = run_scores
    return build_ablation_summary(ablation_name, condition_results)


def format_summary_table(summary):
    lines = [
        f"Ablation: {summary['ablation']}",
        f"Baseline reference score: {summary['baseline_score']}",
        "",
        f"{'Condition':<22} {'Mean ± Std':<18} {'Δ vs baseline':>14}",
        "-" * 56,
    ]
    for name, stats in summary["conditions"].items():
        mean_std = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
        delta = stats["delta_vs_baseline"]
        sign = "+" if delta > 0 else ""
        lines.append(f"{name:<22} {mean_std:<18} {sign}{delta:.2f}")
    return "\n".join(lines)
