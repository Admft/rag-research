"""Run ablation conditions via the existing experiment pipeline."""

import json
import shutil
from pathlib import Path

from config import PROJECT_ROOT
from experiment_runner import format_experiment_report, save_experiment_result
from indexing import build_experiment_index, close_cached_indices
from pipeline import load_questions, run_experiment

from ablation.configs import (
    ABLATIONS,
    BASELINE,
    BASELINE_FOLDER,
    BASELINE_REFERENCE_SCORE,
    get_ablation,
    merged_settings,
    to_experiment_config,
)
from ablation.stats import (
    build_ablation_summary,
    extract_scores,
    format_summary_table,
    load_summary_json,
    rebuild_summary_from_disk,
    write_summary_json,
)

ABLATION_RESULTS_ROOT = (
    PROJECT_ROOT / "experiments" / "Results" / "Test Runs and Ablations"
)


def ablation_output_dir(folder_name):
    return ABLATION_RESULTS_ROOT / folder_name


def condition_run_dir(folder_name, condition_name, run_number, flat=False):
    base = ablation_output_dir(folder_name)
    if flat:
        return base / f"run_{run_number}"
    return base / condition_name / f"run_{run_number}"


def ensure_index(config, index_cache, show_progress=False):
    key = config.index_key()
    if key not in index_cache:
        print(f"Building index for chunk={config.chunk_size}, embedding={config.embedding_model}...")
        close_cached_indices(index_cache)
        index_cache[key] = build_experiment_index(config, show_progress=show_progress)[0]
    else:
        print("Reusing cached index for this chunk/embedding setting.")
    return index_cache[key]


def write_ablation_mirror(folder_name, condition_name, run_number, run_dir, summary, flat=False):
    dest = condition_run_dir(folder_name, condition_name, run_number, flat=flat)
    dest.mkdir(parents=True, exist_ok=True)

    report_src = run_dir / "REPORT.txt"
    if report_src.exists():
        shutil.copy2(report_src, dest / "REPORT.txt")

    scores = extract_scores(summary)
    (dest / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    data_src = run_dir / "data.json"
    if data_src.exists():
        shutil.copy2(data_src, dest / "data.json")

    questions_src = run_dir / "questions.jsonl"
    if questions_src.exists():
        shutil.copy2(questions_src, dest / "questions.jsonl")


def prepare_run_dir(folder_name, condition_name, run_number, force=False, flat=False):
    dest = condition_run_dir(folder_name, condition_name, run_number, flat=flat)
    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Run folder already exists: {dest}\nUse --force to overwrite."
            )


def run_single_pipeline(config, questions, index_cache, show_progress=False):
    index = ensure_index(config, index_cache, show_progress=show_progress)
    payload = run_experiment(
        config=config,
        questions=questions,
        index=index,
        retrieval_only=False,
        show_progress=show_progress,
    )
    return payload


def save_and_mirror(config, payload, folder_name, condition_name, run_number, flat=False):
    run_dir, _master_log = save_experiment_result(
        config,
        {
            "summary": payload["summary"],
            "questions": payload["questions"],
        },
        run_mode="full_pipeline",
        run_kind="ablation",
        report_builder=format_experiment_report,
    )
    write_ablation_mirror(
        folder_name,
        condition_name,
        run_number,
        run_dir,
        payload["summary"],
        flat=flat,
    )
    return run_dir, payload["summary"]


def run_condition(
    folder_name,
    condition_name,
    overrides,
    runs,
    questions,
    index_cache,
    round_name,
    force=False,
    show_progress=False,
    flat=False,
):
    settings = merged_settings(overrides)
    final_scores = []

    for run_number in range(1, runs + 1):
        prepare_run_dir(folder_name, condition_name, run_number, force=force, flat=flat)
        config = to_experiment_config(
            settings,
            name=f"ablation_{round_name}_{condition_name}_r{run_number}",
            round_name=round_name,
            description=f"Ablation condition {condition_name} run {run_number}",
        )

        print()
        print("-" * 72)
        label = folder_name if flat else condition_name
        print(f"Condition: {label} | Run {run_number}/{runs}")
        print(f"Override: {overrides or '(none — locked baseline)'}")
        print("-" * 72)

        payload = run_single_pipeline(config, questions, index_cache, show_progress=show_progress)
        run_dir, summary = save_and_mirror(
            config,
            payload,
            folder_name,
            condition_name,
            run_number,
            flat=flat,
        )

        score = summary["final_score"]
        final_scores.append(float(score))
        print(f"Saved: {run_dir}")
        print(f"Mirrored: {condition_run_dir(folder_name, condition_name, run_number, flat=flat)}")
        print(f"Final score: {score}")

    return final_scores


def run_ablation(
    ablation_id,
    runs=3,
    condition_filter=None,
    force=False,
    show_progress=False,
):
    ablation = get_ablation(ablation_id)
    questions = load_questions()
    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    conditions = ablation.conditions
    if condition_filter:
        conditions = [c for c in conditions if c.name == condition_filter]
        if not conditions:
            raise KeyError(
                f"Unknown condition '{condition_filter}' for ablation {ablation.id}. "
                f"Valid: {[c.name for c in ablation.conditions]}"
            )

    print("=" * 72)
    print(f"{ablation.id}: {ablation.folder}")
    print(f"Question: {ablation.question}")
    print(f"Runs per condition: {runs}")
    print("=" * 72)

    index_cache = {}
    condition_results = {}

    for condition in conditions:
        scores = run_condition(
            folder_name=ablation.folder,
            condition_name=condition.name,
            overrides=condition.overrides,
            runs=runs,
            questions=questions,
            index_cache=index_cache,
            round_name=f"ablation_{ablation.id.lower()}",
            force=force,
            show_progress=show_progress,
        )
        condition_results[condition.name] = scores

    close_cached_indices(index_cache)

    summary = build_ablation_summary(ablation.folder, condition_results)
    summary_path = ablation_output_dir(ablation.folder) / "summary.json"
    write_summary_json(summary_path, summary)
    print()
    print(format_summary_table(summary))
    print(f"\nSummary written: {summary_path}")
    return summary


def run_locked_baseline(runs=3, force=False, show_progress=False):
    questions = load_questions()
    if not questions:
        raise RuntimeError("No eval questions found in data/eval/questions.jsonl")

    print("=" * 72)
    print("Locked Baseline (run 007 config)")
    print(f"Runs: {runs}")
    print("=" * 72)

    index_cache = {}
    final_scores = run_condition(
        folder_name=BASELINE_FOLDER,
        condition_name="baseline",
        overrides={},
        runs=runs,
        questions=questions,
        index_cache=index_cache,
        round_name="ablation_baseline",
        force=force,
        show_progress=show_progress,
        flat=True,
    )
    close_cached_indices(index_cache)

    summary = build_ablation_summary(
        BASELINE_FOLDER,
        {"baseline": final_scores},
        baseline_score=BASELINE_REFERENCE_SCORE,
    )
    summary_path = ablation_output_dir(BASELINE_FOLDER) / "summary.json"
    write_summary_json(summary_path, summary)
    print()
    print(format_summary_table(summary))
    print(f"\nSummary written: {summary_path}")
    return summary


def run_all_ablations(runs=3, force=False, show_progress=False):
    run_locked_baseline(runs=runs, force=force, show_progress=show_progress)
    for ablation in ABLATIONS:
        run_ablation(
            ablation.id,
            runs=runs,
            force=force,
            show_progress=show_progress,
        )


def print_ablation_summary(ablation_id):
    ablation = get_ablation(ablation_id)
    summary_path = ablation_output_dir(ablation.folder) / "summary.json"

    if summary_path.exists():
        summary = load_summary_json(summary_path)
    else:
        summary = rebuild_summary_from_disk(ablation_output_dir(ablation.folder), ablation.folder)
        if not summary["conditions"]:
            raise FileNotFoundError(
                f"No completed runs found for {ablation.id} in {ablation_output_dir(ablation.folder)}"
            )
        write_summary_json(summary_path, summary)

    print(format_summary_table(summary))
    return summary
