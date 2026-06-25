"""Overnight supervisor: completion tracking, stall detection, status files."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ablation.configs import (
    ABLATIONS,
    BASELINE_FOLDER,
    DEFAULT_RUNS_PER_CONDITION,
    get_ablation,
)
from ablation.runner import ABLATION_RESULTS_ROOT, condition_run_dir

STATUS_PATH = ABLATION_RESULTS_ROOT / "ablation_supervisor_status.json"


@dataclass(frozen=True)
class ExpectedRun:
    folder_name: str
    condition_name: str | None
    run_number: int
    flat: bool

    @property
    def run_dir(self) -> Path:
        return condition_run_dir(
            self.folder_name,
            self.condition_name or "baseline",
            self.run_number,
            flat=self.flat,
        )

    @property
    def label(self) -> str:
        if self.flat:
            return f"{self.folder_name}/run_{self.run_number}"
        return f"{self.folder_name}/{self.condition_name}/run_{self.run_number}"


def expected_runs(runs_per_condition: int = DEFAULT_RUNS_PER_CONDITION) -> list[ExpectedRun]:
    jobs: list[ExpectedRun] = []
    for run_number in range(1, runs_per_condition + 1):
        jobs.append(ExpectedRun(BASELINE_FOLDER, None, run_number, flat=True))
    for ablation in ABLATIONS:
        for condition in ablation.conditions:
            for run_number in range(1, runs_per_condition + 1):
                jobs.append(ExpectedRun(ablation.folder, condition.name, run_number, flat=False))
    return jobs


def run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "scores.json").exists()


def run_is_in_progress(run_dir: Path) -> bool:
    return (run_dir / "checkpoint.jsonl").exists() and not run_is_complete(run_dir)


def count_checkpoint_lines(run_dir: Path) -> int:
    path = run_dir / "checkpoint.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _runs_per_condition_from_argv(argv: list[str]) -> int:
    if "--runs" in argv:
        idx = argv.index("--runs")
        if idx + 1 < len(argv):
            return int(argv[idx + 1])
    return DEFAULT_RUNS_PER_CONDITION


def expected_runs_for_argv(argv: list[str]) -> list[ExpectedRun]:
    runs = _runs_per_condition_from_argv(argv)

    if "--all" in argv:
        return expected_runs(runs)

    jobs: list[ExpectedRun] = []

    if "--baseline" in argv:
        for run_number in range(1, runs + 1):
            jobs.append(ExpectedRun(BASELINE_FOLDER, None, run_number, flat=True))

    if "--ablation" in argv:
        idx = argv.index("--ablation")
        ablation = get_ablation(argv[idx + 1])
        condition_filter = None
        if "--condition" in argv:
            condition_filter = argv[argv.index("--condition") + 1]
        for condition in ablation.conditions:
            if condition_filter and condition.name != condition_filter:
                continue
            for run_number in range(1, runs + 1):
                jobs.append(ExpectedRun(ablation.folder, condition.name, run_number, flat=False))

    if jobs:
        return jobs

    return expected_runs(runs)


def scan_progress(jobs: list[ExpectedRun] | None = None) -> dict:
    jobs = jobs or expected_runs()
    complete = []
    in_progress = []
    pending = []

    for job in jobs:
        run_dir = job.run_dir
        if run_is_complete(run_dir):
            complete.append(job)
        elif run_is_in_progress(run_dir):
            in_progress.append({
                "label": job.label,
                "questions_done": count_checkpoint_lines(run_dir),
                "run_dir": str(run_dir),
            })
        else:
            pending.append(job.label)

    return {
        "total_runs": len(jobs),
        "complete_runs": len(complete),
        "in_progress": in_progress,
        "pending_count": len(pending),
        "pending_next": pending[:5],
        "is_complete": len(complete) == len(jobs),
    }


def latest_progress_mtime() -> float:
    latest = 0.0
    root = ABLATION_RESULTS_ROOT
    if not root.exists():
        return latest
    for path in root.rglob("*"):
        if path.name in {"checkpoint.jsonl", "scores.json"} and path.is_file():
            latest = max(latest, path.stat().st_mtime)
    status = STATUS_PATH
    if status.exists():
        latest = max(latest, status.stat().st_mtime)
    return latest


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def format_status_report(progress: dict | None = None) -> str:
    progress = progress or scan_progress()
    lines = [
        "Ablation supervisor status",
        f"  Complete: {progress['complete_runs']}/{progress['total_runs']} runs",
        f"  Pending:  {progress['pending_count']} runs",
    ]
    if progress["in_progress"]:
        lines.append("  In progress:")
        for item in progress["in_progress"]:
            lines.append(
                f"    - {item['label']} ({item['questions_done']}/60 questions checkpointed)"
            )
    elif progress["pending_next"]:
        lines.append(f"  Next up: {progress['pending_next'][0]}")
    if progress["is_complete"]:
        lines.append("  ALL RUNS COMPLETE")
    status = load_status()
    if status:
        lines.append("")
        lines.append(f"  Supervisor attempts: {status.get('attempts', '?')}")
        lines.append(f"  Last update: {status.get('last_update', '?')}")
        if status.get("message"):
            lines.append(f"  Message: {status['message']}")
    return "\n".join(lines)


def build_status_snapshot(
    *,
    attempts: int,
    child_pid: int | None,
    message: str,
    progress: dict | None = None,
) -> dict:
    progress = progress or scan_progress()
    return {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attempts": attempts,
        "child_pid": child_pid,
        "message": message,
        "progress": progress,
    }
