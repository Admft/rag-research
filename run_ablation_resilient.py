#!/usr/bin/env python3
"""Run ablations with automatic restart on segfaults and other crashes.

Each retry spawns a fresh Python process so GPU/native-library corruption does not
carry over. Incomplete runs resume from checkpoint.jsonl automatically.

Overnight mode (--until-complete) keeps restarting until every expected run folder
has scores.json, with stall detection if no checkpoint progress is observed.

Example:
  ./start_overnight_ablations.sh
  .venv/bin/python run_ablation_resilient.py --all --repair-venv --until-complete
  .venv/bin/python run_ablation_resilient.py --status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from ablation.supervisor import (  # noqa: E402
    build_status_snapshot,
    expected_runs_for_argv,
    format_status_report,
    latest_progress_mtime,
    scan_progress,
    write_status,
)

RUN_SCRIPT = ROOT / "run_ablation.py"

CRASH_EXIT_CODES = {
    139,  # SIGSEGV
    -11,
    134,  # SIGABRT
    -6,
    135,  # SIGBUS
    -7,
}

DEFAULT_STALL_TIMEOUT = 2700  # 45 min without checkpoint/scores progress
SAFE_RETRY_DELAY = 120
SAFE_STALL_TIMEOUT = 5400  # 90 min
SAFE_CRASH_COOLDOWN = 60  # extra pause after segfault before retry


def build_child_env(*, cpu_embed: bool, safe: bool) -> dict[str, str]:
    env = os.environ.copy()
    if cpu_embed or safe:
        # Keep embeddings + reranker off GPU; Ollama still uses the GPU.
        env["CUDA_VISIBLE_DEVICES"] = ""
    if safe:
        env.setdefault("OLLAMA_NUM_PARALLEL", "1")
        env.setdefault("OMP_NUM_THREADS", "4")
    return env


def repair_venv(python: str) -> None:
    """Full venv repair — partial sklearn/scipy reinstall often makes WSL2 corruption worse."""
    script = ROOT / "scripts" / "repair_venv_full.sh"
    if script.exists():
        print("[supervisor] running full venv repair (scripts/repair_venv_full.sh)...", flush=True)
        subprocess.run(["bash", str(script)], cwd=ROOT, check=False)
        return
    print("[supervisor] repairing venv packages (scikit-learn, scipy, psutil)...", flush=True)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "scikit-learn",
            "scipy",
            "psutil",
        ],
        cwd=ROOT,
        check=False,
    )


def log_line(line: str, log_handle) -> None:
    print(line, flush=True)
    if log_handle:
        log_handle.write(line + "\n")
        log_handle.flush()


def run_child_with_stall_watch(
    cmd: list[str],
    *,
    stall_timeout: int,
    attempts: int,
    child_env: dict[str, str],
    log_handle,
) -> int:
    last_progress = max(latest_progress_mtime(), time.time())
    proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env)

    write_status(
        build_status_snapshot(
            attempts=attempts,
            child_pid=proc.pid,
            message="child running",
        )
    )

    try:
        while proc.poll() is None:
            time.sleep(60)
            current = latest_progress_mtime()
            if current > last_progress:
                last_progress = current
                write_status(
                    build_status_snapshot(
                        attempts=attempts,
                        child_pid=proc.pid,
                        message="progress detected",
                    )
                )
                continue

            stalled_for = int(time.time() - last_progress)
            if stalled_for >= stall_timeout:
                log_line(
                    f"[supervisor] no progress for {stalled_for}s — killing stalled child (pid {proc.pid})",
                    log_handle,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                write_status(
                    build_status_snapshot(
                        attempts=attempts,
                        child_pid=None,
                        message=f"killed stalled child after {stalled_for}s",
                    )
                )
                return 124
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        raise

    return int(proc.returncode or 0)


def run_watchdog(
    child_argv: list[str],
    *,
    retry_delay: int,
    max_retries: int,
    repair_on_crash: bool,
    until_complete: bool,
    stall_timeout: int,
    crash_cooldown: int,
    cpu_embed: bool,
    safe: bool,
    log_path: Path | None,
) -> int:
    python = sys.executable
    cmd = [python, str(RUN_SCRIPT), *child_argv]
    child_env = build_child_env(cpu_embed=cpu_embed, safe=safe)
    expected_jobs = expected_runs_for_argv(child_argv)
    attempt = 0
    log_handle = log_path.open("a", encoding="utf-8") if log_path else None

    try:
        while True:
            progress = scan_progress(expected_jobs)
            if until_complete and progress["is_complete"]:
                log_line("[supervisor] all expected runs complete", log_handle)
                write_status(
                    build_status_snapshot(
                        attempts=attempt,
                        child_pid=None,
                        message="all runs complete",
                        progress=progress,
                    )
                )
                return 0

            attempt += 1
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line("", log_handle)
            log_line("=" * 72, log_handle)
            log_line(f"[supervisor] attempt {attempt} at {stamp}", log_handle)
            log_line(
                f"[supervisor] progress {progress['complete_runs']}/{progress['total_runs']} runs complete",
                log_handle,
            )
            log_line(f"[supervisor] command: {' '.join(cmd)}", log_handle)
            if child_env.get("CUDA_VISIBLE_DEVICES") == "":
                log_line("[supervisor] CPU-only embeddings (CUDA_VISIBLE_DEVICES='')", log_handle)
            log_line("=" * 72, log_handle)

            write_status(
                build_status_snapshot(
                    attempts=attempt,
                    child_pid=None,
                    message="starting child",
                    progress=progress,
                )
            )

            returncode = run_child_with_stall_watch(
                cmd,
                stall_timeout=stall_timeout,
                attempts=attempt,
                child_env=child_env,
                log_handle=log_handle,
            )

            progress = scan_progress(expected_jobs)
            if returncode == 0 and not (until_complete and not progress["is_complete"]):
                log_line("[supervisor] finished successfully", log_handle)
                write_status(
                    build_status_snapshot(
                        attempts=attempt,
                        child_pid=None,
                        message="child exited cleanly",
                        progress=progress,
                    )
                )
                return 0

            if until_complete and progress["is_complete"]:
                log_line("[supervisor] all expected runs complete", log_handle)
                return 0

            crashed = returncode in CRASH_EXIT_CODES
            log_line(f"[supervisor] child exited with code {returncode}", log_handle)

            if max_retries and attempt >= max_retries:
                log_line(f"[supervisor] stopping after {attempt} attempt(s)", log_handle)
                return returncode

            if repair_on_crash and (crashed or returncode in {1, 124}):
                repair_venv(python)

            wait = retry_delay
            if crashed and crash_cooldown:
                wait += crash_cooldown
                log_line(
                    f"[supervisor] crash cooldown +{crash_cooldown}s before retry",
                    log_handle,
                )

            if returncode == 124:
                log_line(
                    f"[supervisor] stall detected — retrying in {wait}s",
                    log_handle,
                )
            elif crashed:
                log_line(
                    f"[supervisor] segfault/crash — retrying in {wait}s (checkpoints resume)",
                    log_handle,
                )
            else:
                log_line(f"[supervisor] retrying in {wait}s", log_handle)

            write_status(
                build_status_snapshot(
                    attempts=attempt,
                    child_pid=None,
                    message=f"retrying after exit {returncode}",
                    progress=progress,
                )
            )
            time.sleep(wait)
    finally:
        if log_handle:
            log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation tests with auto-restart on segfaults.",
        epilog=(
            "All other flags are forwarded to run_ablation.py "
            "(e.g. --all, --ablation A1, --runs 3)."
        ),
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=30,
        help="Seconds to wait before retrying after a crash (default: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum attempts (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--repair-venv",
        action="store_true",
        help="Reinstall scikit-learn/scipy/psutil before each retry",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help=(
            "Conservative preset: --until-complete --repair-venv --cpu-embed, "
            f"{SAFE_RETRY_DELAY}s retry delay, {SAFE_STALL_TIMEOUT}s stall timeout, "
            f"+{SAFE_CRASH_COOLDOWN}s after segfaults"
        ),
    )
    parser.add_argument(
        "--cpu-embed",
        action="store_true",
        help="Force CPU embeddings via CUDA_VISIBLE_DEVICES='' (Ollama still uses GPU)",
    )
    parser.add_argument(
        "--crash-cooldown",
        type=int,
        default=0,
        help="Extra seconds to wait after a segfault before retrying (default: 0)",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="Keep restarting until every expected ablation run has scores.json",
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=DEFAULT_STALL_TIMEOUT,
        help="Kill and restart if no checkpoint/scores progress for this many seconds (default: 2700)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print supervisor progress and exit",
    )
    parser.add_argument(
        "--log",
        metavar="PATH",
        default="experiments/Results/ablation_watchdog.log",
        help="Append supervisor output to this log file",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a log file",
    )
    args, child_argv = parser.parse_known_args()

    if args.safe:
        args.repair_venv = True
        args.until_complete = True
        args.cpu_embed = True
        args.retry_delay = SAFE_RETRY_DELAY
        args.stall_timeout = SAFE_STALL_TIMEOUT
        args.crash_cooldown = SAFE_CRASH_COOLDOWN

    if args.status:
        jobs = expected_runs_for_argv(child_argv) if child_argv else None
        print(format_status_report(scan_progress(jobs)))
        raise SystemExit(0)

    if not child_argv:
        parser.error("pass at least one run_ablation.py argument (e.g. --all or --ablation A1)")

    if not RUN_SCRIPT.exists():
        raise SystemExit(f"Missing entry script: {RUN_SCRIPT}")

    log_path = None if args.no_log else Path(args.log)
    if log_path and not log_path.is_absolute():
        log_path = ROOT / log_path
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    code = run_watchdog(
        child_argv,
        retry_delay=args.retry_delay,
        max_retries=args.max_retries,
        repair_on_crash=args.repair_venv,
        until_complete=args.until_complete,
        stall_timeout=args.stall_timeout,
        crash_cooldown=args.crash_cooldown,
        cpu_embed=args.cpu_embed,
        safe=args.safe,
        log_path=log_path,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
