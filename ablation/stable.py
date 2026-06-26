"""Stability helpers — keep torch/sentence-transformers off the GPU during ablations."""

from __future__ import annotations

import os
import time

CRASH_EXIT_CODES = {139, -11, 134, -6, 135, -7}

# Pauses between eval steps — wall-clock only, does not change scores or model behavior.
DEFAULT_BREATHER_QUESTION_S = 10
DEFAULT_BREATHER_RUN_S = 180
DEFAULT_QUESTION_TIMEOUT_S = 600
DEFAULT_QUESTION_TIMEOUT_LAST_S = 1200


def enable_stable_mode() -> None:
    """Best-effort defaults to reduce WSL2/CUDA segfaults during long ablation runs."""
    os.environ.setdefault("RAG_CPU_EMBED", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("RAG_BREATHER_QUESTION_S", str(DEFAULT_BREATHER_QUESTION_S))
    os.environ.setdefault("RAG_BREATHER_RUN_S", str(DEFAULT_BREATHER_RUN_S))
    os.environ.setdefault("RAG_QUESTION_TIMEOUT_S", str(DEFAULT_QUESTION_TIMEOUT_S))
    os.environ.setdefault("RAG_QUESTION_TIMEOUT_LAST_S", str(DEFAULT_QUESTION_TIMEOUT_LAST_S))


def stable_env() -> dict[str, str]:
    enable_stable_mode()
    return os.environ.copy()


def is_crash_exit(code: int) -> bool:
    return code in CRASH_EXIT_CODES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def breather_question_seconds() -> int:
    return _env_int("RAG_BREATHER_QUESTION_S", 0)


def breather_run_seconds() -> int:
    return _env_int("RAG_BREATHER_RUN_S", 0)


def question_timeout_seconds(*, final_attempt: bool = False) -> int:
    if final_attempt:
        return _env_int("RAG_QUESTION_TIMEOUT_LAST_S", DEFAULT_QUESTION_TIMEOUT_LAST_S)
    return _env_int("RAG_QUESTION_TIMEOUT_S", 0)


def breather_after_question() -> None:
    seconds = breather_question_seconds()
    if seconds > 0:
        print(f"[stable] breather {seconds}s after question", flush=True)
        time.sleep(seconds)


def breather_after_run(label: str = "") -> None:
    seconds = breather_run_seconds()
    if seconds > 0:
        suffix = f" ({label})" if label else ""
        print(f"[stable] breather {seconds}s after run{suffix}", flush=True)
        time.sleep(seconds)
