import json
import re
import time

import requests

from config import (
    OLLAMA_GENERATION_MAX_TOKENS,
    OLLAMA_GENERATION_TIMEOUT,
    OLLAMA_JUDGE_MAX_TOKENS,
    OLLAMA_JUDGE_TIMEOUT,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_URL,
)


def call_ollama(
    prompt,
    model=None,
    timeout=None,
    json_mode=False,
    json_schema=None,
    max_tokens=None,
    retries=None,
):
    model = model or OLLAMA_MODEL
    if max_tokens is None:
        max_tokens = (
            OLLAMA_JUDGE_MAX_TOKENS if json_mode and not json_schema
            else OLLAMA_GENERATION_MAX_TOKENS
        )
    if timeout is None:
        timeout = (
            OLLAMA_JUDGE_TIMEOUT if json_mode and not json_schema
            else OLLAMA_GENERATION_TIMEOUT
        )
    retries = OLLAMA_MAX_RETRIES if retries is None else retries
    start = time.perf_counter()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if json_schema:
        payload["format"] = json_schema
    elif json_mode:
        payload["format"] = "json"

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            return response.json()["response"], elapsed
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status >= 500 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise last_error


def parse_json_response(text):
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")

    return json.loads(text[start:end + 1])
