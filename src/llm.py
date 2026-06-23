import json
import re
import time

import requests

from config import OLLAMA_MODEL, OLLAMA_URL


def call_ollama(prompt, model=None, timeout=180, json_mode=False, max_tokens=2048):
    model = model or OLLAMA_MODEL
    start = time.perf_counter()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    else:
        payload["options"] = {"num_predict": max_tokens}

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - start

    return response.json()["response"], elapsed


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
