import re

import requests

from llm import call_ollama, parse_json_response
from prompts import extract_final_answer, format_context

SCORE_WEIGHTS = {
    "answer_correctness": 0.35,
    "faithfulness": 0.25,
    "context_recall": 0.20,
    "context_precision": 0.10,
    "citation_accuracy": 0.10,
}

# Judge receives capped context even when generation uses more chunks (top_k > 5 or
# large chunk_size). Full retrieved list is still used for citation_accuracy.
JUDGE_MAX_CONTEXT_CHUNKS = 5
# Word budget keeps judge input below the ~2,500-token failure zone (top7 at 1,792
# words was fine; top10 / chunk512 at ~2,560 broke). Five 512-word chunks still
# exceed this, so the word cap applies in addition to the chunk cap.
JUDGE_MAX_CONTEXT_WORDS = 1920


def judge_context_chunks(retrieved):
    selected = []
    word_count = 0

    for chunk in retrieved[:JUDGE_MAX_CONTEXT_CHUNKS]:
        chunk_words = chunk["text"].split()
        chunk_word_count = len(chunk_words)

        if word_count + chunk_word_count <= JUDGE_MAX_CONTEXT_WORDS:
            selected.append(chunk)
            word_count += chunk_word_count
            continue

        remaining = JUDGE_MAX_CONTEXT_WORDS - word_count
        if remaining > 0:
            trimmed = dict(chunk)
            trimmed["text"] = " ".join(chunk_words[:remaining])
            selected.append(trimmed)
        break

    return selected


def compute_final_score(metrics):
    total = 0.0
    for name, weight in SCORE_WEIGHTS.items():
        total += metrics.get(name, 0.0) * weight
    return round(total, 2)


def cited_doc_ids(answer):
    ids = set()
    for match in re.finditer(r"\[Doc\s*(\d+)\]", answer, re.IGNORECASE):
        ids.add(int(match.group(1)))
    return ids


def estimate_citation_accuracy(answer, retrieved, expected_source):
    answer_text = extract_final_answer(answer)
    doc_ids = cited_doc_ids(answer_text)

    if not doc_ids:
        return 0.0

    cited_sources = []
    valid_ids = 0
    for doc_id in doc_ids:
        index = doc_id - 1
        if 0 <= index < len(retrieved):
            valid_ids += 1
            cited_sources.append(retrieved[index]["source"])

    if valid_ids == 0:
        return 0.0

    score = (valid_ids / len(doc_ids)) * 40.0

    expected_lower = expected_source.lower()
    if any(source.lower() == expected_lower for source in cited_sources):
        score += 50.0

    if len(set(cited_sources)) == 1 and cited_sources[0].lower() == expected_lower:
        score += 10.0

    return min(score, 100.0)


def normalize_metric_scores(metrics):
    for key in ("answer_correctness", "faithfulness", "context_recall", "context_precision"):
        value = float(metrics.get(key, 0))
        metrics[key] = max(0.0, min(100.0, value))
    return metrics


def fallback_metric_scores(question, expected_answer, answer, retrieved):
    answer_text = extract_final_answer(answer)
    expected_words = set(expected_answer.lower().split())
    answer_words = set(answer_text.lower().split())
    overlap = len(expected_words & answer_words)
    correctness = min(100.0, (overlap / max(len(expected_words), 1)) * 100)

    context_text = " ".join(item["text"].lower() for item in retrieved)
    context_overlap = len(expected_words & set(context_text.split()))
    context_recall = min(100.0, (context_overlap / max(len(expected_words), 1)) * 100)

    return {
        "answer_correctness": correctness,
        "faithfulness": correctness * 0.8,
        "context_recall": context_recall,
        "context_precision": 60.0 if retrieved else 0.0,
        "judge_fallback": True,
    }


def judge_metrics(question, expected_answer, expected_source, answer, retrieved, judge_model=None):
    answer_text = extract_final_answer(answer)
    context = format_context(judge_context_chunks(retrieved))

    judge_prompt = f"""You are a strict RAG evaluator. Do not answer the question.
Score only the final answer content (ignore planning sections if present).

Return ONLY a JSON object with exactly these numeric keys (0-100):
{{
  "answer_correctness": 0,
  "faithfulness": 0,
  "context_recall": 0,
  "context_precision": 0
}}

Definitions:
- answer_correctness: how well the generated answer matches the expected answer
- faithfulness: is the generated answer supported by the retrieved context only
- context_recall: does the retrieved context contain the information needed for the expected answer
- context_precision: are the retrieved chunks relevant and low-noise for the question

Question:
{question}

Expected answer:
{expected_answer}

Expected source file:
{expected_source}

Retrieved context:
{context}

Generated answer:
{answer_text}
"""

    try:
        raw, _ = call_ollama(judge_prompt, json_mode=True, model=judge_model)
        return normalize_metric_scores(parse_json_response(raw))
    except (ValueError, requests.RequestException):
        pass

    try:
        retry_prompt = judge_prompt + "\nReminder: output JSON only, no explanation."
        raw, _ = call_ollama(retry_prompt, json_mode=True, model=judge_model)
        return normalize_metric_scores(parse_json_response(raw))
    except (ValueError, requests.RequestException):
        return fallback_metric_scores(
            question, expected_answer, answer_text, judge_context_chunks(retrieved)
        )


def score_answer(question, expected_answer, expected_source, answer, retrieved, judge_model=None):
    metrics = judge_metrics(
        question=question,
        expected_answer=expected_answer,
        expected_source=expected_source,
        answer=answer,
        retrieved=retrieved,
        judge_model=judge_model,
    )
    metrics["citation_accuracy"] = estimate_citation_accuracy(
        answer, retrieved, expected_source
    )
    metrics["final_score"] = compute_final_score(metrics)
    return metrics
