from llm import call_ollama, parse_json_response

SCORE_WEIGHTS = {
    "answer_correctness": 0.35,
    "faithfulness": 0.25,
    "context_recall": 0.20,
    "context_precision": 0.10,
    "citation_accuracy": 0.10,
}


def compute_final_score(metrics):
    total = 0.0
    for name, weight in SCORE_WEIGHTS.items():
        total += metrics.get(name, 0.0) * weight
    return round(total, 2)


def estimate_citation_accuracy(answer, retrieved, expected_source):
    score = 0.0
    answer_lower = answer.lower()
    expected_lower = expected_source.lower()

    cited_sources = []
    for item in retrieved:
        source = item["source"]
        chunk_tag = f"chunk {item['chunk_index']}"
        if source.lower() in answer_lower or chunk_tag in answer_lower:
            cited_sources.append(source)

    if not cited_sources:
        return 0.0

    if any(source.lower() == expected_lower for source in cited_sources):
        score += 70.0

    if len(set(cited_sources)) == 1:
        score += 30.0
    else:
        score += 10.0

    return min(score, 100.0)


def normalize_metric_scores(metrics):
    for key in ("answer_correctness", "faithfulness", "context_recall", "context_precision"):
        value = float(metrics.get(key, 0))
        metrics[key] = max(0.0, min(100.0, value))
    return metrics


def fallback_metric_scores(question, expected_answer, answer, retrieved):
    expected_words = set(expected_answer.lower().split())
    answer_words = set(answer.lower().split())
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


def judge_metrics(question, expected_answer, expected_source, answer, retrieved):
    context = "\n\n".join(
        f"[{item['source']} chunk {item['chunk_index']}]\n{item['text']}"
        for item in retrieved
    )

    judge_prompt = f"""You are a strict RAG evaluator. Do not answer the question.
Your only job is to score the system output.

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
{answer}
"""

    raw, _ = call_ollama(judge_prompt, timeout=180, json_mode=True)
    try:
        return normalize_metric_scores(parse_json_response(raw))
    except ValueError:
        retry_prompt = judge_prompt + "\nReminder: output JSON only, no explanation."
        raw, _ = call_ollama(retry_prompt, timeout=180, json_mode=True)
        try:
            return normalize_metric_scores(parse_json_response(raw))
        except ValueError:
            return fallback_metric_scores(question, expected_answer, answer, retrieved)


def score_answer(question, expected_answer, expected_source, answer, retrieved):
    metrics = judge_metrics(
        question=question,
        expected_answer=expected_answer,
        expected_source=expected_source,
        answer=answer,
        retrieved=retrieved,
    )
    metrics["citation_accuracy"] = estimate_citation_accuracy(
        answer, retrieved, expected_source
    )
    metrics["final_score"] = compute_final_score(metrics)
    return metrics
