import json
from datetime import datetime, timezone

from config import (
    CHUNK_SIZE_WORDS,
    DATASET_STAGE,
    OVERLAP_WORDS,
    RESULTS_DIR,
    RESULT_TIMEZONE,
    TARGET_DOC_COUNT_MAX,
    TARGET_DOC_COUNT_MIN,
    TARGET_QUESTION_COUNT,
)


RUN_CATEGORIES = {
    "build_index": {
        "label": "build",
        "category": "baseline",
        "subcategory": "retrieval",
        "title": "Baseline retrieval",
    },
    "evaluate_retrieval": {
        "label": "eval",
        "category": "baseline",
        "subcategory": "evaluation",
        "title": "Baseline evaluation",
    },
    "generation": {
        "label": "gen",
        "category": "baseline",
        "subcategory": "generation",
        "title": "Baseline generation",
    },
}


def run_results_dir(run_type):
    info = RUN_CATEGORIES[run_type]
    return RESULTS_DIR / info["category"] / info["subcategory"]


def latest_result_path(run_type, extension="json"):
    return run_results_dir(run_type) / f"latest.{extension}"


def build_config_snapshot():
    from config import (
        COLLECTION_NAME,
        DATASET_STAGE,
        DISTANCE,
        EMBEDDING_MODEL_NAME,
        EVAL_TOP_KS,
        OLLAMA_MODEL,
        QDRANT_PATH,
        TARGET_QUESTION_COUNT,
    )

    step_words = CHUNK_SIZE_WORDS - OVERLAP_WORDS

    return {
        "dataset_stage": DATASET_STAGE,
        "target_question_count": TARGET_QUESTION_COUNT,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "step_words": step_words,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "generation_model": OLLAMA_MODEL,
        "collection_name": COLLECTION_NAME,
        "distance": DISTANCE,
        "qdrant_path": str(QDRANT_PATH),
        "eval_top_ks": EVAL_TOP_KS,
    }


def get_run_times():
    utc_now = datetime.now(timezone.utc)
    central_dt = utc_now.astimezone(RESULT_TIMEZONE)
    return utc_now, central_dt


def config_slug(settings=None):
    settings = settings or build_config_snapshot()
    return f"chunk{settings['chunk_size_words']}-overlap{settings['overlap_words']}"


def format_central_timestamp(central_dt):
    tz_abbr = central_dt.tzname() or "CT"
    return central_dt.strftime(f"%Y-%m-%d_%H-%M-%S_{tz_abbr}")


def build_run_basename(run_type, central_dt, settings=None, metrics=None):
    label = RUN_CATEGORIES[run_type]["label"]
    parts = [
        label,
        config_slug(settings),
    ]

    if run_type == "evaluate_retrieval" and metrics:
        recall_1 = metrics.get("1", {}).get("recall")
        mrr_1 = metrics.get("1", {}).get("mrr")
        if recall_1 is not None and mrr_1 is not None:
            parts.append(f"r1-{recall_1:.3f}_mrr1-{mrr_1:.3f}")

    parts.append(format_central_timestamp(central_dt))
    return "__".join(parts)


def load_latest_build_run():
    latest_path = latest_result_path("build_index")
    if not latest_path.exists():
        return None

    with latest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_settings_block(settings):
    lines = [
        "INDEX SETTINGS",
        "-" * 40,
        f"Chunk size:      {settings['chunk_size_words']} words",
        f"Overlap:         {settings['overlap_words']} words",
        f"Step size:       {settings['step_words']} words",
        f"Embedding model: {settings['embedding_model']}",
        f"Collection:      {settings['collection_name']}",
        f"Distance:        {settings['distance']}",
    ]
    return "\n".join(lines)


def format_metric_line(name, value, hits=None, total=None):
    line = f"{name:<10} {value:.3f}"
    if hits is not None and total is not None:
        line += f"  ({hits}/{total} questions)"
    return line


def format_build_report(payload):
    settings = payload["index_settings"]
    stats = payload["build_stats"]
    lines = [
        "=" * 72,
        "INDEX BUILD RESULTS",
        "=" * 72,
        f"Run time: {payload['run_time_central']}",
        "",
        format_settings_block(settings),
        "",
        "BUILD SUMMARY",
        "-" * 40,
        f"Documents indexed: {stats['document_count']}",
        f"Total chunks:      {stats['chunk_count']}",
        f"Vector size:       {stats['vector_size']}",
        f"Avg chunk length:  {stats['avg_chunk_words']:.1f} words",
        "",
        "SOURCE BREAKDOWN",
        "-" * 40,
    ]

    for source in stats["source_breakdown"]:
        lines.append(
            f"  {source['source']}: {source['chunk_count']} chunks "
            f"(avg {source['avg_words']:.1f} words)"
        )

    lines.extend([
        "",
        "CHUNKS CREATED",
        "-" * 40,
    ])

    for chunk in stats["chunks"]:
        preview = chunk["preview"]
        lines.append(
            f"  [{chunk['chunk_index']}] {chunk['source']} "
            f"({chunk['word_count']} words): \"{preview}\""
        )

    lines.extend([
        "",
        "OUTPUT PATHS",
        "-" * 40,
        f"Chunks file:  {payload['outputs']['chunks_path']}",
        f"Qdrant path:  {payload['outputs']['qdrant_path']}",
        "",
        "NEXT STEP",
        "-" * 40,
        "Run evaluate_retrieval.py for baseline evaluation.",
        "Run milestone_status.py to track Part 21 progress.",
        "See docs/part21-next-milestone.md for the full checklist.",
        "=" * 72,
    ])

    return "\n".join(lines)


def format_eval_report(payload):
    settings = payload["index_settings"]
    summary = payload["summary"]
    lines = [
        "=" * 72,
        "RETRIEVAL EVALUATION RESULTS",
        "=" * 72,
        f"Run time: {payload['run_time_central']}",
        f"Eval set: {payload['eval_file']} ({payload['question_count']} questions)",
        "",
        format_settings_block(settings),
        "",
        "SUMMARY METRICS",
        "-" * 40,
    ]

    for top_k in settings["eval_top_ks"]:
        key = str(top_k)
        recall = summary["metrics"][f"recall_at_{key}"]
        mrr = summary["metrics"][f"mrr_at_{key}"]
        lines.append(format_metric_line(
            f"Recall@{top_k}",
            recall["value"],
            recall["hits"],
            recall["total"],
        ))
        lines.append(format_metric_line(f"MRR@{top_k}", mrr["value"]))
        lines.append("")

    lines.extend([
        "PER-QUESTION RESULTS",
        "-" * 40,
    ])

    for item in payload["questions"]:
        status = "HIT" if item["found_rank"] is not None else "MISS"
        rank_text = f"rank {item['found_rank']}" if item["found_rank"] else "not in top results"

        lines.extend([
            "",
            f"[{item['id']}] {item['question']}",
            f"  Expected source: {item['expected_source']}",
            f"  Expected answer: {item['expected_answer']}",
            f"  Result: {status} ({rank_text})",
        ])

        for top_k in settings["eval_top_ks"]:
            hit = "yes" if item["found_rank"] is not None and item["found_rank"] <= top_k else "no"
            lines.append(f"  Recall@{top_k}: {hit}")

        lines.append("  Top retrieved chunks:")
        for hit in item["retrieved"]:
            lines.append(
                f"    #{hit['rank']} {hit['source']} chunk {hit['chunk_index']} "
                f"(score {hit['score']:.3f})"
            )
            if hit.get("text_preview"):
                lines.append(f"       \"{hit['text_preview']}\"")

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def format_generation_report(payload):
    settings = payload["index_settings"]
    lines = [
        "=" * 72,
        "BASELINE GENERATION RESULT",
        "=" * 72,
        f"Run time: {payload['run_time_central']}",
        "",
        format_settings_block(settings),
        "",
        "GENERATION SETTINGS",
        "-" * 40,
        f"Model: {settings['generation_model']}",
        f"Top-k retrieved: {payload['top_k']}",
        "",
        "QUESTION",
        "-" * 40,
        payload["query"],
        "",
        "ANSWER",
        "-" * 40,
        payload["answer"],
        "",
        "RETRIEVED SOURCES",
        "-" * 40,
    ]

    for hit in payload["retrieved"]:
        lines.extend([
            f"  #{hit['rank']} {hit['source']} chunk {hit['chunk_index']} (score {hit['score']:.3f})",
            f"     \"{hit['text_preview']}\"",
            "",
        ])

    lines.append("=" * 72)
    return "\n".join(lines)


def save_run(run_type, data, report_formatter=None, metrics=None):
    output_dir = run_results_dir(run_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    utc_now, central_dt = get_run_times()
    settings = build_config_snapshot()
    basename = build_run_basename(run_type, central_dt, settings, metrics=metrics)
    category_info = RUN_CATEGORIES[run_type]

    payload = {
        "run_type": run_type,
        "run_category": category_info["title"],
        "run_name": basename,
        "run_time_utc": utc_now.isoformat(),
        "run_time_central": central_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "run_timezone": "America/Chicago",
        "milestone": {
            "name": "Part 21: Local RAG baseline over 20-50 documents",
            "dataset_stage": settings["dataset_stage"],
            "target_documents": f"{TARGET_DOC_COUNT_MIN}-{TARGET_DOC_COUNT_MAX}",
            "target_questions": TARGET_QUESTION_COUNT,
        },
        "index_settings": settings,
        **data,
    }

    json_path = output_dir / f"{basename}.json"
    latest_json_path = latest_result_path(run_type, "json")
    txt_path = output_dir / f"{basename}.txt"
    latest_txt_path = latest_result_path(run_type, "txt")

    for path in (json_path, latest_json_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    if report_formatter:
        report = report_formatter(payload)
        for path in (txt_path, latest_txt_path):
            path.write_text(report + "\n", encoding="utf-8")

    return json_path, txt_path


def save_build_results(documents, chunks, vector_size, outputs):
    source_stats = {}

    for chunk in chunks:
        word_count = len(chunk["text"].split())
        entry = source_stats.setdefault(chunk["source"], {"word_counts": []})
        entry["word_counts"].append(word_count)

    source_breakdown = []
    chunk_rows = []

    for source, info in sorted(source_stats.items()):
        counts = info["word_counts"]
        source_breakdown.append({
            "source": source,
            "chunk_count": len(counts),
            "avg_words": sum(counts) / len(counts),
        })

    for chunk in chunks:
        words = chunk["text"].split()
        chunk_rows.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "word_count": len(words),
            "preview": " ".join(words[:12]) + ("..." if len(words) > 12 else ""),
        })

    all_word_counts = [row["word_count"] for row in chunk_rows]

    return save_run("build_index", {
        "build_stats": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "vector_size": vector_size,
            "avg_chunk_words": sum(all_word_counts) / len(all_word_counts),
            "sources": [doc["source"] for doc in documents],
            "source_breakdown": source_breakdown,
            "chunks": chunk_rows,
        },
        "outputs": outputs,
    }, report_formatter=format_build_report)


def build_eval_summary(metrics, question_count, top_ks):
    summary_metrics = {}

    for top_k in top_ks:
        key = str(top_k)
        recall = metrics[key]["recall"]
        mrr = metrics[key]["mrr"]
        hits = metrics[key]["hits"]

        summary_metrics[f"recall_at_{key}"] = {
            "value": recall,
            "hits": hits,
            "total": question_count,
            "label": f"Recall@{top_k}",
        }
        summary_metrics[f"mrr_at_{key}"] = {
            "value": mrr,
            "label": f"MRR@{top_k}",
        }

    return {
        "questions_evaluated": question_count,
        "metrics": summary_metrics,
    }


def save_eval_results(eval_file, question_count, metrics, per_question, top_ks):
    linked_build = load_latest_build_run()
    build_ref = None

    if linked_build:
        build_ref = {
            "run_name": linked_build.get("run_name"),
            "run_time_utc": linked_build.get("run_time_utc"),
            "run_time_central": linked_build.get("run_time_central"),
            "build_stats": linked_build.get("build_stats"),
        }

    return save_run("evaluate_retrieval", {
        "eval_file": str(eval_file),
        "question_count": question_count,
        "linked_build": build_ref,
        "summary": build_eval_summary(metrics, question_count, top_ks),
        "metrics": {
            f"recall_at_{k}": metrics[str(k)]["recall"]
            for k in top_ks
        } | {
            f"mrr_at_{k}": metrics[str(k)]["mrr"]
            for k in top_ks
        },
        "questions": per_question,
    }, report_formatter=format_eval_report, metrics=metrics)


def save_generation_results(query, answer, retrieved, top_k):
    linked_build = load_latest_build_run()
    build_ref = None

    if linked_build:
        build_ref = {
            "run_name": linked_build.get("run_name"),
            "run_time_central": linked_build.get("run_time_central"),
            "build_stats": linked_build.get("build_stats"),
        }

    retrieved_rows = []
    for rank, point in enumerate(retrieved, start=1):
        words = point.payload["text"].split()
        preview = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
        retrieved_rows.append({
            "rank": rank,
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "score": point.score,
            "text_preview": preview,
        })

    return save_run("generation", {
        "query": query,
        "answer": answer,
        "top_k": top_k,
        "linked_build": build_ref,
        "retrieved": retrieved_rows,
    }, report_formatter=format_generation_report)
