import json
from datetime import datetime, timezone

from config import RESULTS_DIR


def build_config_snapshot():
    from config import (
        CHUNK_SIZE_WORDS,
        COLLECTION_NAME,
        DISTANCE,
        EMBEDDING_MODEL_NAME,
        EVAL_TOP_KS,
        OVERLAP_WORDS,
        QDRANT_PATH,
    )

    step_words = CHUNK_SIZE_WORDS - OVERLAP_WORDS

    return {
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "step_words": step_words,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection_name": COLLECTION_NAME,
        "distance": DISTANCE,
        "qdrant_path": str(QDRANT_PATH),
        "eval_top_ks": EVAL_TOP_KS,
    }


def load_latest_build_run():
    latest_path = RESULTS_DIR / "latest_build_index.json"
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
        f"Run time: {payload['run_time_local']}",
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
        "Run evaluate_retrieval.py to get Recall@k and MRR@k scores.",
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
        f"Run time: {payload['run_time_local']}",
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


def save_run(run_type, data, report_formatter=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    run_time_local = timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    payload = {
        "run_type": run_type,
        "run_time_utc": timestamp.isoformat(),
        "run_time_local": run_time_local,
        "index_settings": build_config_snapshot(),
        **data,
    }

    json_path = RESULTS_DIR / f"{run_type}_{stamp}.json"
    latest_json_path = RESULTS_DIR / f"latest_{run_type}.json"
    txt_path = RESULTS_DIR / f"{run_type}_{stamp}.txt"
    latest_txt_path = RESULTS_DIR / f"latest_{run_type}.txt"

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
            "build_run_time_utc": linked_build.get("run_time_utc"),
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
    }, report_formatter=format_eval_report)
