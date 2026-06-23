import json

from config import (
    CHUNK_SIZE_WORDS,
    DATASET_STAGE,
    EVAL_TOP_KS,
    OVERLAP_WORDS,
    TARGET_DOC_COUNT_MAX,
    TARGET_DOC_COUNT_MIN,
    TARGET_QUESTION_COUNT,
)
from run_storage import get_run_times, load_latest_build_run, save_run_folder


def build_config_snapshot():
    from config import (
        COLLECTION_NAME,
        DISTANCE,
        EMBEDDING_MODEL_NAME,
        OLLAMA_MODEL,
        QDRANT_PATH,
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
        "INDEX BUILD",
        "=" * 72,
        f"Run folder: {payload['run_folder']}",
        f"Run time:   {payload['run_time_central']}",
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

    lines.extend(["", "CHUNKS CREATED", "-" * 40])
    for chunk in stats["chunks"]:
        lines.append(
            f"  [{chunk['chunk_index']}] {chunk['source']} "
            f"({chunk['word_count']} words): \"{chunk['preview']}\""
        )

    lines.extend([
        "",
        "OUTPUT PATHS",
        "-" * 40,
        f"Chunks file: {payload['outputs']['chunks_path']}",
        f"Qdrant path: {payload['outputs']['qdrant_path']}",
        "=" * 72,
    ])
    return "\n".join(lines)


def format_eval_report(payload):
    settings = payload["index_settings"]
    summary = payload["summary"]
    lines = [
        "=" * 72,
        "RETRIEVAL EVALUATION",
        "=" * 72,
        f"Run folder: {payload['run_folder']}",
        f"Run time:   {payload['run_time_central']}",
        f"Eval set:   {payload['eval_file']} ({payload['question_count']} questions)",
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
            f"Recall@{top_k}", recall["value"], recall["hits"], recall["total"],
        ))
        lines.append(format_metric_line(f"MRR@{top_k}", mrr["value"]))
        lines.append("")

    lines.extend(["PER-QUESTION RESULTS", "-" * 40])
    for item in payload["questions"]:
        status = "HIT" if item["found_rank"] is not None else "MISS"
        rank_text = f"rank {item['found_rank']}" if item["found_rank"] else "not in top results"
        lines.extend([
            "",
            f"[{item['id']}] {item['question']}",
            f"  Expected source: {item['expected_source']}",
            f"  Result: {status} ({rank_text})",
            "  Top retrieved chunks:",
        ])
        for hit in item["retrieved"]:
            lines.append(
                f"    #{hit['rank']} {hit['source']} chunk {hit['chunk_index']} "
                f"(score {hit['score']:.3f})"
            )

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def format_generation_report(payload):
    settings = payload["index_settings"]
    lines = [
        "=" * 72,
        "GENERATION (single question)",
        "=" * 72,
        f"Run folder: {payload['run_folder']}",
        f"Run time:   {payload['run_time_central']}",
        "",
        format_settings_block(settings),
        "",
        f"Model: {settings['generation_model']} | Top-k: {payload['top_k']}",
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
        doc_id = hit.get("rank", "?")
        lines.append(
            f"  [Doc {doc_id}] {hit['source']} chunk {hit['chunk_index']} "
            f"(score {hit['score']:.3f})"
        )
        lines.append(f"     \"{hit['text_preview']}\"")
    lines.extend(["", "=" * 72])
    return "\n".join(lines)


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
    settings = build_config_snapshot()

    data = {
        "index_settings": settings,
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
    }

    return save_run_folder(
        run_name="build_index",
        run_kind="build_index",
        data=data,
        report_builder=format_build_report,
        summary_row={
            "chunk_size": settings["chunk_size_words"],
            "overlap": settings["overlap_words"],
            "question_count": len(documents),
        },
    )


def build_eval_summary(metrics, question_count, top_ks):
    summary_metrics = {}
    for top_k in top_ks:
        key = str(top_k)
        recall = metrics[key]["recall"]
        mrr = metrics[key]["mrr"]
        hits = metrics[key]["hits"]
        summary_metrics[f"recall_at_{key}"] = {
            "value": recall, "hits": hits, "total": question_count,
            "label": f"Recall@{top_k}",
        }
        summary_metrics[f"mrr_at_{key}"] = {"value": mrr, "label": f"MRR@{top_k}"}
    return {"questions_evaluated": question_count, "metrics": summary_metrics}


def save_eval_results(eval_file, question_count, metrics, per_question, top_ks):
    settings = build_config_snapshot()
    linked_build = load_latest_build_run()
    build_ref = None
    if linked_build:
        build_ref = {
            "run_folder": linked_build.get("run_folder"),
            "run_time_central": linked_build.get("run_time_central"),
            "build_stats": linked_build.get("build_stats"),
        }

    data = {
        "index_settings": settings,
        "eval_file": str(eval_file),
        "question_count": question_count,
        "linked_build": build_ref,
        "summary": build_eval_summary(metrics, question_count, top_ks),
        "metrics": {
            **{f"recall_at_{k}": metrics[str(k)]["recall"] for k in top_ks},
            **{f"mrr_at_{k}": metrics[str(k)]["mrr"] for k in top_ks},
        },
        "questions": per_question,
    }

    return save_run_folder(
        run_name="evaluate_retrieval",
        run_kind="evaluate_retrieval",
        data=data,
        report_builder=format_eval_report,
        questions=per_question,
        summary_row={
            "chunk_size": settings["chunk_size_words"],
            "overlap": settings["overlap_words"],
            "top_k": max(top_ks),
            "retriever": "dense",
            "embedding_model": settings["embedding_model"],
            "recall_at_k": round(metrics[str(max(top_ks))]["recall"] * 100, 2),
            "mrr_at_k": round(metrics[str(max(top_ks))]["mrr"] * 100, 2),
            "question_count": question_count,
            "run_mode": "retrieval_only",
        },
    )


def save_generation_results(query, answer, retrieved, top_k):
    settings = build_config_snapshot()
    linked_build = load_latest_build_run()
    build_ref = None
    if linked_build:
        build_ref = {
            "run_folder": linked_build.get("run_folder"),
            "run_time_central": linked_build.get("run_time_central"),
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

    data = {
        "index_settings": settings,
        "query": query,
        "answer": answer,
        "top_k": top_k,
        "linked_build": build_ref,
        "retrieved": retrieved_rows,
    }

    return save_run_folder(
        run_name="generation",
        run_kind="generation",
        data=data,
        report_builder=format_generation_report,
        summary_row={"top_k": top_k, "question_count": 1},
    )
