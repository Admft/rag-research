from indexing import build_experiment_index, close_cached_indices
from pipeline import run_experiment
from run_storage import MASTER_LOG, save_run_folder


def format_experiment_report(payload):
    summary = payload["summary"]
    config = summary["config"]
    lines = [
        "=" * 72,
        f"EXPERIMENT: {config['name']}",
        "=" * 72,
        f"Run folder: {payload['run_folder']}",
        f"Round:      {config.get('round', '')}",
        f"Mode:       {payload.get('run_mode', '')}",
        f"Run time:   {payload['run_time_central']}",
        "",
        "CONFIG",
        "-" * 40,
        f"Chunk size:       {config['chunk_size']}",
        f"Overlap:          {config['chunk_overlap']}",
        f"Top-k:            {config['top_k']}",
        f"Retriever:        {config['retriever']}",
        f"Embedding model:  {config['embedding_model']}",
        f"Reranker:         {config['reranker']}",
        f"Query transform:  {config['query_transform']}",
        f"Prompt:           {config['prompt']}",
        f"Context filter:   {config['context_filter']}",
        f"Generator:        {config.get('generator', '—')}",
        f"Judge:            {config.get('judge', '—')}",
        "",
        "SCORES",
        "-" * 40,
    ]

    if "final_score" in summary:
        lines.extend([
            f"Final score:          {summary['final_score']}",
            f"Answer correctness:   {summary['answer_correctness']}",
            f"Faithfulness:         {summary['faithfulness']}",
            f"Context recall:       {summary['context_recall']}",
            f"Context precision:    {summary['context_precision']}",
            f"Citation accuracy:    {summary['citation_accuracy']}",
            f"Answer parse rate:    {summary.get('answer_parse_rate', 0) * 100:.1f}%",
        ])

    lines.extend([
        f"Recall@{config['top_k']}:            {summary['recall_at_k'] * 100:.1f}%",
        f"MRR@{config['top_k']}:                {summary['mrr_at_k'] * 100:.1f}%",
        f"Avg latency:          {summary['avg_latency_s']}s",
        "",
        "PER-QUESTION",
        "-" * 40,
    ])

    for item in payload["questions"]:
        lines.append("")
        lines.append(f"[{item['id']}] {item['question']}")
        lines.append(f"  Expected source: {item['expected_source']}")
        lines.append(f"  Recall hit: {'yes' if item['recall_hit'] else 'no'} (rank {item['found_rank']})")
        if "answer" in item:
            lines.append(f"  Final score: {item['metrics']['final_score']}")
            if not item.get("answer_parsed", True):
                lines.append("  Warning: structured output parse failed (JSON or <answer> missing)")
            display = item.get("answer") or item.get("raw_answer", "")
            lines.append(f"  Answer: {display[:300]}")

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def experiment_summary_row(summary, run_mode):
    config = summary["config"]
    row = {
        "run_mode": run_mode,
        "round": config.get("round", ""),
        "chunk_size": config["chunk_size"],
        "overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "retriever": config["retriever"],
        "embedding_model": config["embedding_model"],
        "reranker": config["reranker"],
        "query_transform": config["query_transform"],
        "prompt": config["prompt"],
        "context_filter": config["context_filter"],
        "recall_at_k": round(summary.get("recall_at_k", 0.0) * 100, 2),
        "mrr_at_k": round(summary.get("mrr_at_k", 0.0) * 100, 2),
        "avg_latency": summary.get("avg_latency_s", ""),
        "question_count": summary.get("question_count", ""),
    }
    if "final_score" in summary:
        row.update({
            "final_score": summary["final_score"],
            "answer_correctness": summary.get("answer_correctness", ""),
            "faithfulness": summary.get("faithfulness", ""),
            "context_recall": summary.get("context_recall", ""),
            "context_precision": summary.get("context_precision", ""),
            "citation_accuracy": summary.get("citation_accuracy", ""),
        })
    return row


def save_experiment_result(
    config,
    payload,
    run_mode="full_pipeline",
    run_kind="experiment",
    run_number=None,
    report_builder=None,
):
    data = {
        "run_mode": run_mode,
        "summary": payload["summary"],
        "questions": payload["questions"],
    }
    return save_run_folder(
        run_name=config.name,
        run_kind=run_kind,
        data=data,
        report_builder=report_builder or format_experiment_report,
        questions=payload["questions"],
        summary_row=experiment_summary_row(payload["summary"], run_mode),
        run_number=run_number,
    )


def run_grid(configs, questions, retrieval_only=False, show_progress=False):
    index_cache = {}
    results = []

    for i, config in enumerate(configs, start=1):
        print()
        print("=" * 72)
        print(f"[{i}/{len(configs)}] {config.name} ({config.round})")
        print("=" * 72)

        key = config.index_key()
        if key not in index_cache:
            print("Building index...")
            close_cached_indices(index_cache)
            index_cache[key] = build_experiment_index(config, show_progress=show_progress)[0]
        else:
            print("Reusing cached index for this chunk/embedding setting.")

        payload = run_experiment(
            config=config,
            questions=questions,
            index=index_cache[key],
            retrieval_only=retrieval_only,
            show_progress=show_progress,
        )

        run_dir, master_log = save_experiment_result(
            config,
            {
                "summary": payload["summary"],
                "questions": payload["questions"],
            },
            run_mode="retrieval_only" if retrieval_only else "full_pipeline",
        )
        results.append(payload["summary"])

        print(f"Saved run folder: {run_dir}")
        if "final_score" in payload["summary"]:
            print(f"Final score: {payload['summary']['final_score']}")
        print(
            f"Recall@{config.top_k}: {payload['summary']['recall_at_k'] * 100:.1f}% | "
            f"MRR@{config.top_k}: {payload['summary']['mrr_at_k'] * 100:.1f}%"
        )

    print()
    print(f"Master log: {master_log}")
    close_cached_indices(index_cache)
    return results
