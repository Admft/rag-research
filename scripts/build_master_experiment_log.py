#!/usr/bin/env python3
"""Build MASTER_EXPERIMENT_LOG.txt at project root from all result files."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MASTER_EXPERIMENT_LOG.txt"
RESULTS = ROOT / "experiments" / "Results"
BASELINE_DIR = RESULTS / "Runs to Establish the Baseline"
ABLATIONS_DIR = RESULTS / "Test Runs and Ablations"

BASELINE_REF = 83.22

ABLATIONS = [
    ("A1", "Ablation 1 - Retriever Type", "How much does hybrid retrieval contribute vs dense alone?", "retriever"),
    ("A2", "Ablation 2 - Generator Model Size", "How much of the gain is the model vs the pipeline?", "generator"),
    ("A3", "Ablation 3 - Output Format", "Does JSON enforcement help vs XML scratchpad?", "prompt"),
    ("A4", "Ablation 4 - Context Filter", "Is there a filter threshold that helps, or does all filtering hurt?", "context_filter"),
    ("A5", "Ablation 5 - Embedding Model", "Does a larger embedding model improve retrieval measurably?", "embedding_model"),
    ("A6", "Ablation 6 - Reranker", "How much does the reranker contribute?", "reranker"),
    ("A7", "Ablation 7 - Query Transformation", "Does HyDE help on abstractly phrased questions?", "query_transform"),
    ("A8", "Ablation 8 - Top-K", "Does giving the reranker more candidates help?", "top_k"),
    ("A9", "Ablation 9 - Chunk Size", "Is 256 tokens actually optimal?", "chunk_size"),
]

BASELINE_RUNS = [
    ("000", "og_baseline", "000__og_baseline__2026-06-22_23-40-42_CDT"),
    ("001", "grid smoke test", "001__baseline__2026-06-22_20-28-46_CDT"),
    ("002", "first full grid baseline", "002__baseline__2026-06-22_20-51-37_CDT"),
    ("003", "scratchpad/answer XML tags", "003__baseline__2026-06-22_22-43-46_CDT"),
    ("004", "token budget + scratchpad brevity", "004__baseline__2026-06-22_22-51-27_CDT"),
    ("005", "re-run of 004", "005__baseline__2026-06-22_23-04-09_CDT"),
    ("006", "v2 stack (3 changes)", "006__baseline__2026-06-23_00-38-23_CDT"),
    ("007", "v2 fixes — best run", "007__baseline__2026-06-23_01-10-54_CDT"),
]

RUN_CHANGES = {
    "000": "OG baseline: chunk 120/30, dense, no reranker, og_strict prompt, llama3.1:8b",
    "001": "Grid smoke test, retrieval_only, 2 questions, chunk 512, dense, no reranker",
    "002": "First full pipeline: chunk 256, dense, bge reranker, strict_context_with_citations, llama3.1:8b. Variables changed vs 001: chunk 512→256, added reranker, full pipeline",
    "003": "Added <scratchpad>/<answer> XML isolation. vs 002: prompt format only (1 variable)",
    "004": "Scratchpad brevity + gen max tokens 2560. vs 003: prompt token budget (1 variable)",
    "005": "Identical re-run of 004 config. vs 004: none (0 variables, variance check)",
    "006": "v2 stack: hybrid retriever + strict_context_json + qwen2.5:14b + top_sentences_5 filter. vs 005: 4 variables stacked (STACKING WARNING)",
    "007": "Disabled context_filter (none) + citation normalization in JSON schema. vs 006: 2 variables (filter off + citation fix)",
}

STACKING = {
    "000": 1, "001": 1, "002": 3, "003": 1, "004": 1, "005": 0, "006": 4, "007": 2,
}


def w(lines: list[str], text: str = "") -> None:
    lines.append(text)


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def find_run_dir(folder_name: str) -> Path | None:
    p = BASELINE_DIR / folder_name
    if p.exists():
        return p
    p = RESULTS / "runs" / folder_name
    if p.exists():
        return p
    return None


def fmt_scores(s: dict) -> str:
    if not s:
        return "NOT AVAILABLE"
    keys = [
        "final_score", "answer_correctness", "faithfulness", "context_recall",
        "context_precision", "citation_accuracy", "answer_parse_rate",
        "recall_at_k", "mrr_at_k", "avg_latency_s",
    ]
    parts = []
    for k in keys:
        if k in s:
            v = s[k]
            if k in {"recall_at_k", "mrr_at_k", "answer_parse_rate"} and v <= 1:
                v = round(v * 100, 2)
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def fmt_config_from_summary(summary: dict) -> str:
    c = summary.get("config", {})
    if not c:
        return "see data.json"
    fields = [
        "chunk_size", "chunk_overlap", "top_k", "retriever", "embedding_model",
        "reranker", "query_transform", "prompt", "context_filter", "generator", "judge",
    ]
    return "\n".join(f"  {f}: {c.get(f, '—')}" for f in fields if f in c)


def per_question_block(rows: list[dict]) -> list[str]:
    out = []
    if not rows:
        out.append("  Per-question: NOT AVAILABLE")
        return out
    out.append("  Per-question scores:")
    for r in rows:
        m = r.get("metrics", {})
        fs = m.get("final_score", "?")
        rh = r.get("recall_hit", "?")
        fr = r.get("found_rank", "—")
        out.append(
            f"    {r.get('id','?'):30s} final={fs:>6} recall_hit={rh} found_rank={fr}"
        )
    return out


def scores_from_data(data: dict) -> dict:
    s = data.get("summary", {})
    out = {}
    for k in [
        "final_score", "answer_correctness", "faithfulness", "context_recall",
        "context_precision", "citation_accuracy", "answer_parse_rate",
        "recall_at_k", "mrr_at_k", "avg_latency_s",
    ]:
        if k in s:
            out[k] = s[k]
    return out


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return round(m, 2), round(sd, 2)


def ablation_condition_scores(ablation_folder: str, condition: str, run_n: int) -> dict | None:
    p = ABLATIONS_DIR / ablation_folder / condition / f"run_{run_n}" / "scores.json"
    return load_json(p)


def ablation_questions(ablation_folder: str, condition: str, run_n: int) -> list[dict]:
    p = ABLATIONS_DIR / ablation_folder / condition / f"run_{run_n}" / "questions.jsonl"
    return load_questions(p)


def collect_question_scores_across(paths: list[Path], qid: str) -> list[tuple[str, float]]:
    out = []
    for p in paths:
        for row in load_questions(p):
            if row.get("id") == qid:
                out.append((p.parent.name, row["metrics"]["final_score"]))
    return out


def main() -> None:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w(lines, "MASTER_EXPERIMENT_LOG.txt")
    w(lines, f"Generated: {now}")
    w(lines, f"Project root: {ROOT}")
    w(lines, "")

    # === SECTION 1 ===
    w(lines, "===================================================================")
    w(lines, "1. THE STORY")
    w(lines, "===================================================================")
    w(lines, """---
PROJECT: Local RAG Pipeline Optimization for Academic Paper
RESEARCH QUESTION: Which RAG pipeline components contribute most 
to performance, and how do individual changes interact?

STORY:
This project built and optimized a local RAG pipeline over 12 
academic papers on RAG itself, evaluated on 60 hand-written 
questions across those papers (5 questions per paper, 12 papers total).

Phase 1 (Runs 000-007): Iterative baseline development.
Starting from a vanilla baseline scoring 57.48, we made 
incremental changes to reach a best score of 83.22 (run 007).
Several runs stacked multiple changes simultaneously, which 
confounded attribution — this is addressed in Phase 4.

Phase 2 (Locked Baseline): Run 007's exact config was locked 
and run 3 times to establish mean ± std, confirming score 
stability and measuring sampling variance.

Phase 3 (BEIR): The locked baseline config was run unchanged 
on three external domains (NFCorpus/medical, SciFact/scientific, 
FiQA/financial) to test generalizability beyond the development 
corpus. Key finding: performance degrades predictably with 
query format distance from training conditions — natural 
language questions work best, keyword queries break the system.

Phase 4 (Ablations A1-A9): Nine isolated ablations, each 
changing exactly one variable from the locked baseline, run 
3 times each to measure individual component contributions 
with variance.
---""")

    # === SECTION 2 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "2. SYSTEM SETUP")
    w(lines, "===================================================================")
    w(lines, """Hardware:
  CPU: Intel i9-14900K
  RAM: 64 GB
  GPU: NVIDIA RTX 4070 Ti, 12 GB VRAM
  OS: WSL2 (Linux) on Windows

Inference:
  Ollama (local), models pulled to ~/.ollama

Vector DB:
  Qdrant local mode, cosine similarity, path data/qdrant/

Embedding models:
  BAAI/bge-small-en-v1.5 (primary, locked baseline + most ablations)
  BAAI/bge-large-en-v1.5 (A5 ablation condition)

Reranker:
  BAAI/bge-reranker-base cross-encoder (CPU in ablation runs)

Generator models:
  qwen2.5:14b (primary, run 006+ and locked baseline)
  llama3.1:8b (runs 000-005 and A2 ablation condition)

Judge models:
  qwen2.5:14b (run 006+ and all Phase 2-4)
  llama3.1:8b (runs 000-005, same as generator)

Corpus:
  12 RAG academic PDFs in data/raw/ (ALL 12 indexed in every RAG-paper eval run)
  0 documents skipped at load time (verified load_raw_documents: 12 loaded, skipped=[])

Corpus inventory (12 PDFs, 60 eval questions = 5 per paper):
  2305.14283v3.pdf  Query Rewriting (queryrewrite_q1-q5)
  2309.01431v2.pdf  RGB benchmark (rgb_q1-q5)
  2309.15217v2.pdf  Ragas (ragas_q1-q5)
  2310.11511v1.pdf  SELF-RAG (selfrag_q1-q5)
  2311.08377v1.pdf  FILCO (filco_q1-q5)
  2312.10997v5.pdf  RAG survey (ragsurvey_q1-q5)
  2401.15884v3.pdf  CRAG (crag_q1-q5)
  2404.19705v2.pdf  ADAPT-LLM (adaptllm_q1-q5)
  2406.15319v3.pdf  LongRAG (longrag_q1-q5)
  2407.01219v1.pdf  RAG best practices (ragbest_q1-q5)
  2407.16833v2.pdf  SELF-ROUTE (selfroute_q1-q5)
  2501.07391v1.pdf  Enhancing RAG (enhancingrag_q1-q5)

Note: Run 000 (OG baseline) used separate index path data/qdrant_og with 1273 chunks
from the same 12 PDFs (documents=12, skipped=[]). Runs 002-007 and all Phase 2-4
ablations use data/qdrant with 558 chunks (chunk_size=256, overlap=50) from same
12 PDFs. BEIR Phase 3 uses external corpora, NOT these 12 PDFs.

BEIR datasets (Phase 3):
  NFCorpus: 3633 documents, 50 eval queries sampled
  SciFact: 5183 documents, 50 eval queries sampled
  FiQA: 57638 documents, 50 eval queries sampled""")

    # === SECTION 3 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "3. EVALUATION FRAMEWORK")
    w(lines, "===================================================================")
    w(lines, """Scoring formula (src/scoring.py):
  final_score = 0.35 × answer_correctness
              + 0.25 × faithfulness
              + 0.20 × context_recall
              + 0.10 × context_precision
              + 0.10 × citation_accuracy

Metric definitions:
  answer_correctness: LLM judge scores how well generated answer matches expected answer (0-100)
  faithfulness: LLM judge — is answer supported by retrieved context only (0-100)
  context_recall: LLM judge — do retrieved chunks contain info needed for expected answer (0-100)
  context_precision: LLM judge — are retrieved chunks relevant and low-noise (0-100)
  citation_accuracy: deterministic parser for [Doc N] tags vs expected source PDF (0-100)
  answer_parse_rate: fraction of questions where answer block parsed cleanly (JSON or XML)
  recall_at_k: binary per question — expected source PDF in top-k retrieved chunks, averaged
  mrr_at_k: mean reciprocal rank of expected source in top-k (0 if not found)
  avg_latency_s: wall-clock seconds per question (retrieve + generate + judge)

Judge boundary warning:
  Runs 000-005: judge = llama3.1:8b (same as generator)
  Runs 006-007 and all Phase 2-4: judge = qwen2.5:14b
  Scores are NOT directly comparable across this boundary — different judge = different ruler.""")

    # === SECTION 4 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "4. PHASE 1: BASELINE DEVELOPMENT (runs 000-007)")
    w(lines, "===================================================================")

    progression = []

    for run_id, name, folder in BASELINE_RUNS:
        w(lines, "")
        w(lines, f"--- RUN {run_id} — {name} ---")
        w(lines, f"Folder: {folder}")
        run_dir = find_run_dir(folder)
        if not run_dir:
            w(lines, "STATUS: NOT FOUND")
            continue
        data = load_json(run_dir / "data.json")
        scores_path = run_dir / "scores.json"
        scores = load_json(scores_path) if scores_path.exists() else None
        if data and not scores:
            scores = scores_from_data(data)
        summary = data.get("summary", {}) if data else {}
        w(lines, "Config:")
        w(lines, fmt_config_from_summary(summary) if summary else "  see REPORT.txt")
        w(lines, f"What changed vs previous: {RUN_CHANGES.get(run_id, 'see 000-007_FULL_REPORT.txt')}")
        n_vars = STACKING.get(run_id, 1)
        w(lines, f"Stacking note: {n_vars} variable(s) changed" + (" — FLAG: >1" if n_vars > 1 else ""))
        w(lines, f"Scores: {fmt_scores(scores)}")
        qrows = load_questions(run_dir / "questions.jsonl")
        lines.extend(per_question_block(qrows))
        if scores and scores.get("final_score") is not None:
            progression.append((run_id, scores))

    w(lines, "")
    w(lines, "Progression table:")
    w(lines, "Run | Final | Correctness | Faithfulness | Ctx Recall | Citation | Parse | Recall@5 | What changed")
    prog_map = {
        "000": ("57.48", "55.33", "64.17", "77.42", "0.0", "100.0", "98.33", RUN_CHANGES["000"][:60]),
        "001": ("—", "—", "—", "—", "—", "—", "100.0", "retrieval only"),
        "002": ("66.04", "69.75", "66.53", "69.8", "38.67", "—", "98.33", "first full grid + reranker"),
        "003": ("70.53", "64.95", "68.83", "71.63", "94.17", "—", "98.33", "XML scratchpad/answer tags"),
        "004": ("74.39", "70.08", "76.55", "72.17", "93.17", "—", "98.33", "token budget + brevity"),
        "005": ("74.03", "71.65", "69.55", "74.28", "94.5", "—", "98.33", "re-run 004"),
        "006": ("67.08", "64.5", "73.33", "63.25", "66.33", "100.0", "100.0", "v2 stack 4 changes"),
        "007": ("83.22", "80.08", "84.25", "87.67", "91.33", "100.0", "100.0", "filter off + citation fix"),
    }
    for rid, *_ in BASELINE_RUNS:
        row = prog_map.get(rid)
        if row:
            w(lines, f"{rid} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]}")

    # === SECTION 5 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "5. PERSISTENT FAILURES ANALYSIS")
    w(lines, "===================================================================")

    baseline_q_paths = []
    for _, _, folder in BASELINE_RUNS:
        rd = find_run_dir(folder)
        if rd and (rd / "questions.jsonl").exists():
            baseline_q_paths.append(rd / "questions.jsonl")

    failure_ids = ["adaptllm_q5", "queryrewrite_q5", "selfrag_q1", "ragbest_q1", "rgb_q2"]
    for qid in failure_ids:
        w(lines, "")
        w(lines, f"--- {qid} ---")
        scores_across = collect_question_scores_across(baseline_q_paths, qid)
        for folder, sc in scores_across:
            w(lines, f"  {folder}: final_score={sc}")
        if qid == "adaptllm_q5":
            w(lines, """  ANALYSIS (adaptllm_q5):
  Scored 10.0 on every full-pipeline run from 006 onward with qwen judge.
  Recall@5 reports 100% (document found) but passage-level miss.
  Root cause: pypdf extraction broke chunk boundaries in 2404.19705v2.pdf,
  splitting the bottleneck conclusion across chunks. Chunk 4 ends mid-word
  ("even withou"), answer is in chunk 5 but chunk 4 wins retrieval because
  it is semantically similar but truncated.
  Key insight: Recall@5 (document-level) masked a passage-level failure.
  Earlier llama-era runs scored higher (28-67) only because model hallucinated
  from wrong chunks. The 10.0 refusal is correct behavior.""")
        elif qid == "queryrewrite_q5":
            w(lines, "  ANALYSIS: cross-contamination — wrong chunks win retrieval ranking")
        elif qid == "selfrag_q1":
            w(lines, "  ANALYSIS: generation variance on abstract questions; hybrid fixed retrieval in 007")
        elif qid == "ragbest_q1":
            w(lines, "  ANALYSIS: completeness failure — model drops 'repacking' from expected answer")
        elif qid == "rgb_q2":
            w(lines, "  ANALYSIS: omits specific output string required by expected answer")

    # === SECTION 6 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "6. PHASE 2: LOCKED BASELINE VARIANCE (3 runs)")
    w(lines, "===================================================================")
    w(lines, """Locked baseline config (run 007, frozen):
  chunk_size: 256, overlap: 50, top_k: 5
  retriever: hybrid, embedding: BAAI/bge-small-en-v1.5
  reranker: bge, query_transform: none
  prompt: strict_context_json, context_filter: none
  generator: qwen2.5:14b, judge: qwen2.5:14b""")

    locked_scores = []
    for rn in range(1, 4):
        w(lines, "")
        w(lines, f"--- Locked Baseline run_{rn} ---")
        sp = ABLATIONS_DIR / "Locked Baseline" / f"run_{rn}" / "scores.json"
        sc = load_json(sp)
        w(lines, f"Scores: {fmt_scores(sc)}")
        qrows = load_questions(ABLATIONS_DIR / "Locked Baseline" / f"run_{rn}" / "questions.jsonl")
        lines.extend(per_question_block(qrows))
        if sc:
            locked_scores.append(sc)

    if locked_scores:
        w(lines, "")
        w(lines, "Aggregate across 3 locked baseline runs:")
        for metric in ["final_score", "answer_correctness", "faithfulness", "context_recall",
                        "context_precision", "citation_accuracy", "recall_at_k", "mrr_at_k", "avg_latency_s"]:
            vals = [s[metric] for s in locked_scores if metric in s]
            if vals:
                m, sd = mean_std(vals)
                w(lines, f"  {metric}: mean={m} std={sd} min={min(vals)} max={max(vals)}")
        w(lines, "  Interpretation: sampling variance on final_score is ~1.5 points (std)")

    # === SECTION 7 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "7. PHASE 3: BEIR CROSS-DOMAIN GENERALIZATION")
    w(lines, "===================================================================")

    beir_sets = [
        ("BEIR - NFCorpus", "NFCorpus", "Medical/nutrition corpus. Keyword-style web queries."),
        ("BEIR - SciFact", "SciFact", "Scientific claim verification. Claim-style queries."),
        ("BEIR - FiQA", "FiQA", "Financial QA. Conversational forum questions."),
    ]
    for folder, key, desc in beir_sets:
        w(lines, "")
        w(lines, f"--- {key} ---")
        w(lines, f"Description: {desc}")
        w(lines, f"Result folder: {folder}")
        sc = load_json(ABLATIONS_DIR / folder / "scores.json")
        w(lines, f"Scores: {fmt_scores(sc)}")
        qrows = load_questions(ABLATIONS_DIR / folder / "questions.jsonl")
        lines.extend(per_question_block(qrows))

    w(lines, "")
    w(lines, "Cross-domain comparison:")
    w(lines, "Domain       | Format         | Questions | Final | Recall@5")
    w(lines, "RAG papers   | NL questions   | 60        | 83.22 | 100%")
    w(lines, "SciFact      | Claims         | 50        | 56.73 | 78%")
    w(lines, "NFCorpus     | Keywords       | 50        | 32.21 | 10%")
    w(lines, "FiQA         | Conversational | 50        | 45.15 | 34%")
    w(lines, "")
    w(lines, "KEY: NFCorpus keyword queries are a task mismatch (short keyword vs NL questions),")
    w(lines, "not purely a domain failure. Performance degrades with query format distance.")

    # === SECTION 8 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "8. PHASE 4: ABLATION STUDY (A1-A9)")
    w(lines, "===================================================================")

    ablation_winners = []

    for aid, folder, question, var_key in ABLATIONS:
        w(lines, "")
        w(lines, f"=== ABLATION {aid} — {folder} ===")
        w(lines, f"Research question: {question}")
        w(lines, f"Variable changed: {var_key}")
        w(lines, "Everything else: locked baseline (frozen)")
        summary_path = ABLATIONS_DIR / folder / "summary.json"
        summary = load_json(summary_path)
        if not summary or not summary.get("conditions"):
            w(lines, "STATUS: NOT YET RUN")
            continue

        best_cond = None
        best_mean = -1
        w(lines, "")
        w(lines, "Summary table:")
        w(lines, "Condition | Mean | Std | Delta | Run1 | Run2 | Run3")
        for cond, stats in summary["conditions"].items():
            runs = stats.get("runs", [])
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            delta = stats.get("delta_vs_baseline", 0)
            rstr = " | ".join(str(r) for r in runs) if runs else "—"
            w(lines, f"{cond} | {mean} | {std} | {delta:+} | {rstr}")
            if mean > best_mean:
                best_mean = mean
                best_cond = cond

        for cond, stats in summary["conditions"].items():
            w(lines, "")
            w(lines, f"--- Condition: {cond} ---")
            w(lines, f"Config override: {var_key}={cond}")
            for rn in range(1, 4):
                sc = ablation_condition_scores(folder, cond, rn)
                w(lines, f"  Run {rn} scores: {fmt_scores(sc)}")
                qrows = ablation_questions(folder, cond, rn)
                if qrows:
                    w(lines, f"  Run {rn} per-question:")
                    lines.extend(per_question_block(qrows)[1:])  # skip header duplicate

        w(lines, "")
        w(lines, f"Winner (highest mean): {best_cond} ({best_mean})")
        ablation_winners.append((aid, best_cond, best_mean, summary))

        # Interpretation stubs
        if aid == "A1":
            w(lines, "Interpretation: hybrid vs dense statistically tied (~-1 pt vs baseline); no clear winner.")
        elif aid == "A4":
            w(lines, "Interpretation: context filter strongly hurts; top_sentences_5 drops ~14 pts vs baseline.")
        elif aid == "A6":
            w(lines, "Interpretation: reranker contributes ~5 pts; removing it hurts substantially.")
        elif aid == "A8":
            w(lines, "Interpretation: top10 scores ~10 — LIKELY BROKEN RUN (context overflow or prompt failure); treat as invalid.")
        elif aid == "A9":
            w(lines, "Interpretation: chunk 256 optimal; 512 scores ~14 — LIKELY BROKEN RUN (too few/overlarge chunks); treat as invalid.")

    # === SECTION 9 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "9. FINDINGS SUMMARY")
    w(lines, "===================================================================")

    w(lines, """
FINDING 1 — Context filter regression (A4)
  top_sentences_5: mean 68.89 (-14.33 vs baseline)
  top_sentences_10: mean 78.99 (-4.23)
  none: mean 82.85 (-0.37)
  Filtering after reranking destroys ranked context; reranker already surfaces relevance.

FINDING 2 — Hybrid retrieval fixing persistent dense failure
  selfrag_q1: dense failed retrieval in early runs; hybrid + BM25 normalization fixed in 007.
  A1: hybrid 82.19 vs dense 82.06 — tied on aggregate but hybrid fixes specific retrieval failures.

FINDING 3 — Judge model variance
  llama3.1:8b judge (000-005) vs qwen2.5:14b judge (006+): scores not comparable.
  Locked baseline std ~1.52 on final_score with identical config.

FINDING 4 — Recall@5 masking passage-level failures
  adaptllm_q5: 100% recall@5 but 10.0 final score on qwen runs.
  Document-level recall insufficient; passage-level recall needed.

FINDING 5 — Query format sensitivity (BEIR)
  RAG NL 83.22 > SciFact claims 56.73 > FiQA conv 45.15 > NFCorpus keywords 32.21
  Keyword queries are task mismatch not domain failure.

FINDING 6 — Ablation component contributions (valid runs only)
  Largest positive deltas: A5 bge-large (+0.95), A3 xml (+0.88), A8 top5 (+0.51)
  Largest negative deltas: A4 top_sentences_5 (-14.33), A2 llama3.1-8b (-9.49), A6 none (-5.15)
  A7 hyde hurt (-1.8). A1 retriever type negligible (~-1 pt).
  INVALID DATA: A8 top10 (~10), A9 chunk 512 (~14) — investigate before paper claims.""")

    # === SECTION 10 ===
    w(lines, "")
    w(lines, "===================================================================")
    w(lines, "10. PAPER NOTES")
    w(lines, "===================================================================")
    w(lines, """TARGET VENUE: GroundLM 2026 @ EMNLP 2026
  Workshop: Grounding Language Models — Learning Faithfully and Efficiently
  Deadline: June 29, 2026 (direct submission, AoE)
  Format: Short paper 4 pages or long paper 8 pages
  Track: Non-archival option available
  Submit via: OpenReview

FALLBACK VENUE: COLING 2027
  Deadline: ~September 2026

CONTRIBUTION TYPE: Empirical measurement study
  Not a new method — systematic characterization of RAG component interactions
  and failure modes with reproducible cases.

FRAMING:
  "We conduct a controlled ablation of nine RAG pipeline components and document
  underreported failure modes including: post-retrieval filtering negating reranker
  gains, dense retrieval systematically failing on specific query types that BM25
  trivially handles, small LLM judges introducing variance that masks real improvements,
  and document-level recall metrics masking passage-level retrieval failures."

KNOWN WEAKNESSES TO ADDRESS IN PAPER:
  - Judge model changed mid-experiment (runs 000-005 vs 006+)
  - Several early runs stacked multiple changes (especially 006)
  - Single corpus domain until BEIR added
  - N=3 runs has low statistical power for t-tests
  - Questions written by same person who built pipeline
  - A8 top10 and A9 chunk 512 runs show anomalously low scores — likely bugs""")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
