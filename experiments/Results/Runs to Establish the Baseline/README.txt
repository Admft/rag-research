RAG Research — Results
======================

Everything from every run lives here. One folder per run.

STRUCTURE
---------

experiments/Results/
  MASTER_LOG.txt           Start here. Compares ALL runs.
  summary.csv              Spreadsheet-friendly table of all runs.
  000-007_FULL_REPORT.txt  Baseline progression report (research write-up).
  runs/
    000__og_baseline__2026-06-22_.../
    001__baseline__2026-06-22_.../
    002__baseline__2026-06-22_.../
      REPORT.txt           Human-readable report for THIS run.
      data.json            Full raw data.
      questions.jsonl      Per-question details.

INSIDE EACH RUN FOLDER
----------------------

  REPORT.txt       Read this first.
  data.json        Machine-readable full output.
  questions.jsonl  One line per eval question (experiments + eval runs).

NAMING
------

  {number}__{run_name}__{date}_{time}_CDT

  Examples:
    000__og_baseline__2026-06-22_23-40-42_CDT   (original baseline)
    001__baseline__2026-06-22_20-28-46_CDT      (grid smoke test)
    004__baseline__2026-06-22_22-51-27_CDT      (best grid baseline)

  Run 000 is reserved for the OG baseline (src/run_og_baseline.py).
  Runs 001+ are from the experiment grid (src/run_experiments.py).

RUN TYPES
---------

  og_baseline          Original first-dev baseline (run 000)
  experiment           Grid test (chunk size, top-k, retriever, etc.)
  build_index          Indexed documents into Qdrant
  evaluate_retrieval   Baseline retrieval eval (Recall@k, MRR@k)
  generation           Single question answered via Ollama

SCORES
------

  Retrieval only  → higher Recall@k / MRR@k is better
  Full pipeline   → higher final_score is better (0-100)

  Full methodology: README.md (Scoring section) and
  000-007_FULL_REPORT.txt (section 3).

  Retrieval (no LLM):
    Recall@k  — expected PDF in top-k? (binary per question, averaged)
    MRR@k     — mean of 1/rank when found, else 0

  Full pipeline (per question, then averaged):
    answer_correctness, faithfulness, context_recall, context_precision
      → LLM judge (Ollama JSON mode, same model as generation)
    citation_accuracy
      → deterministic [Doc X] parse (0 if no citations)
    final_score
      → weighted sum: 35% / 25% / 20% / 10% / 10%

  Also reported: answer_parse_rate, avg latency.
  Recall@k and MRR@k are NOT included in final_score.

BASELINE PROGRESSION (000–007)
-------------------------------

  000  OG baseline — chunk 120/30, no reranker, og_strict prompt
  001  Grid smoke test — retrieval-only, 2 questions
  002  First full grid baseline — chunk 256, BGE reranker, citation prompt
  003  Scratchpad/answer XML tags — citation accuracy 38% → 94%
  004  Token budget + scratchpad brevity — best llama era (74.39)
  005  Re-run of 004 config — confirms ~74 final score
  006  v2 stack — hybrid, JSON, qwen2.5:14b, top_sentences_5 (67.08)
  007  v2 fixes — full context + citation norm — best overall (83.22)

  Details: 000-007_FULL_REPORT.txt

REGENERATE MASTER LOG
---------------------

  python src/regenerate_experiment_log.py
