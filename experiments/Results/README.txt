RAG Research — Results
======================

Everything from every run lives here. One folder per run.

STRUCTURE
---------

experiments/Results/
  MASTER_LOG.txt     ← Start here. Compares ALL runs.
  summary.csv        ← Spreadsheet-friendly table of all runs.
  runs/
    001__baseline__2026-06-22_00-05-14_CDT/
      REPORT.txt     ← Human-readable report for THIS run.
      data.json      ← Full raw data.
      questions.jsonl  ← Per-question details (when applicable).

INSIDE EACH RUN FOLDER
----------------------

  REPORT.txt       Read this first.
  data.json        Machine-readable full output.
  questions.jsonl  One line per eval question (experiments + eval runs).

NAMING
------

  {number}__{run_name}__{date}_{time}_CDT

  Examples:
    001__baseline__2026-06-22_00-05-14_CDT
    005__chunk_512__2026-06-21_23-51-04_CDT
    010__build_index__2026-06-22_12-00-00_CDT

RUN TYPES
---------

  experiment           Grid test (chunk size, top-k, retriever, etc.)
  build_index          Indexed documents into Qdrant
  evaluate_retrieval   Baseline retrieval eval (Recall@k, MRR@k)
  generation           Single question answered via Ollama

SCORES
------

  Retrieval only  → higher Recall@k / MRR@k is better
  Full pipeline   → higher final_score is better (0-100)

REGENERATE MASTER LOG
---------------------

  python src/regenerate_experiment_log.py
